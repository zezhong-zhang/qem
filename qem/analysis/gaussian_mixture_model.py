import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def safe_ln(x):
    """Natural logarithm function, avoiding division by zero warnings.
    
    Parameters:
        x: The value to take the logarithm of.
        
    Returns:
        The natural logarithm of x.
    """
    x[x < sys.float_info.min] = sys.float_info.min
    return np.log(x)


class GaussianMixtureModel:
    """
    Gaussian Mixture Model for quantitative electron microscopy data analysis.

    This class implements a GMM for fitting scattering cross-section data,
    with support for statistical atom counting and various initialization methods.

    Attributes:
        cross_sections (np.ndarray): The scattering cross-section data.
        electron_dose (float): The number of electrons per pixel.
        fit_result (GmmResult): The result of the GMM fitting.
        selected_data (np.ndarray): The selected cross-section data for fitting.
        data_bounds (np.ndarray): Min and max values of the selected data.
        initial_means (np.ndarray): Initial mean values for GMM components.
        curve_function: The curve function used for 2D GMM fitting.
        curve_parameters: Parameters of the curve function.
        component_range (np.ndarray): Range of component numbers to test.
    """

    def __init__(self, cross_sections: np.ndarray, electron_dose: float = None):
        """Initialize Gaussian Mixture Model.
        
        Args:
            cross_sections: Scattering cross-section data array
            electron_dose: Number of electrons per pixel (for dose-dependent width)
        """
        self.cross_sections = cross_sections
        self.electron_dose = electron_dose
        self.fit_result = None
        self.selected_data = None
        self.data_bounds = None
        self.initial_means = None
        self.curve_function = None
        self.curve_parameters = None
        self.component_range = np.array([], dtype=int)

    def initialize_fitting_conditions(
        self,
        num_components: list[int],
        data_channels,
        optimization_metric: str,
        scoring_methods: list[str],
        initialization_method: str,
        convergence_tolerance: float,
        max_iterations: int,
        initial_weights,
        initial_means,
        initial_widths,
        step_sizes,
        constraints: list[str],
    ):
        """
        Initializes the conditions for the Gaussian Mixture Model.

        Args:
            n_component (int or list): Number of components in the mixture model. If an integer is provided, the range of components will be from 1 to n_component. If a list is provided, the range will be from the first element to the last element of the list.
            use_scs_channel (int or list): The channel(s) to be used for the model. If an integer is provided, only that channel will be used. If a list is provided, the channels specified in the list will be used. If None is provided, the first channel will be used by default.
            metric (str): The metric to be used for the model.
            score_method (list): List of score methods to be used.Available methods: icl: Integrated Completed Likelihood, aic: Akaike Information Criterion, bic: Bayesian Information Criterion, gic: Generalized Information Criterion, clc: Consistent Likelihood Criterion, awe: Akaike's Weighted Estimate, en: Entropy, nllh: Negative Log-Likelihood.
            init_method (str): The initialization method to be used.
            lim_rate (float): The rate of convergence for the model.
            lim_ite (int): The maximum number of iterations for the model.
            given_weight (array-like): The given weights for the model.
            given_mean (array-like): The given means for the model.
            given_width (array-like): The given widths for the model.
            fit_step_size (float): The step size for fitting the model.
            constraint (str): The constraint to be applied to the model.

        Returns:
            None
        """
        # Prepare component range
        if isinstance(num_components, int):
            self.component_range = np.arange(1, num_components + 1)
        elif isinstance(num_components, list):
            if initial_means is None:
                num_components[0] = 1
            self.component_range = np.arange(num_components[0], num_components[-1] + 1)
        
        # Add negative log-likelihood to scoring methods
        if "nllh" not in scoring_methods:
            scoring_methods.append("nllh")
        self.scoring_methods = scoring_methods
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        
        # Determine data channels to use
        if data_channels is None:
            num_channels = self.cross_sections.shape[1] if len(self.cross_sections.shape) > 1 else 1
            if num_channels == 1:
                data_channels = [0]
            elif num_channels == 2:
                data_channels = [0, 1]
            else:
                logging.info("Only support up to 2D GMM, using first channel only\n")
                data_channels = [0]
        elif isinstance(data_channels, int):
            data_channels = [data_channels]
        
        self.active_channels = data_channels
        self.num_dimensions = len(self.active_channels)
        self.selected_data = self.cross_sections[:, self.active_channels]
        self.data_bounds = np.array([self.selected_data.min(0), self.selected_data.max(0)])
        
        # Store fitting parameters
        self.optimization_metric = optimization_metric
        self.initial_weights = initial_weights
        self.initial_means = initial_means
        self.initial_widths = initial_widths
        
        # Set up curve fitting for 2D case
        if self.selected_data.shape[1] == 2:
            self.curve_function = self._polynomial_curve_5th_order
            self.curve_parameters, _ = curve_fit(
                self.curve_function, self.selected_data[:, 0], self.selected_data[:, 1]
            )
        
        self.initialization_method = initialization_method
        self.step_sizes = step_sizes
        self.constraints = constraints

    def fit_gaussian_mixture_model(
        self,
        num_components,
        data_channels=None,
        optimization_metric="icl",
        scoring_methods=None,
        initialization_method="middle",
        convergence_tolerance=1e-5,
        max_iterations=1e5,
        initial_weights=None,
        initial_means=None,
        initial_widths=None,
        step_sizes=None,
        constraints=None,
        use_first_local_minimum=True,
    ):
        """
        Fits a Gaussian Mixture Model (GMM) to the data.

        Args:
            num_components: The number of components in the GMM.
            data_channels: The channel to use for fitting the GMM.
            optimization_metric (str): The metric used for model selection. Default is "icl".
            scoring_methods (list): The scoring method(s) used for model selection. Default is ["icl"].
            initialization_method (str): The initialization method for the GMM. Default is "middle".
            convergence_tolerance (float): The convergence threshold for the optimization. Default is 1e-5.
            max_iterations (float): The maximum number of iterations for the optimization. Default is 1e5.
            initial_weights: The initial weights of the GMM components. Default is None.
            initial_means: The initial means of the GMM components. Default is None.
            initial_widths: The initial widths of the GMM components. Default is None.
            step_sizes (list): The step sizes for fitting the GMM. Default is [1, [1, 1], [1, 1]].
            constraints (list): The constraints applied to the GMM. Default is [].
            use_first_local_minimum (bool): Whether to use first local minimum instead of global minimum.

        Returns:
            None
        """
        # constraint = ['uni_width', 'no_cov', '45deg', 'dose_width']

        if constraints is None:
            constraints = []
        if step_sizes is None:
            step_sizes = [1, [1, 1], [1, 1]]
        if scoring_methods is None:
            scoring_methods = ["icl"]
        
        self.initialize_fitting_conditions(
            num_components,
            data_channels,
            optimization_metric,
            scoring_methods,
            initialization_method,
            convergence_tolerance,
            max_iterations,
            initial_weights,
            initial_means,
            initial_widths,
            step_sizes,
            constraints,
        )
        
        gmm_results = self._initialize_results_dictionary(
            self.component_range, scoring_methods, self.num_dimensions == 2
        )

        # Store weight, mean, width from previous iteration
        previous_parameters = [np.array([]), np.array([[None]]), np.array([[]])]

        for num_comp in tqdm(self.component_range, desc="Fitting GMM components"):
            parameters, scores = self._optimize_single_component_count(
                num_comp, last_means=previous_parameters[1]
            )
            gmm_results["weight"].append(parameters[0])
            gmm_results["mean"].append(parameters[1])
            gmm_results["width"].append(parameters[2])
            
            for score_name, score_value in scores.items():
                gmm_results["score"][score_name].append(score_value)
            
            previous_parameters = parameters

        self.fit_result = GmmResult(
            gmm_results["weight"],
            gmm_results["mean"],
            gmm_results["width"],
            gmm_results["score"],
            self.cross_sections.shape[1] if len(self.cross_sections.shape) > 1 else 1,
            self.selected_data,
            self.curve_parameters,
        )
        
        # Store component selection recommendation
        if use_first_local_minimum:
            self.recommended_components = self._find_first_local_minimum(
                gmm_results["score"][optimization_metric]
            )
        else:
            self.recommended_components = np.argmin(
                gmm_results["score"][optimization_metric]
            ) + 1
        
        logging.info(f"Recommended number of components: {self.recommended_components}")
    
    def _find_first_local_minimum(self, scores):
        """Find the first local minimum in the scores array.
        
        Args:
            scores: Array of scores for different component numbers
            
        Returns:
            int: Component number (1-indexed) corresponding to first local minimum
        """
        scores = np.array(scores)
        
        # Start from component 2 (index 1) to have neighbors
        for i in range(1, len(scores) - 1):
            # Check if this is a local minimum
            if scores[i] < scores[i-1] and scores[i] < scores[i+1]:
                return i + 1  # Convert to 1-indexed
        
        # If no local minimum found, return global minimum
        return np.argmin(scores) + 1
    
    def get_optimal_components(self, method="recommendation"):
        """Get the optimal number of components.
        
        Args:
            method: Method to determine optimal components
                   - "recommendation": Use the recommended value (first local minimum)
                   - "global_min": Use global minimum of scoring metric
                   - "user": Allow user to choose interactively
                   
        Returns:
            int: Optimal number of components
        """
        if not hasattr(self, 'fit_result') or self.fit_result is None:
            raise ValueError("Please run fit_gaussian_mixture_model first")
        
        if method == "recommendation":
            return self.recommended_components
        elif method == "global_min":
            return np.argmin(self.fit_result.score[self.optimization_metric]) + 1
        elif method == "user":
            return self._interactive_component_selection()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _interactive_component_selection(self):
        """Allow user to interactively select the number of components."""
        print(f"\\nRecommended number of components: {self.recommended_components}")
        print("\\nScoring summary:")
        
        for method_name, scores in self.fit_result.score.items():
            min_idx = np.argmin(scores)
            print(f"{method_name.upper()}: minimum at {min_idx + 1} components (score: {scores[min_idx]:.3f})")
        
        while True:
            try:
                user_choice = input(f"\\nEnter number of components (1-{len(self.component_range)}) or 'r' for recommendation: ")
                
                if user_choice.lower() == 'r':
                    return self.recommended_components
                
                choice = int(user_choice)
                if 1 <= choice <= len(self.component_range):
                    return choice
                else:
                    print(f"Please enter a number between 1 and {len(self.component_range)}")
            except ValueError:
                print("Please enter a valid number or 'r' for recommendation")

    def _optimize_single_component_count(self, num_components, last_means):
        """Optimize GMM for a specific number of components.
        
        Args:
            num_components: Number of Gaussian components to fit
            last_means: Mean values from previous component count iteration
            
        Returns:
            tuple: (optimized_parameters, scores_dict)
        """
        candidate_means = self._initialize_means(
            self.initialization_method, last_means, num_components
        )
        
        if self.num_dimensions == 2 and candidate_means[0].shape[1] != 2:
            candidate_means = self._add_second_channel(
                candidate_means, self.curve_function, self.curve_parameters
            )
        
        parameter_candidates = []
        score_candidates = {method: [] for method in self.scoring_methods}
        
        for mean_candidate in candidate_means:
            weights = self._initialize_weights(num_components)
            widths = self._initialize_widths(num_components)
            
            optimized_params, final_scores = self._expectation_maximization(
                weights, mean_candidate, widths
            )
            parameter_candidates.append(optimized_params)
            
            for method_name, score_value in final_scores.items():
                score_candidates[method_name].append(score_value)
        
        # Select best parameters based on optimization metric
        best_index = np.argmin(score_candidates[self.optimization_metric])
        best_scores = {method: scores[best_index] for method, scores in score_candidates.items()}
        
        return parameter_candidates[best_index], best_scores

    def _initialize_weights(self, num_components):
        """Initialize component weights for GMM.
        
        Args:
            num_components: Number of Gaussian components
            
        Returns:
            np.ndarray: Initial weight values
        """
        if self.initial_weights is None:
            return np.ones(num_components) / num_components
        else:
            if isinstance(self.initial_weights, list):
                return self.initial_weights[num_components - 1]
            else:
                return self.initial_weights[:num_components]

    def _initialize_widths(self, num_components):
        """Initialize component widths (variances) for GMM.
        
        Args:
            num_components: Number of Gaussian components
            
        Returns:
            np.ndarray: Initial width values
        """
        default_width = ((self.data_bounds[1] - self.data_bounds[0]) / (2 * num_components)) ** 2
        default_width = np.expand_dims(default_width, axis=0)  # Add component dimension
        
        if self.initial_widths is None:
            return default_width
        else:
            if isinstance(self.initial_widths, list):
                widths = self.initial_widths[num_components - 1]
            else:
                widths = self.initial_widths[:num_components]
            
            # Handle 2D case
            if default_width.shape[2] == 2:
                if widths.shape[0] > 1:
                    default_width = np.repeat(default_width, num_components, axis=0)
                if widths.shape[2] < 2:
                    widths = np.concatenate([widths, [default_width[..., -1]]], axis=-1)
            return widths

    def _initialize_means(self, method, previous_means, num_components):
        """Initialize mean values for GMM components.
        
        Args:
            method: Initialization method ('equionce', 'equimul', 'middle', 'finegrid', 'initvalue')
            previous_means: Mean values from previous iteration
            num_components: Number of components to initialize
            
        Returns:
            list: List of candidate mean initializations
        """
        data_min, data_max = self.data_bounds[0][0], self.data_bounds[1][0]
        
        if num_components == 1 and method != "initvalue":
            mean = np.zeros((1, 1))
            mean[0, 0] = (data_min + data_max) / 2
            return [mean]
        
        # Use only first channel for initialization
        previous_means = np.expand_dims(previous_means[:, 0], -1)
        
        if method == "equionce":
            return [np.expand_dims(
                np.linspace(data_min, data_max, num_components + 1, endpoint=False)[1:], -1
            )]
        
        elif method == "equimul":
            num_trials = 20
            base_means = np.expand_dims(
                np.linspace(data_min, data_max, num_components + 1, endpoint=False)[1:], -1
            )
            delta = (
                (base_means[1] - base_means[0])
                * (np.random.rand(num_trials, base_means.shape[0], base_means.shape[1]) - 0.5)
                / 2
            )
            candidate_means = delta + base_means
            candidate_means = np.clip(candidate_means, data_min, data_max)
            return list(candidate_means)
        
        elif method == "middle":
            boundary_points = np.insert(previous_means, (0, num_components - 1), [data_min, data_max])
            mean_candidates = []
            for i in range(num_components):
                new_mean = (boundary_points[i] + boundary_points[i + 1]) / 2
                mean_candidates.append(np.insert(previous_means, i, new_mean, axis=0))
            return mean_candidates
        
        elif method == "finegrid":
            grid_points = np.linspace(
                data_min, data_max, self.component_range[-1] + 1, endpoint=False
            )[1:]
            mean_candidates = []
            for point in grid_points:
                mean_candidates.append(np.sort(np.insert(previous_means, 0, point, axis=0)))
            return mean_candidates
        
        elif method == "initvalue":
            if isinstance(self.initial_means, list):
                return [self.initial_means[num_components - 1][:, self.active_channels]]
            else:
                return [self.initial_means[:num_components, self.active_channels]]
        
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def _expectation_maximization(self, weights, means, widths):
        """Perform Expectation-Maximization algorithm for GMM fitting.
        
        Args:
            weights: Initial component weights
            means: Initial component means
            widths: Initial component widths/variances
            
        Returns:
            tuple: (optimized_parameters, final_scores)
        """
        gaussian_components = GaussianComponents(
            weights, means, widths, self.selected_data, self.electron_dose
        )
        
        if gaussian_components.has_failed:
            return [weights, means, widths], self._create_failed_scores()
        
        current_log_likelihood = self._calculate_log_likelihood(gaussian_components.component_array)
        convergence_rate = 1
        iteration_count = 0
        
        while (
            convergence_rate > self.convergence_tolerance
            and iteration_count < self.max_iterations
        ):
            gaussian_components.maximization_step(self.step_sizes, self.constraints)
            
            if self._means_have_coincided(gaussian_components.means) or gaussian_components.has_failed:
                break
            
            new_log_likelihood = self._calculate_log_likelihood(gaussian_components.component_array)
            convergence_rate = abs(new_log_likelihood - current_log_likelihood) / abs(current_log_likelihood)
            current_log_likelihood = new_log_likelihood
            iteration_count += 1
        
        if iteration_count == self.max_iterations:
            logging.info("GMM fitting did not converge within maximum iterations\n")
        
        # Sort components by mean values
        final_weights, final_means, final_widths = [
            gaussian_components.weights, gaussian_components.means, gaussian_components.variances
        ]
        sort_indices = np.argsort(final_means[:, 0])
        final_means = np.take_along_axis(final_means, np.expand_dims(sort_indices, -1), axis=0)
        final_weights = np.take_along_axis(final_weights, sort_indices, axis=0)
        
        final_scores = self._calculate_information_criteria(
            gaussian_components.responsibilities, current_log_likelihood
        )
        
        return [final_weights, final_means, final_widths], final_scores

    def import_coordinates(self, coordinate):
        self._coordinates = coordinate
        self._num_column = np.size(coordinate, 1)

    def plot_thickness(self, n_component, show_component=None):
        component_case = n_component - 1
        self.component = self.fit_result.idxComponentOfScs(component_case)
        plt.figure()
        plt.scatter(
            self._coordinates[0, :],
            self._coordinates[1, :],
            marker="o",
            c=self.component,
        )
        ax = plt.gca()
        ax.set_aspect("equal")
        plt.colorbar()
        plt.show(block=False)
        if show_component is not None:
            idx = np.zeros(self._num_column, dtype=bool)
            for c in show_component:
                idx += self.component == c
            x = self._coordinates[0, idx]
            y = self._coordinates[1, idx]
            t = self.component[idx]
            plt.figure()
            plt.scatter(
                self._coordinates[0, :],
                self._coordinates[1, :],
                marker=".",
                c=self.component,
            )
            plt.scatter(x, y, marker="x", c=t)
            ax = plt.gca()
            ax.set_aspect("equal")
            plt.show(block=False)
            plt.pause(1)
        return None

    def plot_criteria(self, criteria=None):
        xaxis = self.n_component_list
        fig, ax = plt.subplots(1, 1)
        for cri in criteria:
            plt.plot(xaxis, self.fit_result.score[cri], label=cri)
            plt.plot(
                np.argmin(self.fit_result.score[cri]) + 1, min(self.fit_result.score[cri]), "o"
            )
        ax.legend(loc="upper center")
        plt.show(block=False)
        plt.pause(1)
        return None

    def plot_histogram(self, n_component: int, use_dim=None, bin=None):
        if use_dim is None or use_dim > self.scs.shape[1]:
            use_dim = self.scs.shape[1]
        if bin is None:
            bin = np.size(self.val, axis=0) // 10

        if use_dim != 2 and use_dim != 1:
            logging.info("only support up to 2 dimensions")
            return
        elif use_dim == 2:
            plt.figure()
            plt.hist2d(self.val[0, :], self.val[1, :], bins=bin)
        elif use_dim == 1:
            plt.figure()
            plt.hist(self.val[0, :], bins=bin)

        if n_component is None:
            min_icl_comp = np.argmin(self.fit_result.score["icl"])
            logging.info(
                f"Number of components is chosen to be {min_icl_comp+ 1} based on ICL.\n"
            )
            component_case = n_component - 1
        else:
            component_case = n_component - 1
            weight = self.fit_result.weight[component_case]
            mean = self.fit_result.mean[component_case]
            width = self.fit_result.width[component_case]
            if use_dim == 1:
                plt.hist(self.val[:, self.channel], bins=bin)
                for c in range(component_case):
                    sigma = width[c] ** 0.5
                    w = weight[c] * self._num_column * sigma
                    mu = mean[c]
                    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, bin)
                    plt.plot(
                        x,
                        w
                        / (sigma * np.sqrt(2 * np.pi))
                        * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                    )
                    plt.text(mu, w / (sigma * np.sqrt(2 * np.pi)) * 1.1, str(c + 1))
            elif use_dim == 2:
                plt.scatter(self.val[:, 0], self.val[:, 1], marker="o", c="b")
                t = np.linspace(-np.pi, np.pi, bin)
                for c in range(component_case):
                    mu = mean[c]
                    sigma = width[c] ** 0.5
                    x = mu[0, 0] + sigma[0, 0] * np.cos(t)
                    y = mu[0, 1] + sigma[0, 1] * np.sin(t)
                    plt.plot(x, y)
                    plt.text(mu[0, 0], mu[0, 1], str(c + 1))
        plt.show(block=False)
        return None

    @staticmethod
    def _means_have_coincided(means):
        """Check if any component means are too close together.
        
        Args:
            means: Array of component mean values
            
        Returns:
            bool: True if means have coincided, False otherwise
        """
        tolerance = 1e-3
        mean_differences = means[1:] - means[:-1]
        coincidence_distances = np.sqrt(np.sum(mean_differences**2, axis=1))
        return (coincidence_distances < tolerance).any()

    @staticmethod
    def _calculate_log_likelihood(component_array):
        """Calculate log-likelihood from component array.
        
        Args:
            component_array: Array of component probability values
            
        Returns:
            float: Log-likelihood value
        """
        return np.log(component_array.sum(0)).sum()

    @staticmethod
    def _calculate_component_array(weights, means, widths, data_values):
        """Calculate component probability array for GMM.
        
        Args:
            weights: Component weights
            means: Component means
            widths: Component widths/variances
            data_values: Input data points
            
        Returns:
            np.ndarray: Component probability array
        """
        normalization_factors = weights * np.prod(widths * 2 * np.pi, axis=-1) ** (-0.5)
        
        if means.shape[1] == 2:
            squared_distances = np.sum(
                (data_values - np.expand_dims(means, 1)) ** 2 
                / np.expand_dims(widths, 1) / 2, axis=-1
            )
        else:
            squared_distances = np.squeeze(
                (data_values - np.expand_dims(means, 1)) ** 2 
                / np.expand_dims(widths, 1) / 2
            )
        
        exponential_terms = np.exp(-squared_distances)
        component_array = normalization_factors.reshape((len(normalization_factors), 1)) * exponential_terms
        return component_array

    def _calculate_information_criteria(self, responsibilities, log_likelihood, scoring_methods=None):
        """Calculate various information criteria for model selection.
        
        Args:
            responsibilities: Component responsibility matrix (tau)
            log_likelihood: Current log-likelihood value
            scoring_methods: List of scoring methods to calculate (optional, uses instance default)
                   - score (dict): A dictionary containing different scores calculated based on the given tau and llh.

        Note on Score Calculation Methods:
            - aic: Akaike Information Criterion
            - gic: Generalized Information Criterion
            - bic: Bayesian Information Criterion
            - clc: Consistent Likelihood Criterion
            - awe: Akaike's Weighted Estimate
            - icl: Integrated Completed Likelihood
            - nllh: Negative Log-Likelihood
            - en: Entropy 
            
        Returns:
            dict: Dictionary of calculated scores
        """
        if scoring_methods is None:
            scoring_methods = self.scoring_methods
        penalty_factor = 2
        num_components, num_data_points = responsibilities.shape
        
        # Calculate number of parameters
        # Weights (minus 1 for constraint that weights sum to 1)
        num_weight_params = (self.step_sizes[0] != 0) * (num_components - 1)
        
        # Means
        num_mean_params = num_components * self.num_dimensions
        
        # Widths/Variances
        num_width_params = [1, num_components]  # [dimensions_per_component, num_components]
        
        if "uni_width" in self.constraints:
            num_width_params[1] = 1  # Uniform width across components
        
        if self.num_dimensions == 2:
            num_width_params[0] = 3  # 2x2 covariance matrix has 3 unique elements
            if ("45deg" in self.constraints) or ("no_cov" in self.constraints):
                num_width_params[0] = 2  # Simplified covariance structure
        
        total_params = num_weight_params + num_mean_params + (num_width_params[0] * num_width_params[1])
        
        # Calculate entropy
        safe_responsibilities = responsibilities * safe_ln(responsibilities)
        safe_responsibilities[responsibilities == 0] = 0
        entropy = -np.sum(safe_responsibilities)
        
        # Define scoring functions
        score_functions = {
            "aic": lambda: -2 * log_likelihood / self.num_dimensions + 2 * total_params,
            "gic": lambda: -2 * log_likelihood / self.num_dimensions + penalty_factor * total_params,
            "bic": lambda: -2 * log_likelihood / self.num_dimensions + total_params * np.log(total_params),
            "clc": lambda: -2 * log_likelihood / self.num_dimensions + 2 * entropy,
            "awe": lambda: (
                -2 * log_likelihood / self.num_dimensions
                + 2 * entropy
                + 2 * total_params * (3/2 + np.log(num_data_points))
            ),
            "icl": lambda: (
                -2 * log_likelihood / self.num_dimensions
                + 2 * entropy
                + total_params * np.log(num_data_points)
            ),
            "nllh": lambda: -log_likelihood,
            "en": lambda: entropy,
        }
        
        calculated_scores = {}
        for method_name in scoring_methods:
            if method_name in score_functions:
                calculated_scores[method_name] = score_functions[method_name]()
            else:
                logging.warning(f"Unknown scoring method: {method_name}. Available methods: {list(score_functions.keys())}")
        
        return calculated_scores
        
    def plot_interactive_gmm_selection(self, cross_sections, element_name="GMM", 
                                      save_results=False, interactive_selection=True):
        """Plot GMM fitting results with interactive component selection and colored areas.
        
        Args:
            cross_sections: Cross-section data used for fitting
            element_name: Name of the element or dataset for labeling
            save_results: Whether to save the plot
            interactive_selection: Whether to allow interactive component selection
            
        Returns:
            int: Selected number of components
        """
        import matplotlib.pyplot as plt
        
        try:
            from matplotlib.widgets import Button, Slider
        except ImportError:
            logging.warning("Interactive widgets not available. Using non-interactive mode.")
            interactive_selection = False
        
        # Create color palette for components
        colors = plt.cm.Set3(np.linspace(0, 1, 12))  # Get 12 distinct colors
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot histogram
        n, bins, patches = ax1.hist(cross_sections.flatten(), bins=50, alpha=0.8, density=True, 
                                   color='lightgray', edgecolor='black', label='Data')
        
        # Store plotting state
        plot_state = {
            'current_components': self.recommended_components,
            'gmm_model': self,
            'cross_sections': cross_sections,
            'ax1': ax1,
            'ax2': ax2,
            'colors': colors,
            'element_name': element_name,
            'plot_closed': False
        }
        
        # Initial plot
        self._update_gmm_plot(plot_state)
        
        # Plot information criteria
        self._plot_information_criteria(plot_state)
        
        if interactive_selection:
            # Add interactive controls
            self._add_interactive_controls(fig, plot_state)
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)  # Make room for slider and controls
            
            def on_close(event):
                plot_state['plot_closed'] = True
            
            fig.canvas.mpl_connect('close_event', on_close)
            
            # Show plot and wait for user interaction
            plt.show(block=False)
            
            # Wait for user to close the plot
            while not plot_state['plot_closed']:
                plt.pause(0.1)
            
            return plot_state['current_components']
        else:
            # Non-interactive mode
            plt.tight_layout()
            
            if save_results:
                filename = f'gmm_analysis_{element_name}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logging.info(f"GMM analysis plot saved as {filename}")
            
            plt.show()
            return self.recommended_components
    
    def _update_gmm_plot(self, plot_state):
        """Update the GMM plot with current component selection."""
        ax1 = plot_state['ax1']
        colors = plot_state['colors']
        current_components = plot_state['current_components']
        cross_sections = plot_state['cross_sections']
        element_name = plot_state['element_name']
        
        # Simple approach: clear everything except the first collection (histogram) and redraw
        # Keep only the histogram (first collection)
        while len(ax1.lines) > 0:
            ax1.lines[0].remove()
        while len(ax1.collections) > 1:
            ax1.collections[-1].remove()
        
        # Get current model parameters
        component_idx = current_components - 1
        weights = self.fit_result.weight[component_idx]
        means = self.fit_result.mean[component_idx]
        widths = self.fit_result.width[component_idx]
        
        # Use cached x_range if available, otherwise create and cache it
        if 'x_range' not in plot_state:
            plot_state['x_range'] = np.linspace(cross_sections.min(), cross_sections.max(), 300)
        x_range = plot_state['x_range']
        
        mixture_pdf = np.zeros_like(x_range)
        
        # Plot each component
        for i, (w, m, var) in enumerate(zip(weights.flatten(), means.flatten(), widths.flatten())):
            std = np.sqrt(var)
            component_pdf = w * (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - m) / std) ** 2)
            mixture_pdf += component_pdf
            
            # Plot component curve
            color = colors[i % len(colors)]
            ax1.plot(x_range, component_pdf, '--', color=color, linewidth=2, alpha=0.8, 
                    label=f'Component {i+1} (N={i+1})')
            
            # Fill area under curve with transparent color
            ax1.fill_between(x_range, 0, component_pdf, color=color, alpha=0.8)
        
        # Plot mixture
        ax1.plot(x_range, mixture_pdf, 'r-', linewidth=3, label='GMM Fit')
        
        # Get the handles and labels from the plot
        legend_result = ax1.get_legend_handles_labels()
        if len(legend_result) == 2:
            handles, labels = legend_result
        else:
            handles, labels = [], []
        
        items_per_column = 5 # Decide how many items you want per column

        # Calculate the number of columns needed
        num_columns = np.ceil(len(handles) / items_per_column) if handles else 1
        
        # Set labels and title
        ax1.set_xlabel('Scattering Cross-Section')
        ax1.set_ylabel('Probability Density')
        ax1.legend(loc='upper right', ncol=num_columns, fontsize='small')
        ax1.grid(True, alpha=0.8)
        
        # Update title to show current selection
        recommended_text = " (Recommended)" if current_components == self.recommended_components else ""
        ax1.set_title(f'GMM Fitting - {element_name} ({current_components} components{recommended_text})')
        
        # Don't call draw here - let the main callback handle it
    
    def _plot_information_criteria(self, plot_state):
        """Plot information criteria for model selection."""
        ax2 = plot_state['ax2']
        current_components = plot_state['current_components']
        
        # Clear and redraw the whole plot each time for simplicity
        ax2.clear()
        
        scores = self.fit_result.score
        component_range = np.arange(1, len(scores['icl']) + 1)
        
        # Plot the curves
        ax2.plot(component_range, scores['icl'], 'o-', linewidth=2, markersize=8, label='ICL', color='blue')
        if 'aic' in scores:
            ax2.plot(component_range, scores['aic'], 's-', linewidth=2, markersize=6, label='AIC', color='green')
        if 'bic' in scores:
            ax2.plot(component_range, scores['bic'], '^-', linewidth=2, markersize=6, label='BIC', color='orange')
        
        # Highlight current selection
        current_score = scores['icl'][current_components - 1]
        ax2.plot(current_components, current_score, 'ro', label='Your Choice',markersize=15, 
                markerfacecolor='lightgreen', markeredgecolor='darkred', markeredgewidth=3, zorder=10)
        
        # Highlight recommended selection  
        recommended_score = scores['icl'][self.recommended_components - 1]
        ax2.plot(self.recommended_components, recommended_score, 'go', label='First Local Minimum', markersize=12,
                markerfacecolor='red', markeredgecolor='darkgreen', markeredgewidth=2, zorder=9)
        
        # Add vertical line for current selection
        ax2.axvline(self.recommended_components, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        # Set labels and formatting
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Information Criterion')
        ax2.set_title('Model Selection')
        ax2.legend()
        ax2.grid(True, alpha=0.8)
        
        # Don't call draw here - let the main callback handle it
    
    def _add_interactive_controls(self, fig, plot_state):
        """Add interactive controls to the plot.""" 
        try:
            from matplotlib.widgets import Button, Slider
        except ImportError:
            logging.warning("Interactive widgets not available.")
            return
            
        # Create component selection slider
        max_components = len(self.fit_result.score['icl'])
        
        # Add slider for component selection
        ax_slider = plt.axes([0.2, 0.02, 0.5, 0.03])
        slider = Slider(
            ax_slider, 
            'Components', 
            1, 
            max_components, 
            valinit=plot_state['current_components'],
            valstep=1,
            valfmt='%d'
        )
        
        # Add debugging and ensure slider works
        def on_component_change(val):
            new_components = int(val)
            print(f"Slider changed to: {new_components}")  # Debug print
            if new_components != plot_state['current_components']:
                plot_state['current_components'] = new_components
                print(f"Updating plots for {new_components} components")  # Debug print
                try:
                    # Update both plots
                    self._update_gmm_plot(plot_state)
                    self._plot_information_criteria(plot_state)
                    
                    # Single draw call at the end
                    fig.canvas.draw()
                    print("Plot update completed")  # Debug print
                except Exception as e:
                    print(f"Error updating plots: {e}")  # Debug print
                    import traceback
                    traceback.print_exc()
        
        slider.on_changed(on_component_change)
        
        # Store slider reference in plot_state so it doesn't get garbage collected
        plot_state['slider'] = slider
        
        # # Add accept button
        # ax_accept = plt.axes([0.85, 0.02, 0.1, 0.05])
        # accept_button = Button(ax_accept, 'Accept', color='lightgreen', hovercolor='green')
        
        # def on_accept(event):
        #     plt.close(fig)
        
        # accept_button.on_clicked(on_accept)
        
        # # Add reset to recommendation button
        # ax_reset = plt.axes([0.02, 0.02, 0.15, 0.05])
        # reset_button = Button(ax_reset, 'Use Recommended', color='lightyellow', hovercolor='yellow')
        
        # def on_reset(event):
        #     slider.reset()
        #     slider.set_val(self.recommended_components)
        #     plot_state['current_components'] = self.recommended_components
        #     self._update_gmm_plot(plot_state)
        #     self._plot_information_criteria(plot_state)
        
        # reset_button.on_clicked(on_reset)
        
        # Add text showing recommendation
        fig.text(0.5, 0.08, f'Recommended: {self.recommended_components} components (first local minimum)', 
                ha='center', va='bottom', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    def _create_failed_scores(self):
        """Create scores dictionary for failed fitting attempts.
        
        Returns:
            dict: Dictionary with infinite scores for all methods
        """
        return {method: np.inf for method in self.scoring_methods}

    def apply_parameter_constraints(self, weights, means, widths):
        """Apply constraints to GMM parameters.
        
        Args:
            weights: Component weights
            means: Component means  
            widths: Component widths/covariances
            
        Returns:
            tuple: (constrained_weights, constrained_means, constrained_widths)
        """
        if ("no_cov" in self.constraints) and self.num_dimensions == 2:
            # Remove off-diagonal covariance terms
            constraint_mask = np.zeros(widths.shape)
            for dim_idx in range(self.num_dimensions):
                constraint_mask[:, dim_idx, dim_idx] = 1
            widths = widths * constraint_mask
        
        if ("45deg" in self.constraints) and self.num_dimensions == 2:
            # Force diagonal covariance with equal variances
            max_variance = np.max([widths[:, 0, 0], widths[:, 1, 1]])
            for dim_idx in range(self.num_dimensions):
                widths[:, dim_idx, dim_idx] = max_variance
        
        return weights, means, widths

    @staticmethod
    def _add_second_channel(mean_candidates, curve_function, curve_parameters):
        """Add second channel values using curve fitting for 2D GMM.
        
        Args:
            mean_candidates: List of 1D mean arrays
            curve_function: Function to calculate second channel values
            curve_parameters: Parameters for the curve function
            
        Returns:
            list: Mean candidates with second channel added
        """
        return [
            np.concatenate(
                (candidate, curve_function(candidate, *curve_parameters)), axis=-1
            )
            for candidate in mean_candidates
        ]

    @staticmethod
    def _initialize_results_dictionary(component_range: np.ndarray, scoring_methods: list, is_2d: bool):
        """Initialize results dictionary for GMM fitting.
        
        Args:
            component_range: Array of component numbers to test
            scoring_methods: List of scoring method names
            is_2d: Whether fitting 2D data
            
        Returns:
            dict: Initialized results dictionary
        """
        score_dict = {method: [] for method in scoring_methods}
        return {
            "weight": [],
            "mean": [],
            "width": [],
            "score": score_dict,
            "scsidx": [],
            "case": component_range,
        }

    # Polynomial curve functions for 2D fitting
    @staticmethod
    def _exponential_curve(x, a, b, c):
        """Exponential curve function."""
        return x**a * b + c

    @staticmethod
    def _polynomial_curve_1st_order(x, a, b):
        """First-order polynomial curve."""
        return x * a + b

    @staticmethod
    def _polynomial_curve_2nd_order(x, a, b, c):
        """Second-order polynomial curve."""
        return x**2 * a + x * b + c

    @staticmethod
    def _polynomial_curve_3rd_order(x, a, b, c, d):
        """Third-order polynomial curve."""
        return x**3 * a + x**2 * b + x * c + d

    @staticmethod
    def _polynomial_curve_4th_order(x, a, b, c, d, e):
        """Fourth-order polynomial curve."""
        return x**4 * a + x**3 * b + x**2 * c + x * d + e

    @staticmethod
    def _polynomial_curve_5th_order(x, a, b, c, d, e, f):
        """Fifth-order polynomial curve."""
        return x**5 * a + x**4 * b + x**3 * c + x**2 * d + x * e + f

    @staticmethod
    def _polynomial_curve_6th_order(x, a, b, c, d, e, f, g):
        """Sixth-order polynomial curve."""
        return x**6 * a + x**5 * b + x**4 * c + x**3 * d + x**2 * e + x * f + g


class GmmResult:
    def __init__(
        self,
        weight: list,
        mean: list,
        width: list,
        score: dict,
        ndim: int,
        val: np.ndarray,
        curve=None,
    ):
        """
        Initialize a GaussianMixtureModel result.

        Args:
            weight (list): The weights of the Gaussian components.
            mean (list): The means of the Gaussian components.
            width (list): The widths of the Gaussian components.
            score (dict): A dictionary containing scores for each component.
            ndim (int): The number of dimensions of the data.
            val (np.ndarray): The input data values.
            curve (Optional): An optional curve parameter.

        Returns:
            None
        """
        self.weight = weight
        self.mean = mean
        self.width = width
        self.score = score
        self.curve = curve
        self.ndim = ndim
        self.val = val

    def idxComponentOfScs(self, id):
        """
        Returns the index of the component in the Gaussian Mixture Model (GMM) that the given id corresponds to.

        Parameters:
        id (int): The id of the scs (sub-component set).

        Returns:
        int: The index of the component in the GMM.

        """
        # point each scs to a specific component
        g = GaussianComponents(self.weight[id], self.mean[id], self.width[id], self.val)
        g._calculate_component_probabilities()
        return np.argmax(g.component_array, 0)

    def idxScsOfComponent(self, id):
        """
        Returns the indices of the scs belonging to a specific component.

        Parameters:
        id (int): The component ID.

        Returns:
        list: A list of indices representing the samples belonging to the specified component.
        """
        # list scs under each component
        idx_c = self.idxComponentOfScs(id)
        idx_s = []
        for c in range(id + 1):
            idx_s.append(np.argwhere(idx_c == c))
        return idx_s


class GaussianComponents:
    """Represents individual Gaussian components in a mixture model.
    
    This class handles the expectation and maximization steps for
    individual Gaussian components during GMM fitting.
    """
    
    def __init__(self, weights, means, variances, data_values, electron_dose=None):
        """Initialize Gaussian components.
        
        Args:
            weights: Component weights
            means: Component means
            variances: Component variances
            data_values: Observed data points
            electron_dose: Electron dose for dose-dependent width modeling
        """
        self.weights = weights
        self.means = means
        self.variances = variances
        self.data_values = data_values
        self.num_components = self.weights.shape[0]
        self.num_data_points, self.num_dimensions = self.data_values.shape
        self.electron_dose = electron_dose
        self.has_failed = False
        
        self._responsibilities = None
        self._responsibility_sums = None
        self._expanded_responsibilities = None
        self._expanded_responsibility_sums = None
        
        self._calculate_component_probabilities()

    def _calculate_component_probabilities(self):
        """Calculate probability contributions from each component."""
        expanded_variances = np.expand_dims(self.variances, axis=1)
        expanded_means = np.expand_dims(self.means, 1)
        expanded_weights = np.expand_dims(self.weights, 1)
        
        component_probabilities = self._compute_gaussian_probabilities(
            expanded_variances, expanded_means, expanded_weights, self.data_values
        )
        self._perform_expectation_step(component_probabilities)
        
        if not self.has_failed:
            self.component_array = component_probabilities

    @staticmethod
    def _compute_gaussian_probabilities(variances, means, weights, data_values):
        """Compute Gaussian probability density values.
        
        Args:
            variances: Component variances (expanded for broadcasting)
            means: Component means (expanded for broadcasting)
            weights: Component weights (expanded for broadcasting)
            data_values: Input data points
            
        Returns:
            np.ndarray: Gaussian probability values
        """
        num_dimensions = means.shape[-1]
        squared_distances = (data_values - means) ** 2
        
        if (variances == 0).any():
            logging.warning("Zero variance detected in Gaussian components")
        
        gaussian_probabilities = (
            (2 * np.pi) ** (-num_dimensions / 2)
            * np.prod(variances, axis=-1) ** (-0.5)
            * np.exp(-0.5 * (squared_distances / variances).sum(axis=-1))
        )
        
        return gaussian_probabilities * weights

    @property
    def responsibilities(self):
        """Get component responsibilities (tau values)."""
        return self._responsibilities

    @responsibilities.setter
    def responsibilities(self, tau_values):
        """Set component responsibilities and compute derived quantities."""
        self._responsibilities = tau_values
        self._responsibility_sums = tau_values.sum(1) + np.finfo("double").eps
        self._expanded_responsibilities = np.expand_dims(tau_values, -1)
        self._expanded_responsibility_sums = self._expanded_responsibilities.sum(1) + np.finfo("double").eps

    def _perform_expectation_step(self, component_probabilities):
        """Perform E-step: calculate component responsibilities.
        
        Args:
            component_probabilities: Probability values from each component
        """
        total_probabilities = np.sum(component_probabilities, axis=0)
        self.responsibilities = component_probabilities / total_probabilities
        
        # Check for degenerate components (very low responsibility)
        if (self._responsibility_sums < 1).any():
            self.has_failed = True

    def pre_maximization_step(self, step_sizes, constraints):
        """Perform preliminary M-step operations.
        
        Args:
            step_sizes: Step sizes for parameter updates
            constraints: List of constraint names to apply
        """
        self._update_weights(step_sizes[0])
        self._calculate_component_probabilities()
        self._update_variances(step_sizes[2], constraints)
        self._calculate_component_probabilities()

    def maximization_step(self, step_sizes, constraints):
        """Perform M-step: update all parameters.
        
        Args:
            step_sizes: Step sizes for parameter updates
            constraints: List of constraint names to apply
        """
        self._update_weights(step_sizes[0])
        self._update_means(step_sizes[1])
        self._update_variances(step_sizes[2], constraints)
        self._calculate_component_probabilities()

    def _update_weights(self, step_size):
        """Update component weights using EM update rule.
        
        Args:
            step_size: Step size for weight updates
        """
        new_weights = self._responsibility_sums / self.num_data_points
        self.weights = (new_weights - self.weights) * step_size + self.weights
        # Ensure weights sum to 1 by adjusting the last component
        self.weights[-1] = 1 - np.sum(self.weights[:-1])

    def _update_means(self, step_sizes):
        """Update component means using EM update rule.
        
        Args:
            step_sizes: Step sizes for mean updates (per dimension)
        """
        weighted_data = (self._expanded_responsibilities * self.data_values).sum(1)
        new_means = weighted_data / self._expanded_responsibility_sums
        
        # Apply step sizes (may be different per dimension)
        step_array = np.array(step_sizes[:self.num_dimensions])
        self.means = (new_means - self.means) * step_array + self.means

    def _update_variances(self, step_sizes, constraints):
        """Update component variances using EM update rule.
        
        Args:
            step_sizes: Step sizes for variance updates (per dimension)
            constraints: List of constraint names to apply
        """
        expanded_means = np.expand_dims(self.means, 1)
        expanded_variances = np.expand_dims(self.variances, 1)
        squared_distances = (self.data_values - expanded_means) ** 2
        
        if "uni_width" in constraints:
            if "dose_width" in constraints:
                dose_variances = expanded_means / self.electron_dose
                individual_variance = (
                    (self._expanded_responsibilities * squared_distances - dose_variances) / expanded_variances**2
                ).sum((0, 1)) / (self._expanded_responsibilities / expanded_variances**2).sum((0, 1))
                individual_variance[individual_variance < 0] = 0
                new_variances = dose_variances.sum(1) + np.expand_dims(individual_variance, 0)
                
            elif "dose_width_simplified" in constraints:
                dose_variances = expanded_means / self.electron_dose
                individual_variance = (
                    self._expanded_responsibilities * squared_distances - dose_variances
                ).sum((0, 1)) / self.num_data_points
                individual_variance[individual_variance < 0] = 0
                new_variances = dose_variances.sum(1) + np.expand_dims(individual_variance, 0)
                
            elif "dose_width_fit" in constraints:
                dose_variances = expanded_means / self.electron_dose
                individual_variance = (expanded_variances - dose_variances).mean((0, 1))
                individual_variance[individual_variance < np.finfo("double").eps] = np.finfo("double").eps
                
                # For now, use simplified approach - full optimization would require minimize import
                new_variances = dose_variances.sum(1) + np.expand_dims(individual_variance, 0)
            else:
                new_variances = np.expand_dims(
                    (self._expanded_responsibilities * squared_distances).sum((0, 1)) / self.num_data_points, 0
                )
        else:
            new_variances = (
                self._expanded_responsibilities * squared_distances
            ).sum(1) / self._expanded_responsibility_sums
        
        # Apply step sizes (may be different per dimension) 
        step_array = np.array(step_sizes[:self.num_dimensions])
        self.variances = (new_variances - self.variances) * step_array + self.variances

    @staticmethod
    def rectifier(val):
        return np.exp(val)

    @staticmethod
    def rev_rectifier(val):
        return np.log(val)

    def __fit_dose_indi_var__(self, var_indi, *args):
        """
        Fits the dose independent variance by minimizing the negative log-likelihood.

        Args:
            var_indi: The dose-independent variance.
            *args: Additional arguments including var_dose, mean, weight, and val.

        Returns:
            The negative log-likelihood value.

        """
        var_dose, mean, weight, val = args
        nllh = -np.log(
            self._ca(self.rectifier(var_indi) + var_dose, mean, weight, val).sum(0)
        ).sum()
        return nllh
