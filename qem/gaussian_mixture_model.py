import numpy as np
from scipy.optimize import curve_fit, minimize
from tqdm import tqdm
from qem.utils import safe_ln
import matplotlib.pyplot as plt


class GaussianMixtureModel:
    """
    Represents a Gaussian Mixture Model.

    Attributes:
        scs (np.ndarray): The scattering cross-section data.
        electron_per_px: The number of electrons per pixel.
        result (dict): The result of the GMM fitting.
        val (np.ndarray): The selected cross-section for GMM fitting.
        minmax (np.ndarray): The minimum and maximum values of the selected data.
        init_mean (np.ndarray): The initial means for GMM fitting.
        curve: The curve used for GMM fitting.
        curve_prmt: The parameters of the curve used for GMM fitting.
    """

    def __init__(self, scs: np.ndarray, electron_per_px=None):
        self.scs = scs
        self.dose = electron_per_px
        self.result = GmmResult
        self.val: np.ndarray
        self.minmax: np.ndarray
        self.init_mean: np.ndarray
        self.curve = None
        self.curve_prmt = None
        self.n_component_list = np.ndarray([], dtype=int)

    def initCondition(
        self,
        n_component: list[int],
        use_scs_channel,
        metric,
        score_method,
        init_method,
        lim_rate,
        lim_ite,
        given_weight,
        given_mean,
        given_width,
        fit_step_size,
        constraint,
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
        # prepare data
        if isinstance(n_component, int):
            self.n_component_list = np.arange(1, n_component + 1)
        elif isinstance(n_component, list):
            if given_mean is None:
                n_component[0] = 1
            self.n_component_list = np.arange(n_component[0], n_component[-1] + 1)
        score_method.append("nllh")
        self.score_method = score_method
        self.lim_rate = lim_rate
        self.lim_ite = lim_ite
        if use_scs_channel is None:
            if np.size(self.scs, 1) == 1:
                use_scs_channel = [0]
            elif np.size(self.scs, 1) == 2:
                use_scs_channel = [0, 1]
            else:
                print("only support up to 2D GMM, only use first channel for now\n")
                use_scs_channel = [0]
        elif isinstance(use_scs_channel, int):
            use_scs_channel = [use_scs_channel]
        self.channel = use_scs_channel
        self.n_dim = len(self.channel)
        self.val = self.scs[:, self.channel]
        self.minmax = np.array([self.val.min(0), self.val.max(0)])
        self.metric = metric
        self.given_weight = given_weight
        self.given_mean = given_mean
        self.given_width = given_width
        if np.size(self.val, 1) == 2:
            self.fit_method = self.polyCurve_5
            self.curve_prmt, _ = curve_fit(
                self.fit_method, self.val[:, 0], self.val[:, 1]
            )
        self.init_method = init_method
        self.step = fit_step_size
        self.constraint = constraint

    def GMM(
        self,
        n_component,
        use_scs_channel=None,
        metric="nllh",
        score_method=["icl"],
        init_method="middle",
        lim_rate=1e-5,
        lim_ite=1e5,
        given_weight=None,
        given_mean=None,
        given_width=None,
        fit_step_size=[1, [1, 1], [1, 1]],
        constraint=[],
    ):
        """
        Fits a Gaussian Mixture Model (GMM) to the data.

        Args:
            n_component: The number of components in the GMM.
            use_scs_channel: The channel to use for fitting the GMM.
            metric (str): The metric used for model selection. Default is "nllh".
            score_method (list): The scoring method(s) used for model selection. Default is ["icl"]. icl: Integrated Completed Likelihood, aic: Akaike Information Criterion, bic: Bayesian Information Criterion, gic: Generalized Information Criterion, clc: Consistent Likelihood Criterion, awe: Akaike's Weighted Estimate, en: Entropy, nllh: Negative Log-Likelihood.
            init_method (str): The initialization method for the GMM. Default is "middle".
            lim_rate (float): The convergence threshold for the optimization. Default is 1e-5.
            lim_ite (float): The maximum number of iterations for the optimization. Default is 1e5.
            given_weight: The initial weights of the GMM components. Default is None.
            given_mean: The initial means of the GMM components. Default is None.
            given_width: The initial widths of the GMM components. Default is None.
            fit_step_size (list): The step sizes for fitting the GMM. Default is [1, [1, 1], [1, 1]].
            constraint (list): The constraints applied to the GMM. Default is []. Possible constraints: uni_width, dose_width, dose_width_simplified, dose_width_fit

        Returns:
            None
        """
        # constraint = ['uni_width', 'no_cov', '45deg', 'dose_width']

        self.initCondition(
            n_component,
            use_scs_channel,
            metric,
            score_method,
            init_method,
            lim_rate,
            lim_ite,
            given_weight,
            given_mean,
            given_width,
            fit_step_size,
            constraint,
        )
        gmm_result = self.initResultDict(
            self.n_component_list, score_method, len(self.channel) == 2
        )

        wmw = [np.array([]), np.array([[None]]), np.array([[]])]

        for n in tqdm(self.n_component_list):
            wmw, score = self.optimize(n, last_mean=wmw[1])
            gmm_result["weight"].append(wmw[0])
            gmm_result["mean"].append(wmw[1])  # the variance of the gaussian
            gmm_result["width"].append(wmw[2])
            for key in score:
                gmm_result["score"][key].append(score[key])

        self.result = GmmResult(
            gmm_result["weight"],
            gmm_result["mean"],
            gmm_result["width"],
            gmm_result["score"],
            np.size(self.scs, 1),
            self.val,
            self.curve_prmt,
        )

    def optimize(self, n_component, last_mean):
        """
        Performs Expectation-Maximization (EM) optimization for the Gaussian Mixture Model.

        Args:
            n_component (int): The number of components in the Gaussian Mixture Model.
            last_mean (list): The mean values of the last iteration.

        Returns:
            tuple: A tuple containing the optimized weights/mean/width and the minimum score.

        """
        mean_list = self.initMean(self.init_method, last_mean, n_component)
        if self.n_dim == 2 and not np.size(mean_list[0], 1) == 2:
            mean_list = self.addChannel(mean_list, self.fit_method, self.curve_prmt)
        wmw_list = []
        score_list = {key: [] for key in self.score_method}
        for mean in mean_list:
            weight = self.initWeight(n_component)
            width = self.initWidth(n_component)
            wmw, score = self.EM(weight, mean, width)
            wmw_list.append(wmw)
            for key in score:
                score_list[key].append(score[key])
        idx = np.argmin(score_list[self.metric])
        min_score = {key: score_list[key][idx] for key in score}
        return wmw_list[idx], min_score

    def initWeight(self, n_component):
        """
        Initializes the weights for the Gaussian mixture model.

        Parameters:
        - n_component (int): The number of components in the Gaussian mixture model.

        Returns:
        - weight (ndarray): An array of weights for each component.

        If `given_weight` is None, the weights are initialized as equal weights.
        If `given_weight` is a list, the weight for the `n_component`-th component is used.
        If `given_weight` is an ndarray, the first `n_component` weights are used.
        """
        if self.given_weight is None:
            weight = np.ones(n_component) / n_component
        else:
            if isinstance(self.given_weight, list):
                weight = self.given_weight[n_component - 1]
            else:
                weight = self.given_weight[:n_component]
        return weight

    def initWidth(self, n_component):
        """
        Initializes the width of the Gaussian components in the Gaussian Mixture Model.

        Parameters:
            n_component (int): The number of Gaussian components.

        Returns:
            numpy.ndarray: The initialized width of the Gaussian components.

        """
        width_temp = ((self.minmax[1] - self.minmax[0]) / (2 * n_component)) ** 2
        width_temp = np.expand_dims(
            width_temp, axis=0
        )  # give common axis for component dimension
        if self.given_width is None:
            return width_temp
        else:
            if isinstance(self.given_width, list):
                width = self.given_width[n_component - 1]
            else:
                width = self.given_width[:n_component]
            if width_temp.shape[2] == 2:
                if width.shape[0] > 1:
                    width_temp = np.repeat(width_temp, n_component, axis=0)
                if width.shape[2] < 2:
                    width = np.concatenate([width, [width_temp[..., -1]]], axis=-1)
            return width

    def initMean(self, init_method, last_mean, n_component):
        """
        Initialize the mean values for the Gaussian Mixture Model.

        Parameters:
        - init_method (str): The initialization method to use.
        - last_mean (numpy.ndarray): The mean values from the previous iteration.
        - n_component (int): The number of components in the Gaussian Mixture Model.

        Returns:
        - mean_list (list): A list of mean values for each component.

        """
        minmax = [self.minmax[0][0], self.minmax[1][0]]
        if n_component == 1 and self.init_method != "initvalue":
            mean = np.zeros((1, 1))
            mean[0, 0] = (minmax[0] + minmax[1]) / 2
            mean_list = [mean]
        else:
            last_mean = np.expand_dims(
                last_mean[:, 0], -1
            )  # only use the first channel for initialization

            # method 1
            if init_method == "equionce":
                mean_list = [
                    np.expand_dims(
                        np.linspace(
                            minmax[0], minmax[1], n_component + 1, endpoint=False
                        )[1:],
                        -1,
                    )
                ]

            # method 2
            if init_method == "equimul":
                repeat = 20
                mean_0 = np.expand_dims(
                    np.linspace(minmax[0], minmax[1], n_component + 1, endpoint=False)[
                        1:
                    ],
                    -1,
                )
                delta = (
                    (mean_0[1] - mean_0[0])
                    * (np.random.rand(repeat, mean_0.shape[0], mean_0.shape[1]) - 0.5)
                    / 2
                )
                mean = delta + mean_0
                mean[mean < minmax[0]] = minmax[0]
                mean[mean > minmax[1]] = minmax[1]
                mean_list = list(mean)

            # method 3
            if init_method == "middle":
                points = np.insert(last_mean, (0, n_component - 1), minmax)
                mean_list = []
                for n in range(n_component):
                    new_point = (points[n] + points[n + 1]) / 2
                    mean_list.append(np.insert(last_mean, n, new_point, axis=0))

            # method 4
            if init_method == "finegrid":
                points = np.linspace(
                    minmax[0], minmax[1], self.n_component_list[-1] + 1, endpoint=False
                )[1:]
                mean_list = []
                for p in points:
                    mean_list.append(np.sort(np.insert(last_mean, 0, p, axis=0)))

            # method 5
            if init_method == "initvalue":
                if isinstance(self.given_mean, list):
                    mean_list = [self.given_mean[n_component - 1, self.channel]]
                else:
                    mean_list = [self.given_mean[:n_component, self.channel]]
        return mean_list

    def EM(self, weight, mean, width):
        """
        Performs the Expectation-Maximization (EM) algorithm for fitting a Gaussian Mixture Model.

        Args:
            weight (ndarray): The initial weights of the Gaussian components.
            mean (ndarray): The initial means of the Gaussian components.
            width (ndarray): The initial widths (variances) of the Gaussian components.

        Returns:
            tuple: A tuple containing the updated weights, means, and widths of the Gaussian components,
                   and the score of the fitting.

        Raises:
            None

        """
        g = GaussianComponents(weight, mean, width, self.val, self.dose)
        if g.b_fail:
            return [weight, mean, width], self.failedScore()
        llh = self.logLikelihood(g.ca)
        rate = 1
        cnt = 0
        # g.preMSTep(self.step, self.constraint)
        while (rate > self.lim_rate) and (cnt < self.lim_ite):
            # g.EStep()
            g.MStep(self.step, self.constraint)
            if self.meanCoincide(g.mean) or g.b_fail:
                break
            llh_new = self.logLikelihood(g.ca)
            rate = abs(llh_new - llh) / abs(llh)
            llh = llh_new
            cnt += 1
        if cnt == self.lim_ite:
            print("fitting did not converge\n")
        weight, mean, width = [g.weight, g.mean, g.var]
        sort_idx = np.argsort(mean[:, 0])
        mean = np.take_along_axis(mean, np.expand_dims(sort_idx, -1), axis=0)
        weight = np.take_along_axis(weight, sort_idx, axis=0)
        score = self.calculateScore(g.tau, llh)
        return [weight, mean, width], score

    def import_coordinates(self, coordinate):
        self._coordinates = coordinate
        self._num_column = np.size(coordinate, 1)

    def plot_thickness(self, n_component, show_component=None):
        component_case = n_component - 1
        self.component = self.result.idxComponentOfScs(component_case)
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
            plt.plot(xaxis, self.result.score[cri], label=cri)
            plt.plot(
                np.argmin(self.result.score[cri]) + 1, min(self.result.score[cri]), "o"
            )
        legend = ax.legend(loc="upper center")
        plt.show(block=False)
        plt.pause(1)
        return None

    def plot_histogram(self, n_component: int, use_dim=None, bin=None):
        if use_dim is None or use_dim > self.scs.shape[1]:
            use_dim = self.scs.shape[1]
        if bin is None:
            bin = np.size(self.val, axis=0) // 10

        if use_dim != 2 and use_dim != 1:
            print("only support up to 2 dimensions")
            return
        elif use_dim == 2:
            plt.figure()
            plt.hist2d(self.val[0, :], self.val[1, :], bins=bin)
        elif use_dim == 1:
            plt.figure()
            plt.hist(self.val[0, :], bins=bin)

        if n_component is None:
            min_icl_comp = np.argmin(self.result.score["icl"])
            print(
                "Number of components is chosen to be ",
                min_icl_comp + 1,
                "based on ICL.\n",
            )
            component_case = n_component - 1
        else:
            component_case = n_component - 1
            weight = self.result.weight[component_case]
            mean = self.result.mean[component_case]
            width = self.result.width[component_case]
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
    def meanCoincide(mean):
        """
        Check if the means of a Gaussian mixture model coincide.

        Args:
            mean (ndarray): Array of means for each component of the Gaussian mixture model.

        Returns:
            bool: True if the means coincide, False otherwise.
        """
        cri = 1e-3
        diff = mean[1:] - mean[:-1]
        if ((diff**2).sum(1) ** 0.5 < cri).any():
            # print('mean coincide')
            return True
        else:
            return False

    @staticmethod
    def logLikelihood(array):
        """
        Calculate the log-likelihood of an array.

        Parameters:
        array (numpy.ndarray): The input array.

        Returns:
        float: The log-likelihood value.
        """
        llh = np.log(array.sum(0)).sum()
        return llh

    @staticmethod
    def componentArray(weight, mean, width, val):
        """
        Calculate the component array for a Gaussian Mixture Model.

        Args:
            weight (ndarray): The weight of each component in the mixture.
            mean (ndarray): The mean of each component in the mixture.
            width (ndarray): The width of each component in the mixture.
            val (ndarray): The input values for which to calculate the component array.

        Returns:
            ndarray: The component array.

        """
        components = weight * np.prod(width * 2 * np.pi, axis=-1) ** -(1 / 2)
        if np.size(mean, 1) == 2:
            distance2 = np.sum(
                (val - np.expand_dims(mean, 1)) ** 2 / np.expand_dims(width, 1) / 2, -1
            )
        else:
            distance2 = np.squeeze(
                (val - np.expand_dims(mean, 1)) ** 2 / np.expand_dims(width, 1) / 2
            )
        distance_term = np.exp(-distance2)
        component_array = components.reshape((len(components), 1)) * distance_term
        return component_array

    def calculateScore(self, tau, llh):
        """
        Calculates the score for the Gaussian Mixture Model.

        Parameters:
        - tau (numpy.ndarray): The tau matrix representing the responsibilities of each component for each data point.
        - llh (float): The log-likelihood of the Gaussian Mixture Model.

        Returns:
        - score (dict): A dictionary containing different scores calculated based on the given tau and llh.

        Score Calculation Methods:
        - aic: Akaike Information Criterion
        - gic: Generalized Information Criterion
        - bic: Bayesian Information Criterion
        - clc: Consistent Likelihood Criterion
        - awe: Akaike's Weighted Estimate
        - icl: Integrated Completed Likelihood
        - nllh: Negative Log-Likelihood
        - en: Entropy

        Note:
        - The score is calculated based on the number of components, dimensions, and other constraints of the Gaussian Mixture Model.

        """
        penalty = 2
        n_component, n_val = np.shape(tau)
        ## weight
        n_para = (self.step[0] != 0) * (n_component - 1)
        ## mean
        n_para_mean = n_component * self.n_dim
        n_para += n_para_mean
        ## width
        n_para_width = [1, n_component]
        if "uni_width" in self.constraint:
            n_para_width[1] = 1
        if self.n_dim == 2:
            n_para_width[0] = 3
            if ("45deg" in self.constraint) or ("no_cov" in self.constraint):
                n_para_width[0] = 2
        n_para += n_para_width[0] * n_para_width[1]

        t = tau * safe_ln(tau)
        t[tau == 0] = 0
        en = -1 * np.sum(t)

        score_calculations = {
            "aic": lambda: -2 * llh / self.n_dim + 2 * n_para,
            "gic": lambda: -2 * llh / self.n_dim + penalty * n_para,
            "bic": lambda: -2 * llh / self.n_dim + n_para * np.log(n_para),
            "clc": lambda: -2 * llh / self.n_dim + 2 * en,
            "awe": lambda: -2 * llh / self.n_dim
            + 2 * en
            + 2 * n_para * (3 / 2 + np.log(n_val)),
            "icl": lambda: -2 * llh / self.n_dim + 2 * en + n_para * np.log(n_val),
            "nllh": lambda: -1 * llh,
            "en": lambda: en,
        }

        score = {}
        for key in self.score_method:
            if key in score_calculations:
                score[key] = score_calculations[key]()
        return score

    def failedScore(self):
        """
        Returns a dictionary with the score method as keys and infinity as values.

        Returns:
            dict: A dictionary with the score method as keys and infinity as values.
        """
        score = {}
        for key in self.score_method:
            score[key] = np.inf
        return score

    def applyConstraint(self, weight, mean, width):
        """
        Applies constraints to the Gaussian mixture model parameters.

        Args:
            weight (numpy.ndarray): The weights of the Gaussian components.
            mean (numpy.ndarray): The means of the Gaussian components.
            width (numpy.ndarray): The widths (covariance matrices) of the Gaussian components.

        Returns:
            tuple: A tuple containing the updated weight, mean, and width arrays after applying the constraints.
        """
        # if 'uni_width' in self.constraint:
        #     width = np.expand_dims(width.mean(0), axis=0)
        if ("no_cov" in self.constraint) and self.n_dim == 2:
            mask = np.zeros(width.shape)
            for n in range(self.n_dim):
                mask[:, n, n] = 1
            width = width * mask
        if ("45deg" in self.constraint) and self.n_dim == 2:
            val = np.max([width[:, 0, 0], width[:, 1, 1]])
            # val = np.abs(width).max()
            for n in range(0, self.n_dim):
                width[:, n, n] = val
        return weight, mean, width

    @staticmethod
    def addChannel(mean_list, method, prmt):
        """
        Adds a new channel to each array in the given list using the specified method and parameters.

        Args:
            list (list): The list of arrays to add a channel to.
            method (function): The method to apply for adding the channel.
            prmt (tuple): The parameters to pass to the method.

        Returns:
            list: The list of arrays with the additional channel added.
        """
        return [np.concatenate((val, method(val, *prmt)), axis=-1) for val in mean_list]

    @staticmethod
    def initResultDict(n_component_list: np.ndarray, score_method: list, b_2d: bool):
        """
        Initialize the result dictionary for the Gaussian Mixture Model.

        Args:
            n_component_list (int): List of the number of components for each case.
            score_method (list): List of score methods.
            b_2d (bool): Boolean indicating whether the data is 2-dimensional.

        Returns:
            dict: The initialized result dictionary with empty lists for each key.
        """
        score_dict = {key: [] for key in score_method}
        gmm_result = {
            "weight": [],
            "mean": [],
            "width": [],
            "score": score_dict,
            "scsidx": [],
            "case": n_component_list,
        }
        return gmm_result

    @staticmethod
    def expCurve(x, a, b, c):
        return x**a * b + c

    @staticmethod
    def polyCurve_1(x, a, b):
        return x * a + b

    @staticmethod
    def polyCurve_2(x, a, b, c):
        return x**2 * a + x * b + c

    @staticmethod
    def polyCurve_3(x, a, b, c, d):
        return x**3 * a + x**2 * b + x * c + d

    @staticmethod
    def polyCurve_4(x, a, b, c, d, e):
        return x**4 * a + x**3 * b + x**2 * c + x * d + e

    @staticmethod
    def polyCurve_5(x, a, b, c, d, e, f):
        return x**5 * a + x**4 * b + x**3 * c + x**2 * d + x * e + f

    @staticmethod
    def polyCurve_6(x, a, b, c, d, e, f, g):
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
        g.componentArray()
        return np.argmax(g.ca, 0)

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
    def __init__(self, weight, mean, var, val, dose=None):
        """
        Gaussian component.

        Args:
            weight (ndarray): The weights of the Gaussian components.
            mean (ndarray): The means of the Gaussian components.
            var (ndarray): The variances of the Gaussian components.
            val (ndarray): The observed values.
            dose (ndarray, optional): The dose values. Defaults to None.
        """
        self.weight = weight
        self.mean = mean
        self.var = var
        self.val = val
        self.n_comp = self.weight.shape
        self.n_val, self.n_dim = self.val.shape
        self.dose = dose
        self.b_fail = False
        self.componentArray()
        self._tau
        self.tau_sum
        self.tau_ex

    def componentArray(self):
        #                   ORIGINAL AXIS         ->     EXPANDED AXIS
        # weight:           (component)           ->  (component, channel)
        # mean:             (component, channel)  ->  (component, data, channel)
        # var (same width): (1, channel)          ->  (1, data, channel)
        # var (diff width): (component, channel)  ->  (component, data, channel)
        # val:              (data, channel)
        # c:                (component, data)
        var = np.expand_dims(self.var, axis=1)
        mean = np.expand_dims(self.mean, 1)
        weight = np.expand_dims(self.weight, 1)
        ca = self._ca(var, mean, weight, self.val)
        self.EStep(ca)
        if not self.b_fail:
            self.ca = ca

    @staticmethod
    def _ca(var, mean, weight, val):
        """
        Calculate the conditional probability density function (PDF) of a Gaussian mixture model.

        Args:
            var (ndarray): The variance of each Gaussian component.
            mean (ndarray): The mean of each Gaussian component.
            weight (ndarray): The weight of each Gaussian component.
            val (ndarray): The input value for which to calculate the PDF.

        Returns:
            ndarray: The conditional probability density function (PDF) of the Gaussian mixture model.

        Raises:
            None

        """
        n_dim = mean.shape[-1]
        dis = (val - mean) ** 2
        if (var == 0).any():
            print("h")
        ca = (
            (2 * np.pi) ** (-n_dim / 2)
            * np.prod(var, -1) ** (-1 / 2)
            * np.exp(-1 / 2 * (dis / var).sum(-1))
        )
        ca *= weight
        return ca

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        self._tau = tau
        self.tau_sum = tau.sum(1) + np.finfo("double").eps
        self.tau_ex = np.expand_dims(tau, -1)
        self.tau_ex_sum = self.tau_ex.sum(1) + np.finfo("double").eps

    def EStep(self, ca):
        """
        Performs the E-step of the Gaussian Mixture Model algorithm.

        Parameters:
        ca (numpy.ndarray): The responsibility matrix, representing the probabilities of each data point belonging to each cluster.

        Returns:
        None
        """
        self.tau = ca / np.sum(ca, 0)
        if (self.tau.sum(1) < 1).any():
            self.b_fail = True

    def preMSTep(self, step, constraint):
        """
        Performs the pre-M-Step of the Gaussian Mixture Model algorithm.

        Args:
            step (tuple): A tuple containing the step values.
            constraint (float): The constraint value.

        Returns:
            None
        """
        self.updateWeight(step[0])
        self.componentArray()
        self.updateVariance(step[2], constraint)
        self.componentArray()

    def MStep(self, step, constraint):
        """
        Performs the M-step of the Expectation-Maximization algorithm for Gaussian Mixture Models.

        Args:
            step (tuple): A tuple containing the step values for updating the model parameters.
                          The tuple should have the following elements:
                          - step[0]: The step value for updating the weights.
                          - step[1]: The step value for updating the means.
                          - step[2]: The step value for updating the variances.
            constraint (bool): A boolean value indicating whether to apply any constraints during the parameter updates for variances.

        Returns:
            None

        """
        self.updateWeight(step[0])
        self.updateMean(step[1])
        self.updateVariance(step[2], constraint)
        self.componentArray()

    def updateWeight(self, step):
        new_weight = self.tau_sum / self.n_val
        self.weight = (new_weight - self.weight) * step + self.weight
        self.weight[-1] = 1 - np.sum(self.weight[:-1])

    def updateMean(self, step):
        new_mean = (self.tau_ex * self.val).sum(1) / self.tau_ex_sum
        self.mean = (new_mean - self.mean) * step[: self.n_dim] + self.mean

    def updateVariance(self, step, constraint):
        mean = np.expand_dims(self.mean, 1)
        var = np.expand_dims(self.var, 1)
        dis = (self.val - mean) ** 2
        if "uni_width" in constraint:
            if "dose_width" in constraint:
                var_dose = mean / self.dose
                var_indi = ((self.tau_ex * dis - var_dose) / var**2).sum((0, 1)) / (
                    self.tau_ex / var**2
                ).sum((0, 1))
                var_indi[var_indi < 0] = 0
                new_var = var_dose.sum(1) + np.expand_dims(var_indi, 0)
            elif "dose_width_simplified" in constraint:
                var_dose = mean / self.dose
                var_indi = (self.tau_ex * dis - var_dose).sum((0, 1)) / self.n_val
                var_indi[var_indi < 0] = 0
                new_var = var_dose.sum(1) + np.expand_dims(var_indi, 0)
            elif "dose_width_fit" in constraint:
                var = np.expand_dims(self.var, axis=1)
                mean = np.expand_dims(self.mean, 1)
                weight = np.expand_dims(self.weight, 1)
                var_dose = mean / self.dose
                args = (var_dose, mean, weight, self.val)
                var_indi = (var - var_dose).mean((0, 1))
                var_indi[var_indi < np.finfo("double").eps] = np.finfo("double").eps
                result = minimize(
                    self.__fit_dose_indi_var__, self.rev_rectifier(var_indi), args=args
                )
                if result.success:
                    var_indi = self.rectifier(result.x)
                    new_var = var_dose.sum(1) + np.expand_dims(var_indi, 0)
                else:
                    # print('cannot find solution')
                    new_var = self.var
            else:
                new_var = np.expand_dims(
                    (self.tau_ex * dis).sum((0, 1)) / self.n_val, 0
                )
        else:
            new_var = (self.tau_ex * dis).sum(1) / self.tau_ex_sum
        self.var = (new_var - self.var) * step[: self.n_dim] + self.var

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
