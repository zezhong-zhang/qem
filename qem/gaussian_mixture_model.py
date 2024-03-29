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
        self.result = {}
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
            use_scs_channel (int or list or None): The channel(s) to be used for the model. If an integer is provided, only that channel will be used. If a list is provided, the channels specified in the list will be used. If None is provided, the first channel will be used by default.
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

    def fit(
        self,
        name: str,
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
            name (str): The name of the GMM model.
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
            constraint (list): The constraints applied to the GMM. Default is [].

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

        self.result[name] = GmmResult(
            name,
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
            tuple: A tuple containing the optimized weights and the minimum score.

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

        score = {}
        for key in self.score_method:
            if key == "aic":
                score[key] = -2 * llh / self.n_dim + 2 * n_para
            if key == "gic":
                score[key] = -2 * llh / self.n_dim + penalty * n_para
            if key == "bic":
                score[key] = -2 * llh / self.n_dim + n_para * np.log(n_para)
            if key == "clc":
                score[key] = -2 * llh / self.n_dim + 2 * en
            if key == "awe":
                score[key] = (
                    -2 * llh / self.n_dim
                    + 2 * en
                    + 2 * n_para * (3 / 2 + np.log(n_val))
                )
            if key == "icl":
                score[key] = -2 * llh / self.n_dim + 2 * en + n_para * np.log(n_val)
            if key == "nllh":
                score[key] = -1 * llh
            if key == "en":
                score[key] = en
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
        return (
            x**6 * a + x**5 * b + x**4 * c + x**3 * d + x**2 * e + x * f + g
        )


class GmmResult:
    def __init__(
        self,
        name: str,
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
            name (str): The name of the Gaussian mixture model.
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
        self.name = name
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
        ca = g.componentArray()
        return np.argmax(ca, 0)

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
            constraint (bool): A boolean value indicating whether to apply any constraints during the parameter updates.

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

#### extra code
class GaussianMixtureModelObject:
    """
    To Solve GMM
    """

    # setter / getter
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: np.ndarray):
        self._value = value
        self._num_column = np.size(self._value, 1)
        self._num_dim = np.size(self.num_dim, 0)

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: np.ndarray):
        self._coordinates = coordinates

    def __init__(self, value: np.ndarray, coordinates: np.ndarray):
        """
        Parameters
        ----------
        value: np.ndarray
            nparray of volume of gaussian, peak intensity, or scattering crossection
        coordinates: np.ndarray
            coordinates of the atomic columns
        """
        self._value = value
        self._coordinates = coordinates
        try:
            self._num_column = np.size(self._value, 1)
            self._num_dim = np.size(self._value, 0)
        except:
            self._value = np.reshape(self._value, [1, np.size(self._value)])
            self._num_column = np.size(self._value, 1)
            self._num_dim = np.size(self._value, 0)
        self._use_dim = self._num_dim
        self.criteria_dict = {}
        self._criteria_list = []
        self.weights_list = []
        self.mean_list = []
        self.var_list = []
        self.component = []
        self.max_n_components = 0
        self.__data_dimension_check__()

    def __data_dimension_check__(self):
        """
        check number of column matches with provided number of coordinates
        """
        assert self._num_column == np.size(
            self._coordinates, 1
        ), "GaussianMixtureModelObject._num_coordinates does noth match with _num_column"
        
    def remove_edge(self, edge_width, image_size):
        lx, ly = image_size
        delete_list = []
        for i in range(self._num_column):
            if (self._coordinates[0,i] < edge_width or 
                self._coordinates[0,i] > lx-edge_width or 
                self._coordinates[1,i] < edge_width or 
                self._coordinates[1,i] > ly-edge_width):
                delete_list.append(i)
        self._coordinates = np.delete(self._coordinates, delete_list, 1)
        self._value = np.delete(self._value, delete_list, 1)
        self._num_column = np.size(self._value, 1)

    @staticmethod
    def cri_test(criteria, llh, num_para, num_column, penalty, tau_array):
        cri = []
        # t = tau_array * np.log(tau_array)
        t = tau_array * safe_ln(tau_array)
        t[tau_array == 0] = 0
        en = -1 * np.sum(t)
        for key in criteria:
            if key == "aic":
                cri.append(-2 * llh + 2 * num_para)
            if key == "gic":
                cri.append(-2 * llh + penalty * num_para)
            if key == "bic":
                cri.append(-2 * llh + num_para * np.log(num_column))
            if key == "clc":
                cri.append(-2 * llh + 2 * en)
            if key == "awe":
                cri.append(
                    -2 * llh + 2 * en + 2 * num_para * (3 / 2 + np.log(num_column))
                )
            if key == "icl":
                cri.append(-2 * llh + 2 * en + num_para * np.log(num_column))
            if key == "nllh":
                cri.append(-1 * llh)
        return cri

    @staticmethod
    def log_likelihood(component_array):
        llh = np.sum(np.log(np.sum(component_array, (0, 2))))
        return llh

    @staticmethod
    def component_array(weight, mean, var, value):
        num_component = np.size(weight, 0)
        num_column = np.size(value, 1)
        num_dim = np.size(value, 2)
        ca = np.zeros([num_component, num_column, 1])
        ca[:, :, 0] = (
            weight[:, None]
            * (2 * np.pi) ** -(num_dim / 2)
            * np.prod(var[0, 0, :]) ** -(1 / 2)
            * np.exp(-np.sum((value - mean) ** 2 / var, 2) / 2)
        )
        return ca

    @staticmethod
    def __init_wmv__(v_max, v_min, mean, num_component, num_dim):
        weight = np.ones(num_component) / num_component
        var = ((v_max - v_min) / (2 * num_component)) ** 2
        mean = np.reshape(mean, (num_component, 1, num_dim))
        var = np.tile(var, (num_component, 1))
        var = np.reshape(var, (num_component, 1, num_dim))
        return weight, mean, var

    def __search__(self, value, criteria, num_component, pos_init):
        b_multi_d = False
        v_max = np.amax(value, 1)
        v_min = np.amin(value, 1)

        if self._use_dim > 1:
            x = value[0, :, 0]
            y = value[0, :, 1:]
            # coef = np.linalg.lstsq(x, y, rcond=None)[0]
            coef = np.squeeze(np.polyfit(x,y,3))
            p = np.poly1d(coef)
            b_multi_d = True

        # uniform distance
        if pos_init[0] == 0:
            mean_0 = np.linspace(v_min[0, 0], v_max[0, 0], num_component)
            if b_multi_d:
                # mean = np.vstack([mean_0, mean_0 * coef[0] + coef[1]]).T
                mean = np.vstack([mean_0, p(mean_0)]).T
            else:
                mean = mean_0
            weight, mean, var = self.__init_wmv__(
                v_max, v_min, mean, num_component, self._use_dim
            )
            weight, mean, var, cri = self.__EM__(weight, mean, var, value, criteria)

        # StatSTEM
        if pos_init[0] == 1:
            if num_component == 1:
                mean_0 = (v_min[0, 0] + v_max[0, 0]) / 2
                if b_multi_d:
                    # mean = np.vstack([mean_0, mean_0 * coef[0] + coef[1]]).T
                    mean = np.vstack([mean_0, p(mean_0)]).T
                else:
                    mean = mean_0
                weight, mean, var = self.__init_wmv__(
                    v_max, v_min, mean, num_component, self._use_dim
                )
                weight, mean, var, cri = self.__EM__(weight, mean, var, value, criteria)
            else:
                mean_last = self.mean_list[num_component - 2][
                    :, 0, 0]
                mean_scale = np.insert(
                    mean_last, (0, num_component - 2), (v_min[0, 0], v_max[0, 0])
                )
                weight_t, mean_t, var_t, cri_t, choose_t = [], [], [], [], []
                criteria_plus = criteria.copy()
                criteria_plus.append('nllh')
                for i in range(num_component):
                    mean_0 = np.append(
                        mean_last, (mean_scale[i] + mean_scale[i + 1]) / 2
                    )
                    if b_multi_d:
                        # mean = np.vstack([mean_0, mean_0 * coef[0] + coef[1]]).T
                        mean = np.vstack([mean_0, p(mean_0)]).T
                    else:
                        mean = mean_0
                    weight, mean, var = self.__init_wmv__(
                        v_max, v_min, mean, num_component, self._use_dim
                    )
                    w, m, v, c = self.__EM__(weight, mean, var, value, criteria_plus)
                    weight_t.append(w)
                    mean_t.append(m)
                    var_t.append(v)
                    choose_t.append(c.pop())
                    cri_t.append(c)
                id = np.argmin(choose_t)
                # plt.plot(choose_t)
                # plt.plot(id,min(choose_t),'o')
                # name = 'full_conv' + str(num_component) + '.png'
                # plt.savefig(name)
                # plt.clf()
                weight = weight_t[id]
                mean = mean_t[id]
                var = var_t[id]
                cri = cri_t[id]

        # Annick's approach
        if pos_init[0] == 2:
            if num_component == 1:
                mean_0 = (v_min[0, 0] + v_max[0, 0]) / 2
                if b_multi_d:
                    # mean = np.vstack([mean_0, mean_0 * coef[0] + coef[1]]).T
                    mean = np.vstack([mean_0, p(mean_0)]).T
                else:
                    mean = mean_0
                weight, mean, var = self.__init_wmv__(
                    v_max, v_min, mean, num_component, self._use_dim
                )
                weight, mean, var, cri = self.__EM__(weight, mean, var, value, criteria)
            else:
                mean_last = self.mean_list[num_component - 2][
                    :, 0, 0]
                mean_scale = np.linspace(v_min[0, 0], v_max[0, 0], self.max_n_components)
                weight_t, mean_t, var_t, cri_t, choose_t = [], [], [], [], []
                criteria_plus = criteria.copy()
                criteria_plus.append('nllh')
                for i in range(self.max_n_components):
                    mean_0 = np.append(mean_last, mean_scale[i])
                    if b_multi_d:
                        # mean = np.vstack([mean_0, mean_0 * coef[0] + coef[1]]).T
                        mean = np.vstack([mean_0, p(mean_0)]).T
                    else:
                        mean = mean_0
                    weight, mean, var = self.__init_wmv__(
                        v_max, v_min, mean, num_component, self._use_dim
                    )
                    w, m, v, c = self.__EM__(weight, mean, var, value, criteria_plus)
                    weight_t.append(w)
                    mean_t.append(m)
                    var_t.append(v)
                    choose_t.append(c.pop())
                    cri_t.append(c)
                id = np.argmin(choose_t)
                weight = weight_t[id]
                mean = mean_t[id]
                var = var_t[id]
                cri = cri_t[id]

        # repeat
        for _ in range(pos_init[1]):
            weight, _, var = self.__init_wmv__(
                v_max, v_min, mean[:, 0, 0], num_component, self._use_dim
            )
            weight, mean, var, cri = self.__EM__(weight, mean, var*2, value, criteria)

        return weight, mean, var, cri

    def __EM__(self, weight, mean, var, value, criteria):
        num_component = np.size(weight, 0)
        num_column = np.size(value, 1)
        num_dim = np.size(value, 2)
        num_para = num_component - 1 + num_dim * num_component + num_dim

        ca = self.component_array(weight, mean, var, value)
        llh = self.log_likelihood(ca)
        change_ratio = 1
        cnt = 0
        while change_ratio > 1e-7:
            tau_array = ca / np.sum(ca, 0)            
            if 0 in np.sum(tau_array, (1, 2)): break
            weight = np.sum(tau_array, (1, 2)) / num_column
            if num_component != 1:
                weight[-1] = 1 - np.sum(weight[:-1])
            mean = np.sum(tau_array * value, 1, keepdims=True) / np.sum(
                tau_array, (1, 2), keepdims=True
            )
            var = (
                np.sum(tau_array * (value - mean) ** 2, (0, 1), keepdims=True)
                / num_column
            )
            var = np.repeat(var, num_component, axis=0)

            ca = self.component_array(weight, mean, var, value)
            llh_new = self.log_likelihood(ca)
            change_ratio = abs(llh_new - llh) / abs(llh)
            llh = llh_new
            cnt += 1
        sort_idx = np.argsort(mean, axis=0)
        t = sort_idx[:, :, 0]
        sort_idx = t[:, :, None]
        mean = np.take_along_axis(mean, sort_idx, axis=0)
        weight = np.take_along_axis(weight, sort_idx[:, 0, 0], axis=0)
        tau_array = ca / np.sum(ca, 0)
        cri = self.cri_test(criteria, llh, num_para, num_column, 2, tau_array)
        return weight, mean, var, cri

    def GMM(
        self, max_component: int, use_dim=None, criteria=["icl"], pos_init=(1, 0)
    ):
        """
        Find components of Gaussian mixture model

        Parameters
        ----------
        max_component: int
            How many components to try.
        use_dim (optional): int
            How many dimensions in the value list will be used. Use all if not specified.
        criteria (optional): list
            Which criteria to use. Options: icl, aic, gic, bic, clc, awe. Use 'icl' if not specified
        pos_init (optional):
            methods to add initial position.
        ----------
        Examples:
            adf_value = [1, 2, 3, 4]
            abf_value = [5, 6, 7, 8]
            x = np.concatenate([adf_value, abf_value])
            GaussianMixtureModelObject ADF(x, coordinates)
            ADF.GMM([1,50], ['icl', 'aic'])
        ----------
        Questions:
            how to calculate derivative of log likelihood to parameter?
        """

        # axis0: number of component
        # axis1: number of column
        # axis2: data dimension

        if use_dim is not None:
            if use_dim > self._num_column:
                self._use_dim = self._num_dim
                print("Use maximal dimension")
            else:
                self._use_dim = use_dim

        self.criteria_dict = {}
        self._criteria_list = []
        self.weights_list = []
        self.mean_list = []
        self.var_list = []
        self.max_n_components = max_component

        value = np.expand_dims(self._value[: self._use_dim, :].T, axis=0)

        for n_component in tqdm(range(1, self.max_n_components+1)):
            weight, mean, var, cri = self.__search__(
                value, criteria, n_component, pos_init=pos_init
            )
            self.weights_list.append(weight)
            self.mean_list.append(mean)
            self.var_list.append(var)
            self._criteria_list.append(cri)

        self._criteria_list = np.transpose(self._criteria_list)
        self.criteria_dict = {key: value for key, value in zip(criteria, self._criteria_list)}
      
    def plot_thickness(self, n_component, show_component=None):
        component_case = n_component - 1
        xmin = np.amin(self._coordinates[0, :])
        xmax = np.amax(self._coordinates[0, :])
        ymin = np.amin(self._coordinates[1, :])
        ymax = np.amax(self._coordinates[1, :])
        weight = self.weights_list[component_case]
        mean = self.mean_list[component_case]
        var = self.var_list[component_case]
        use_dim = np.size(var, 2)
        value = np.expand_dims(self._value[:use_dim, :].T, axis=0)
        ca = self.component_array(weight, mean, var, value)
        self.component = np.argmax(ca[:, :, 0], axis=(0))
        plt.figure()
        plt.scatter(
            self._coordinates[0, :],
            self._coordinates[1, :],
            marker="o",
            c=self.component,
        )
        ax = plt.gca()
        ax.set_xlim([xmin - 10, xmax + 10])
        ax.set_ylim([ymin - 10, ymax + 10])
        ax.set_aspect('equal')
        plt.show(block=False)
        plt.pause(1)
        if show_component is not None:
            # if ~isinstance(show_component, list): show_component = [show_component]
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
            ax.set_xlim([xmin - 10, xmax + 10])
            ax.set_ylim([ymin - 10, ymax + 10])
            ax.set_aspect('equal')
            plt.show(block=False)
            plt.pause(1)

    def plot_criteria(self, criteria=None):
        xaxis = np.arange(self.max_n_components)
        fig, ax = plt.subplots(1,1)
        for cri in criteria:
            plt.plot(xaxis, self.criteria_dict[cri], label=cri)
            plt.plot(
                np.argmin(self.criteria_dict[cri])+1, self.criteria_dict[cri].min(), 'o')
        legend = ax.legend(loc="upper center")
        plt.show(block=False)
        plt.pause(1)

    def plot_histogram(self, n_component=None, use_dim=None, bin=None):
        if use_dim is None or use_dim > self._use_dim:
            use_dim = self._use_dim
        if bin is None:
            bin = np.size(self._value, axis=1) // 10

        if use_dim != 2 and use_dim != 1:
            print("only support up to 2 dimensions")
            return
        elif use_dim == 2:
            plt.figure()
            plt.hist2d(self._value[0, :], self._value[1, :], bins=bin)
        elif use_dim == 1:
            plt.figure()
            plt.hist(self._value[0, :], bins=bin)

        if n_component is None:
            min_icl_comp = np.argmin(self.criteria_dict['icl'])
            print("Number of components is chosen to be ", min_icl_comp+1, "based on ICL.\n")
            component_case = n_component - 1
        else:
            component_case = n_component - 1
            weight = self.weights_list[component_case]
            mean = self.mean_list[component_case]
            var = self.var_list[component_case]
            if use_dim == 1:
                sigma = var[0, 0, 0] ** 0.5
                for c in range(component_case):
                    w = weight[c] * self._num_column * sigma
                    mu = mean[c, 0, 0]
                    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, bin)
                    plt.plot(
                        x,
                        w
                        / (sigma * np.sqrt(2 * np.pi))
                        * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                    )
                    plt.text(mu, w / (sigma * np.sqrt(2 * np.pi)) * 1.1, c + 1)
            elif use_dim == 2:
                t = np.linspace(-np.pi, np.pi, bin)
                sigma = var[0] ** 0.5
                for c in range(component_case):
                    mu = mean[c]
                    x = mu[0,0] + sigma[0,0]*np.cos(t)
                    y = mu[0,1] + sigma[0,1]*np.sin(t)
                    plt.plot(x,y)
                    plt.text(mu[0,0], mu[0,1], c+1)
        plt.show(block=False)
        plt.pause(1)