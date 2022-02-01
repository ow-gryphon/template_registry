import itertools
import sklearn as sk
from sklearn import linear_model
from scipy import stats
import numpy as np
import pandas as pd
import warnings
import time
from sklearn.metrics import log_loss

import warnings

class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculates standard errors,
    t-statistics and p-values for model coefficients (betas).
    Additional attributes available after .fit() are `se`, `t` and `p` which are of the shape (y.shape[1], X.shape[1]),
    which is (n_features, n_coefs) This class sets the intercept to 0 by default, since usually we include it in X.

    This code was taken, with some minor refinements, from
    "https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression"

    NOTE: the additional statistics (se, t, and p) are only accurate if fit_intercept = False.
    So if intercept is needed, it should be provided as part of the X variable
    """

    def __init__(self, *, fit_intercept=False, normalize="deprecated", copy_X=True, n_jobs=None, positive=False,):
        if fit_intercept:
            warnings.warn("Note: se, t, and p-values are not accurate when using fit_intercept." +
                  "Instead, run with fit_intercept = False and add an intercept column to X")

        super(LinearRegression, self).__init__(fit_intercept=False, normalize="deprecated", copy_X=True, n_jobs=None, positive=False,)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        inverse_covariance = np.linalg.inv(np.dot(X.T, X))
        self.se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])[0]
        self.t = self.coef_ / self.se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1])) 
        return self
    

class Exponential_Family():
    """
        Helper class defining some useful functions for the GLM. This is a class used internally by the glm function
        linkfun is the link-function
        linkinv is the inverse link function
        mu_eta is the derivative function of the inverse link function
        variance is the variance of the distribution relative to mu
    """

    def __init__(self):
        self

    def linkfun(self, data_array):
        pass

    def linkinv(self, data_array):
        pass

    def mu_eta(self, data_array):
        pass

    def variance(self, data_array):
        pass

    def check_array(self, data_array):
        if not isinstance(data_array, np.ndarray):
            raise Exception("The data is not an np array")


class LogLink(Exponential_Family):
    """
        Definition of functions related to the log-transformation
    """
    def mu_eta(self, data_array):
        self.check_array(data_array)
        return np.exp(data_array)

    def linkfun(self, data_array):
        self.check_array(data_array)
        return np.log(data_array)

    def linkinv(self, data_array):
        self.check_array(data_array)
        return np.exp(data_array)


class IdentityLink(Exponential_Family):
    """
        Definition of functions related to no-transformation
    """
    def mu_eta(self, data_array):
        self.check_array(data_array)
        return np.ones(len(data_array)).reshape(-1, 1)

    def linkfun(self, data_array):
        self.check_array(data_array)
        return np.ones(len(data_array)).reshape(-1, 1)

    def linkinv(self, data_array):
        self.check_array(data_array)
        return np.ones(len(data_array)).reshape(-1, 1)


class LogitLink(Exponential_Family):
    """
        Definition of functions related to logit-transformation
    """
    def mu_eta(self, data_array):
        self.check_array(data_array)
        return np.exp(-1 * data_array) / np.power((1 + np.exp(-1 * data_array)), 2)

    def linkfun(self, data_array):
        self.check_array(data_array)
        return np.log(data_array / (1 - data_array))

    def linkinv(self, data_array):
        self.check_array(data_array)
        return 1 / (1 + np.exp(-1 * data_array))


class GaussianDist(Exponential_Family):
    """
        Definition of variance for Gaussian distribution
    """
    def variance(self, data_array):
        self.check_array(data_array)
        return np.ones(len(data_array)).reshape(-1, 1)


class BinomialDist(Exponential_Family):
    """
        Definition of variance for binomial distribution
    """
    def variance(self, data_array):
        self.check_array(data_array)
        return data_array * (1 - data_array)


class GammaDist(Exponential_Family):
    """
        Definition of variance for gamma distribution
    """

    def variance(self, data_array):
        self.check_array(data_array)
        return np.power(data_array, 2)


class InvGaussianDist(Exponential_Family):
    """
        Definition of variance for inverse gaussian distribution
    """
    def variance(self, data_array):
        self.check_array(data_array)
        return np.power(data_array, 3)


class PoissonDist(Exponential_Family):
    """
        Definition of variance for poisson distribution
    """
    def variance(self, data_array):
        self.check_array(data_array)
        return data_array


# Gaussians
class IdentityGaussian(IdentityLink, GaussianDist):
    """
        Stitching together the Identity-Link Gaussian GLM properties
    """
    def __init__(self):
        super().__init__()


class LogGaussian(LogLink, GaussianDist):
    """
        Stitching together the Log-Link Gaussian GLM properties
    """
    def __init__(self):
        super().__init__()


class LogitGaussian(LogitLink, GaussianDist):
    """
        Stitching together the Logit-Link Gaussian GLM properties
    """
    def __init__(self):
        super().__init__()


# Binomial
class IdentityBinomial(IdentityLink, BinomialDist):
    """
        Stitching together the Identity-Link Binomial GLM properties
    """
    def __init__(self):
        super().__init__()


class LogBinomial(LogLink, BinomialDist):
    """
        Stitching together the Log-Link Binomial GLM properties
    """

    def __init__(self):
        super().__init__()


class LogitBinomial(LogitLink, BinomialDist):
    """
        Stitching together the Logit-Link Binomial (logistic) GLM properties
    """
    def __init__(self):
        super().__init__()


# Gamma
class IdentityGamma(IdentityLink, GammaDist):
    """
        Stitching together the Identity-Link Gamma GLM properties
    """
    def __init__(self):
        super().__init__()


class LogGamma(LogLink, GammaDist):
    """
        Stitching together the Log-Link Gamma GLM properties
    """
    def __init__(self):
        super().__init__()


class LogitGamma(LogitLink, GammaDist):
    """
        Stitching together the Logit-Link Gamma GLM properties
    """
    def __init__(self):
        super().__init__()


# InvGaussian
class IdentityInvGaussian(IdentityLink, InvGaussianDist):
    """
        Stitching together the Identity-Link InverseGaussian GLM properties
    """
    def __init__(self):
        super().__init__()


class LogInvGaussian(LogLink, InvGaussianDist):
    """
        Stitching together the Log-Link InverseGaussian GLM properties
    """
    def __init__(self):
        super().__init__()


class LogitInvGaussian(LogitLink, InvGaussianDist):
    """
        Stitching together the Logit-Link InverseGaussian GLM properties
    """
    def __init__(self):
        super().__init__()


# Poisson
class IdentityPoisson(IdentityLink, PoissonDist):
    """
        Stitching together the Identity-Link Poisson GLM properties
    """
    def __init__(self):
        super().__init__()


class LogPoisson(LogLink, PoissonDist):
    """
        Stitching together the Log-Link Poisson GLM properties
    """
    def __init__(self):
        super().__init__()


class LogitPoisson(LogitLink, PoissonDist):
    """
        Stitching together the Logit-Link Poisson GLM properties
    """
    def __init__(self):
        super().__init__()


def fit_glm(used_data, y_var, x_var, family = IdentityGaussian(), weight = None, tolerance = 0.001, max_iter = 100,
            show_iter = False):
    """
    Fits generalized linear model to data in Pandas Dataframe format using Iteratively Weighted Least Squares.
    The user passes link and distribution since
    (i) less experienced users are not familiar with the link-distribution pairing, and
    (ii) users may want to alter the distribution without changing link function

    :param used_data: a single pandas DataFrame containing y variable, x variables, and options sample weights
    :param y_var: a single string with the name of the y variable
    :param x_var: a list of strings representing the names of x variables
    :param link: a string with the link function
    :param distribution: a string with the distribution function (part of the exponential family)
    :param weight: a string with the name of the sample weight variable (or can be None)
    :param tolerance: a double representing the level of tolerance for coefficient fluctuation between
    iterations (based on ratio in coefficient changes)
    :param max_iter: an integer with the maximum number of iterations
    :show_iter: a boolean indicating whether the convergence criteria in each iteration should be shown
    :return: a tuple containing the following: (i) summary pandas dataframe with coefficients, standard errors and
    p-values, (ii) final regression in the weighted least squares iterations, (iii) model fit at the response level
    """

    if len(x_var) == 1:
        x_val = used_data[x_var].as_matrix().reshape(-1, 1)
    else:
        x_val = used_data[x_var].as_matrix()

    y_val = used_data[y_var].as_matrix()
    if weight is None:
        weights = None
    else:
        weights = used_data[weight].as_matrix()

    if len(x_var) == 1:
        x_val = used_data[x_var].as_matrix().reshape(-1, 1)
    else:
        x_val = used_data[x_var].as_matrix()
    y_val = used_data[y_var].as_matrix()
    if weight is None:
        weights = None
    else:
        weights = used_data[weight].as_matrix()

    # Step 1, get initial guesses for regression by running standard OLS (possibly with sample weights)
    new_reg = linear_model.LinearRegression(fit_intercept=True)

    new_reg = new_reg.fit(X=x_val, y=family.linkfun(y_val.reshape(-1, 1)), sample_weight=weights)

    for iteration in range(max_iter):
        old_reg = new_reg

        # Get fitted and coefficients
        prior_estimates = np.append(old_reg.intercept_, old_reg.coef_[0])
        prior_fitted = family.linkinv(old_reg.predict(X=x_val))

        # Step 2, get new estimates from first step of GLM
        b_val = y_val.reshape(-1, 1)
        t_val = family.linkfun(prior_fitted)
        g_val = prior_fitted
        g_prime = family.mu_eta(t_val)
        z_val = t_val + (b_val - g_val) / g_prime
        w_val = np.power(g_prime, 2) / family.variance(g_val)
        if weights is None:
            new_weight = w_val.reshape(1, -1)[0]
        else:
            new_weight = w_val.reshape(1, -1)[0] * weights

        new_reg = linear_model.LinearRegression(fit_intercept=True)
        new_reg = new_reg.fit(X=x_val, y=z_val.reshape(-1, 1), sample_weight=new_weight)

        # Get coefficients
        new_estimates = np.append(new_reg.intercept_, new_reg.coef_[0])

        converge_check = np.max(np.abs(new_estimates / prior_estimates - 1))
        if show_iter:
            print("Iteration {}: {}".format(iteration, str(converge_check)))

        if converge_check <= tolerance:
            break

    # Re-estimate final model to also get p-values
    w_x_val = np.append(np.ones(len(x_val)).reshape(-1, 1), x_val, 1) * np.power(new_weight.reshape(-1, 1), 0.5)
    w_z_val = z_val * np.power(new_weight.reshape(-1, 1), 0.5)

    final_reg = LinearRegression(fit_intercept=False)
    final_reg = final_reg.fit(X=w_x_val, y=w_z_val)

    # Check results are the same
    check_final = np.max(np.abs(final_reg.coef_ / new_estimates - 1))
    if (check_final > tolerance):
        raise Exception("Final estimation failed")

    # Get the final model results
    # This is the model fit from the last iteration using linear_model: model_fit = new_reg.predict(X = x_val)
    model_fit = family.linkinv(final_reg.predict(X=np.append(np.ones(len(x_val)).reshape(-1, 1), x_val, 1)))

    # Get the full specification
    regression = pd.DataFrame({"Vars": ["Intercept"] + x_var,
                               "coef": final_reg.coef_[0],
                               "std-err": final_reg.se[0],
                               "t-stat": final_reg.t[0],
                               "p-val": final_reg.p[0]})

    return regression, final_reg, model_fit


def processSubset(X, y, feature_set, regression_type, weight=None):
    # Fit model on feature_set and calculate RSS
    if regression_type == "OLS":
        model = linear_model.LinearRegression(fit_intercept=True)
    elif regression_type == "Logistic":
        model = linear_model.LogisticRegression(fit_intercept=True)
    else:
        raise ValueError("Model type not yet implemented")
        #todo more GLM functions

    if weight is not None:
        model = model.fit(X=X[list(feature_set)], y=y, sample_weight=weight)
    else:
        model = model.fit(X=X[list(feature_set)], y=y, sample_weight=weight)

    if regression_type == "OLS":
        stat_used = "RSS"
        stat = ((model.predict(X[list(feature_set)]) - y) ** 2).sum()
    elif regression_type == "Logistic":
        stat_used = "LogLik"
        stat = log_loss(y, model.predict(X[list(feature_set)]))

    return model, {"statistic": stat_used, "value": stat, "num_var": len(feature_set)}


def getBest(X, y, min_var, max_var, regression_type = "OLS", weight = None, forced_in = None):

    results = []
    models = []

    variables = set(X.columns)
    if forced_in is None:
        new_variables = variables
    else:
        new_variables = variables - set(forced_in)

    min_var = max(1, min_var)
    max_var = min(max_var, len(new_variables))

    for num_var in range(min_var, max_var + 1):
        tic = time.time()
        counter = 0
        for combo in itertools.combinations(new_variables, num_var):
            if forced_in is None:
                temp_model, temp_result = processSubset(X, y, list(combo), regression_type, weight=weight)
            else:
                temp_model, temp_result = processSubset(X, y, forced_in + list(combo), regression_type, weight=weight)

            counter = counter + 1
            results.append(temp_result)
            models.append(temp_model)

        toc = time.time()
        print("Processed", counter, "models on", num_var, "predictors in", "{:.4f}".format(toc - tic), "seconds.")

    # Return the models, along with some other useful information about the model
    return models, pd.DataFrame(results).assign(index = range(0,len(results)))


def forward(X, y, max_var, regression_type = "OLS", weight = None, forced_in = None):

    variables = set(X.columns)
    if forced_in is None:
        all_variables = list(variables)
    else:
        all_variables = list(variables - set(forced_in))

    predictors = []
    master_results = []
    best_models = []
    best_results = []
    for i in range(1, min(max_var, len(all_variables)) + 1):
        # Pull out predictors we still need to process
        remaining_predictors = [p for p in all_variables if p not in predictors]

        tic = time.time()

        results = []
        models = []
        stats = []
        for p in remaining_predictors:
            if forced_in is None:
                temp_model, temp_result = processSubset(X, y, predictors + [p], regression_type, weight)
                temp_result["Vars"] = predictors + [p]
            else:
                temp_model, temp_result = processSubset(X, y, forced_in + predictors + [p], regression_type, weight)
                temp_result["Vars"] = forced_in + predictors + [p]
            
            stats.append(temp_result['value'])
            results.append(temp_result)
            models.append(temp_model)

        toc = time.time()
        print("Processed ", len(models), "models on", len(predictors) + 1, "predictors in", (toc - tic), "seconds.")

        # Choose the model with the highest RSS
        best_item = np.argmin(stats)
        best_model = models[best_item]
        best_result = results[best_item]

        predictors = predictors + [remaining_predictors[best_item]]

        master_results.extend(results)
        best_models.append(best_model)
        best_results.append(best_result)

    # Return the best model, along with some other useful information about the model
    return best_results, best_models, master_results


