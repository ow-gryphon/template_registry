import sklearn as sk
from sklearn import linear_model, metrics
from sklearn.preprocessing import StandardScaler
import pandas as pd
from collections import OrderedDict


def lasso_ols(dataset, DV, IVs, forced_in=None, intercept=True, alpha_list=None):
    '''
    Runs LASSO regression on data for variable selection purposes and generates outputs for all LASSO
    Perform OLS regression on individual independent variables, with optional forced in variables
    :param dataset: pandas dataframe
    :param DV: name of target variable (dependent variable)
    :param IVs: list of names of independent variables
    :param forced_in: optional list of forced in variables. All variables named here will be forced in
    :param intercept: Boolean indicating whether of not to include intercept
    :param alpha_list: List of alpha penalty values
    :return: pandas dataset containing summary statistics and coefficients
    '''

    # Check inputs and reformat if necessary
    if DV is None:
        raise ValueError("You must include DV")
    if IVs is None:
        raise ValueError("You must include IVs")

    if forced_in is not None:
        if isinstance(forced_in, str):
            forced_in = [forced_in]
        # Remove any IVs from the IVs list if they are already being forced in
        IVs = list(set(IVs) - set(forced_in))

    if alpha_list is None:
        alpha_list = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25]

    # Set up the result table in Pandas
    col_info = OrderedDict()
    col_info['Alpha'] = pd.Series([], dtype='float')
    col_info['Variables'] = pd.Series([], dtype='str')
    col_info['Converged?'] = pd.Series([], dtype='bool')
    col_info['Rsq'] = pd.Series([], dtype='float')

    if intercept:
        col_info['Intercept'] = pd.Series([], dtype='float')

    if forced_in is not None:
        vars = forced_in + IVs
    else:
        vars = IVs

    for var in vars:
        col_info["{} Coef".format(var)] = pd.Series([], dtype='float')

    # Create the pandas
    output = pd.DataFrame(col_info)

    scaler = StandardScaler()

    if forced_in is not None:
        model_dataset = dataset[[DV] + IVs + forced_in].dropna()
        model_dataset[IVs + forced_in] = scaler.fit_transform(model_dataset[IVs + forced_in])
        X = model_dataset[forced_in + IVs].copy()
    else:
        model_dataset = dataset[[DV] + IVs].dropna()
        model_dataset[IVs] = scaler.fit_transform(model_dataset[IVs])
        X = model_dataset[IVs].copy()

    Y = model_dataset[DV]

    if forced_in is not None:
        X[forced_in] = X[forced_in] * 1000

    for alpha in alpha_list:

        lasso = linear_model.Lasso(alpha=alpha, fit_intercept=intercept)

        fitted_model = lasso.fit(X, Y)

        results = OrderedDict()

        results['Alpha'] = alpha
        results[
            'Converged?'] = fitted_model.n_iter_ < fitted_model.max_iter  # Assuming max number of iterations met means no convergence

        results['Rsq'] = fitted_model.score(X, Y)

        if intercept:
            results['Intercept'] = fitted_model.intercept_

        var_list = []
        for var in vars:
            var_coef = fitted_model.coef_[vars.index(var)]
            if var in forced_in:
                var_coef = var_coef*1000
            if var_coef != 0:
                var_list.append(var)
            results["{} Coef".format(var)] = var_coef

        results['Variables'] = ";".join(var_list)

        output = pd.concat([output, pd.DataFrame([results])]).reset_index(drop=True)

    return output


def lasso_logistic(dataset, DV, IVs, forced_in=None, intercept=True, C_list=None):
    '''
    Runs LASSO regression on data for variable selection purposes and generates outputs for all LASSO
    Perform OLS regression on individual independent variables, with optional forced in variables
    :param dataset: pandas dataframe
    :param DV: name of target variable (dependent variable)
    :param IVs: list of names of independent variables
    :param forced_in: optional list of forced in variables. All variables named here will be forced in
    :param intercept: Boolean indicating whether of not to include intercept
    :param C_list: List of complexity values used with sklearn's LogisticRegression
    :return: pandas dataset containing summary statistics and coefficients
    '''

    # Check inputs and reformat if necessary
    if DV is None:
        raise ValueError("You must include DV")
    if IVs is None:
        raise ValueError("You must include IVs")

    if forced_in is not None:
        if isinstance(forced_in, str):
            forced_in = [forced_in]
        # Remove any IVs from the IVs list if they are already being forced in
        IVs = list(set(IVs) - set(forced_in))

    if C_list is None:
        C_list = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]

    # Set up the result table in Pandas
    col_info = OrderedDict()
    col_info['C'] = pd.Series([], dtype='float')
    col_info['Variables'] = pd.Series([], dtype='str')
    col_info['Converged?'] = pd.Series([], dtype='bool')
    col_info['Gini'] = pd.Series([], dtype='float')

    if intercept:
        col_info['Intercept'] = pd.Series([], dtype='float')

    if forced_in is not None:
        vars = forced_in + IVs
    else:
        vars = IVs

    for var in vars:
        col_info["{} Coef".format(var)] = pd.Series([], dtype='float')

    # Create the pandas
    output = pd.DataFrame(col_info)

    scaler = StandardScaler()

    if forced_in is not None:
        model_dataset = dataset[[DV] + IVs + forced_in].dropna()
        model_dataset[IVs + forced_in] = scaler.fit_transform(model_dataset[IVs + forced_in])
        X = model_dataset[forced_in + IVs].copy()
    else:
        model_dataset = dataset[[DV] + IVs].dropna()
        model_dataset[IVs] = scaler.fit_transform(model_dataset[IVs])
        X = model_dataset[IVs].copy()
    
    Y = model_dataset[DV]

    if forced_in is not None:
        X[forced_in] = X[forced_in] * 1000

    for C in C_list:
        lasso = linear_model.LogisticRegression(C=C, fit_intercept = intercept, penalty='l1',
                                                intercept_scaling=1000000, solver='liblinear')

        fitted_model = lasso.fit(X, Y)

        results = OrderedDict()

        results['C'] = C
        results[
            'Converged?'] = fitted_model.n_iter_[0] < fitted_model.max_iter  # Assuming max number of iterations met means no convergence

        predictions = fitted_model.predict_proba(X)[:,1]
        results['Gini'] = 2 * metrics.roc_auc_score(y_true = Y, y_score = predictions) - 1

        if intercept:
            results['Intercept'] = fitted_model.intercept_[0]

        var_list = []
        for var in vars:
            var_coef = fitted_model.coef_[0][vars.index(var)]

            if var in forced_in:
                var_coef = var_coef*1000
            if var_coef != 0:
                var_list.append(var)
            results["{} Coef".format(var)] = var_coef

        results['Variables'] = ";".join(var_list)

        output = pd.concat([output, pd.DataFrame([results])]).reset_index(drop=True)

    return output

