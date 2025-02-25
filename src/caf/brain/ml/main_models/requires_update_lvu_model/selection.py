# -*- coding: utf-8 -*- SELECTION
"""
Created on: 04/09/2023
Updated on: 08/09/2023

Original author: Isaac Scott
Last update made by: Isaac Scott
Other updates made by: Adil Zaheer

File purpose: This file contains variable and model optimisation and selection
functions_to_be_processed that feed into the LVU model.

"""
# IMPORTS
import logging
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import inputs

# Done in order to create a log of run messages
LOG = logging.getLogger(__name__)


"""
Constituents to how the select_param function works. 
These can be edited in order to change how your x variables are evaluated and 
subsequently which are dropped and which remain for the final modelling process.

The current set up for is fairly strong for most models but can be 
optimised depending on your data and forecast years. 
"""


def filter_by_corr_importance(x: pd.DataFrame, y: pd.DataFrame, threshold: float):
    """
    The function performs feature selection.

    Features that are kept are
    favoured by the ExtraTreesRepressor algorithm but are also assessed
    for their correlation properties. Highly correlated features with a
    lower importance are dropped from the final selection of features.
    This results in a set of features that are deemed both important and
    uncorrelated.

    Parameters
    ----------

    x: These are the x variables. These variables are being feature
    selected
    y: Y is the target variable. The variable in which are forecasting
    threshold: This is the correlation threshold that can be set by
    the user. The function will find columns in the correlation matrix with
    correlations > threshold. These have the potential to be dropped.
    Returns
    -------

    old_inputs.CorrImpReturn: See class for info.
    """
    model = ExtraTreesRegressor()
    importances = pd.DataFrame(model.fit(x, y).feature_importances_, index=x.columns)
    corr = x.corr()
    x_orig = x.copy()
    for col in corr.columns:
        ser = corr.drop(col)[col]
        checked = ser[ser > threshold]
        if len(checked) == 0:
            continue
        else:
            for ind in checked.index:
                if importances.loc[ind, 0] > importances.loc[col, 0]:
                    try:
                        # It's possible this has already been dropped
                        x.drop(col, axis=1, inplace=True)
                    except:
                        continue

                try:
                    # Possible this has already been dropped
                    x.drop(ind, axis=1, inplace=True)
                except:
                    continue
    dropped_cols = [i for i in x_orig.columns if i not in x.columns]

    return inputs.CorrImpReturn(
        x=x, corr=corr, importances=importances, dropped_cols=dropped_cols
    )


def select_model(
    x: pd.DataFrame,
    y: pd.DataFrame,
    model_names: list[str] = [en.name for en in list(inputs.Models)],
):
    """
    This function scores different types of user specified models in order to
    find the best performing model for your data. The model comparison is
    based upon their mean r2 score from 5 fold cross validation. The
    standard deviation of the final model is also is shown.

    Parameters
    ----------

    x: The input features
    y: The target variable
    model_names: These are the user specified models that will be
    tested

    Returns
    -------
    The function returns the name of the best performing model
    """
    acc = {}
    score = 0
    # Set dummy variable to return if it fails.
    return_model = "dummy"
    for name in model_names:
        model = inputs.Models[name].value()
        scores = cross_val_score(model, x, y, cv=5, scoring="r2")
        mean = scores.mean()
        acc[name] = mean
        if mean > score:
            score = mean
            return_model = name
        LOG.info(f"{name}: Mean Accuracy: {mean}, Std: {scores.std()}")
    return return_model


def select_param(x, y, model_name):
    """
    This function utilises gridsearch in order to find the best x
    parameters. The parameters are returned and are then used within the final
    forecasting model. See PARAM_GRIDS at the top of selection.py to edit
    how the functionality of the gridsearch.
    :param x: The input features
    :param y: The target variable
    :param model_name: Name of model that will be used to conduct the
    gridsearch. In the case of this model, extratreesregressor is used as
    through prior testing this provided the best r2 results.
    :return:
    """
    if model_name == "dummy":
        return None
    estimator = inputs.Models[model_name].value()
    grid = inputs.ModelGrids[model_name].value
    grid_search = GridSearchCV(estimator, grid, cv=5, scoring="r2")
    grid_search.fit(x, y)
    best_params = grid_search.best_params_
    return best_params
