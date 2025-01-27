# -*- coding: utf-8 -*-
"""
Created on: 2/15/2024
Original author: Adil Zaheer
"""

import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from caf.ml.backlog.functions_to_be_processed import select_model
from caf.ml.backlog.old_inputs import CV_models, Models, Default_regression_methods, ModelGrids
from caf.ml.backlog.functions_to_be_processed import get_cv_class
warnings.filterwarnings("ignore", category=UserWarning)


def feature_selection_with_optimization(df,
                                        splits,
                                        repeats,
                                        model_type,
                                        target_column,
                                        cv_method=None):

    # DATA PROCESSING
    x = df.drop(columns=[target_column])
    y = df[target_column]
    scale = StandardScaler()
    x_scaled = scale.fit_transform(x)

    # KEY MODEL CHECKS
    if cv_method and cv_method not in CV_models:
        raise ValueError(f"Invalid cross-validation method: {cv_method}")

    valid_models = [model.value for model in Models] + Default_regression_methods
    for model in model_type:
        if model not in valid_models:
            raise ValueError(f"Invalid model type: {model}. Please provide valid model types.")

    # MODEL OPTIMIZATION
    if model_type is None:
        regression_methods = Default_regression_methods
        model = select_model(x, y, regression_methods)
    elif len(model_type) == 1:
        regression_methods = [model_type[0].value]
        model = select_model(x, y, regression_methods)
    else:
        model = select_model(x, y, model_type)

    # HYPERPARAMTER OPTIMISATION INITIALISATION
    model_name = model.__class__.__name__

    if model_name in ModelGrids.__members__:
        param_grid = ModelGrids[model_name].value
    else:
        raise ValueError(f"Parameter grid not found for model class: {model_name}")

    # MODEL AND PARAMETER INITIALISATION
    best_alpha = None
    best_score = float('inf')
    final_model = None
    outer_scores = []
    selected_features = None
    X_selected_combined = np.zeros_like(x_scaled)

    # CV METHOD INITIALIZATION
    cv_class = get_cv_class(cv_method, splits, repeats) if cv_method else KFold
    cv = cv_class(n_splits=splits, n_repeats=repeats) if "repeated" in str(cv_method) else cv_class(n_splits=splits)

    # CROSS VALIDATION
    for train_index, test_index in tqdm(cv.split(x_scaled), desc="Outer CV Progress"):
        X_train, X_test = x_scaled[train_index], x_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # INNER CV FOR FEATURE SELECTION
        selector = SelectFromModel(estimator=model)
        final_model_inner = model(max_iter=1000)
        selector.fit(X_train, y_train)
        selected_features = selector.get_support()
        X_selected = selector.transform(X_train)
        X_selected_combined[train_index[:, np.newaxis], selected_features] = X_selected

        # GRID SEARCH FOR HYPERPARAMETER OPTIMIZATION
        grid_search = GridSearchCV(estimator=final_model_inner, param_grid=param_grid,
                                   scoring='neg_mean_squared_error', cv=cv)
        grid_search.fit(X_selected, y_train)

        # UPDATE BEST ALPHA AND SCORE
        best_alpha_inner = grid_search.best_params_['alpha']
        best_score_inner = -grid_search.best_score_
        final_model_inner.alpha = best_alpha_inner
        final_model_inner.fit(X_selected, y_train)

        # OUTER SCORE
        X_selected_test = selector.transform(X_test)
        outer_score = final_model_inner.score(X_selected_test, y_test)
        outer_scores.append(outer_score)

        # UPDATE BEST ALPHA AND SCORE IF NECESSARY
        if best_score_inner < best_score:
            best_score = best_score_inner
            best_alpha = best_alpha_inner
            final_model = final_model_inner

    # PRINT RESULTS
    print("Best Alpha:", best_alpha)
    print("Best Score:", best_score)
    print("Selected Features:", np.array(df.columns[:-1])[selected_features])
    print("Mean Outer Score:", np.mean(outer_scores))

    selected_features_df = pd.DataFrame(X_selected_combined[:, selected_features],
                                        columns=np.array(df.columns[:-1])[selected_features])

    return final_model, selected_features_df
