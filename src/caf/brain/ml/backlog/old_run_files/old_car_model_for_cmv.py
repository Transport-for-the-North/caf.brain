# standard imports
import numpy as np
import pandas as pd

# model specific imports
import os
from pathlib import Path
from sklearn.model_selection import KFold, GridSearchCV
from sklearn import linear_model
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import explained_variance_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from caf.ml.backlog.old_inputs import CarInputs2
ALLOWED_MODELS = (ElasticNet, Lasso, Ridge)

from caf.ml.backlog.functions_to_be_processed import DataProcessor
from tidy_data_function import main_tdf
from feature_selection_function_storage import feature_selection
from numeric_data import process_data_numeric


# Save each DataFrame as a separate CSV file
def save_dataframes_to_folder(dataframes_dict, filenames_dict, folder_path):
    # os used in order to output all files with specific names
    for name, df in dataframes_dict.items():
        if name in filenames_dict:
            filename = filenames_dict[name]
            file_path = os.path.join(folder_path, filename)
            df.to_csv(file_path, index=False)


# feature selection function: optimal alpha value for optimal variable combination is calculated
def test_function(df11, num_folds=5, regression_method=ElasticNet):
    """convergence warnings are expected when running this mode, this is not a cause for concern.

    The test_function conducts feature selection in order to determine a combination of features
    that can best explain and therefore prediction_model the target variable.
    The function utilises machine learning through the train and old_run_files split being applied to
    the chosen regression algorithm.
    GridSearch (another machine learning method) is then used to score each iteration, eventually
    selected_features_df is produced which is a potential optimal combination of features.
    This may not be the final best combination but this improvement is to be added in the future.

    In order to provide some understanding within this model, the final selected features are then
    reapplied to a train and old_run_files (ML) split in another function. Hyper-Parameter optimisation is
    then again conducted with a score being produced. This score can help validate how well
    the feature selection performed.
    """
    # Extract X and y
    X = df11.drop(columns='sum_cars').values
    y = df11['sum_cars'].values

    # Apply feature scaling, each feature has mean of 0 and SD of 1 (standard practice for this form of ML)
    scale = StandardScaler()
    X_scaled = scale.fit_transform(X)

    # Initialise (done so created variables have a starting point)
    alpha_range = np.arange(0.1, 10, 0.1)
    # used for the grid search, creates dictionary with each dictionary key containing an alpha range value
    param_grid = {'alpha': alpha_range}
    best_alpha = None
    # set to infinite as best score will always be lower
    best_score = float('inf')
    final_model = None

    # Outer cross-validation loop for model evaluation
    outer_scores = []
    selected_features = None
    X_selected_combined = None

    # data split into training and old_run_files with 5 folds
    for train_index, test_index in tqdm(KFold(n_splits=num_folds, shuffle=True).split(X_scaled),
                                        desc="Outer CV Progress"):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Inner cross-validation loop for feature selection
        # if else statements in order to select which algorithm to used based on user input
        selector = SelectFromModel(estimator=regression_method())
        final_model_inner = regression_method(max_iter=1000)

        # applies feature selection to training and target data
        selector.fit(X_train, y_train)
        # gets binary mask of features (true and false), which features to include in final model
        selected_features = selector.get_support()
        # applies the binary mask to original xtrain to produce final feature selection combination
        X_selected = selector.transform(X_train)

        # in 1st loop iteration, x_scaled_combined is initialised with zeros in the same shape as x_scaled
        # x_selected_combined eventually becomes the best features from each iteration of feature selection
        if X_selected_combined is None:
            X_selected_combined = np.zeros((X_scaled.shape[0], X_scaled.shape[1]))
        # updated x_selected_combined with features from x_selected
        X_selected_combined[train_index[:, np.newaxis], selected_features] = X_selected

        # Perform grid search with cross-validation to find best alpha for selected features
        grid_search = GridSearchCV(estimator=final_model_inner, param_grid=param_grid,
                                   scoring='neg_mean_squared_error', cv=num_folds)
        grid_search.fit(X_selected, y_train)

        # Get the best alpha and best score from inner loop
        best_alpha_inner = grid_search.best_params_['alpha']
        best_score_inner = -grid_search.best_score_

        # Fit the final model with the best alpha from inner loop
        final_model_inner.alpha = best_alpha_inner
        final_model_inner.fit(X_selected, y_train)

        # Evaluate the model on the outer old_run_files set
        X_selected_test = selector.transform(X_test)
        outer_score = final_model_inner.score(X_selected_test, y_test)
        outer_scores.append(outer_score)

        # Update the best alpha and score if necessary
        if best_score_inner < best_score:
            best_score = best_score_inner
            best_alpha = best_alpha_inner
            final_model = final_model_inner

    print("Best Alpha:", best_alpha)
    print("Best Score:", best_score)
    print("Selected Features:", np.array(df11.columns[:-1])[selected_features])
    print("Mean Outer Score:", np.mean(outer_scores))
    if best_alpha >= 9:
        print(f"Alpha value is {best_alpha}. Consider increasing alpha range.")

    selected_features_df = pd.DataFrame(X_selected_combined[:, selected_features],
                                        columns=np.array(df11.columns[:-1])[selected_features])

    return final_model, selected_features_df


def apply_feature_selection(selected_features_df, df21):
    # Get the column names from the feature selection DataFrame
    selected_features = selected_features_df.columns

    # Filter the columns of the DataFrame to be applied based on the selected features
    df_subset = df21[selected_features]

    # Apply feature scaling to the selected subset of the DataFrame
    scaler = StandardScaler()
    scaled_subset = scaler.fit_transform(df_subset)

    # Create a new DataFrame using the scaled features and the selected column names
    scaled_df = pd.DataFrame(scaled_subset, columns=selected_features)

    return scaled_df


def calculate_correlation(x1, x2, y1, y2):
    # access linearity assumptions
    correlations = {}

    for column in x1.columns:
        if column in x2.columns:
            correlation1, _ = pearsonr(x1[column], y1)
            correlation2, _ = pearsonr(x2[column], y2)
            correlations[column] = {'correlation1': correlation1, 'correlation2': correlation2}

    correlation_df = pd.DataFrame(correlations).T
    return correlation_df


def evaluate_independence(x1, x2, y1, y2):
    # Calculate the Spearman's rank correlation coefficient for each feature in X1 and X2
    spearman_1 = {}
    spearman_2 = {}

    for column in x1.columns:
        corr_1, _ = spearmanr(x1[column], y1)
        corr_2, _ = spearmanr(x2[column], y2) if column in x2.columns else (None, None)
        spearman_1[column] = corr_1
        spearman_2[column] = corr_2

    spearman_df_1 = pd.DataFrame.from_dict(spearman_1,
                                           orient='index', columns=['Spearman Correlation 2011'])
    spearman_df_2 = pd.DataFrame.from_dict(spearman_2,
                                           orient='index', columns=['Spearman Correlation 2021'])

    results = pd.concat([spearman_df_1, spearman_df_2], axis=1)

    return results


# multicollinearity old_run_files using the VIF formula
def test_multicollinearity(x):
    vif_results = pd.DataFrame()
    vif_results["Feature"] = x.columns
    vif_results["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    return vif_results


# processes data to be ready for hyperparamter (alpha) optimisation
def df_hyperparam_optimisation(df11: pd.DataFrame,
                               modified_df_11: pd.DataFrame, target_column: str):
    # made lists in order to remove expected warning
    y = df11[target_column].values
    x = modified_df_11.values
    alpha_vals = list(np.arange(1, 10, 1))
    return hyperparam_optimisation(x, y, alpha_vals)


def hyperparam_optimisation(data_x: np.ndarray, data_y: np.ndarray, alpha_vals: list[float]):
    # Find the best alpha value
    # CV each alpha value
    # uses cross validation function
    results = list()
    for alpha in tqdm(alpha_vals):
        mse = cross_validation(data_x, data_y, alpha, k_folds=5)
        results.append((alpha, mse))

    # Pick lowest MSE
    sorted_results = sorted(results, key=lambda x: x[1])
    best_alpha = sorted_results[0][0]
    return best_alpha


# applies specified modelling technique on the final selection of variables and alpha
def retrained_model(x_train, x_test, y_train, alpha: float,
                    regression_method='ElasticNet'):
    if regression_method == 'ElasticNet':
        model = linear_model.ElasticNet(alpha=alpha)
    elif regression_method == 'Lasso':
        model = linear_model.Lasso(alpha=alpha)
    elif regression_method == 'Ridge':
        model = linear_model.Ridge(alpha=alpha)
    else:
        raise ValueError("Invalid regression method.")

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred


# Performs cross validation on train and old_run_files data
def cross_validation(x: np.ndarray, y: np.ndarray, alpha: float, k_folds: int = 5) -> float:
    # define number of folds for kfold cv (based on size of data)
    kf = KFold(n_splits=k_folds)

    # empty list created to store each iteration of the cv as it goes through kth number of folds
    all_test = list()
    all_pred = list()

    # provides train/old_run_files indices to split data into train/old_run_files, repeated k times, represented by i
    for i, (train_index, test_index) in enumerate(kf.split(x, y)):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        # y prediction_model is
        y_pred = retrained_model(
            alpha=alpha,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
        )
        all_test.append(y_test.flatten())
        all_pred.append(y_pred)

    all_test = np.hstack(all_test)
    all_pred = np.hstack(all_pred)
    return mean_squared_error(all_test, all_pred)


# model evaluation function
def evaluate_model(y_true, y_pred):
    # score calculated directly in dictionary for dictionary literal warning
    scores = {
        'R2 Score': r2_score(y_true, y_pred),
        'Mean Absolute Error': mean_absolute_error(y_true, y_pred),
        'Mean Absolute Percentage Error': mean_absolute_percentage_error(y_true, y_pred),
        'Explained Variance Score': explained_variance_score(y_true, y_pred)
    }
    # Convert scores to a DataFrame
    df_scores = pd.DataFrame.from_dict(scores, orient='index', columns=['Score'])

    return df_scores


###################### NEW FUNCTIONS ######################



def main(params: CarInputs2, output_folder, reg_method, custom_regression_methods):
    """ Main function requires the following old_inputs:
    1) Path to the 2011 census data/base year census data you are using
    2) A forecast year set of data
    3) A regression method (Lasso, ElasticNet or Ridge)
    4) A target column (Y variable - what you want to prediction_model)
    These are all specified where main is called below.

    The main function will call all of the above functions_to_be_processed. It will conduct:
    - Feature selection & scaling (test_function)
    - Relevant assumption tests (calculate_correlations, evaluate_independence, test_multi.)
    - Hyper-Parameter optimisation - finding optimal alpha using ML methods
    - The final model is then called (retrained_model)
    - Final model evaluation and then output folders (evaluate_model)

    Prerequisites for model use:
    - The data provided (i.e path_2011, path_2021) must be in the same format.
    The two csv documents must have identical column names and be in the same order.
    The file should also include the target (Y) variable.
    The predicted variable year data does not need to include the target (Y) """
    # process data: raw data -> model format
    data_processor = DataProcessor()
    data = data_processor.process_data(params.x, params.y, params.folder_path,
                                       params.index1, params.index2, params.wide_format,
                                    params.variable_name, params.value_name)
    # tidy data: remove correlated values etc.
    data_to_model = main_tdf(data, output_folder)

    data_final = process_data_numeric(data_to_model)

    final_model, selected_features_df = feature_selection(data_final, num_folds=5,
                                                          selected_algorithm=reg_method,
                                                          target_column=params.target_column)

    print('hi', selected_features_df)

    # applying feature selection to 2021 census data (old_run_files data (21) must match training data (11))
    scaled_df = apply_feature_selection(selected_features_df, df21)
    modified_df_11 = selected_features_df
    modified_df_21 = scaled_df

    # create x and y dataframes for both datasets (X1/Y1 = 2011, X2/Y2 = 2021)
    x1 = modified_df_11
    y1 = df11[params.target_column]
    x2 = modified_df_21
    y2 = df21[params.target_column]

    # calculate correlations using x/y dataframes
    correlations = calculate_correlation(x1, x2, y1, y2)

    # calculate independent and identical assumption
    iid = evaluate_independence(x1, x2, y1, y2)

    # Multicollinearity old_run_files (VIF)
    vif_results_1 = test_multicollinearity(x1)
    vif_results_2 = test_multicollinearity(x2)

    # finds the best alphas for the dataframe post feature selection
    best_alpha = df_hyperparam_optimisation(df11, modified_df_11, params.target_column)

    # create final prediction model
    y_2011 = df11[[params.target_column]].values
    x_2011 = x1.values
    x_21 = x2.values
    y_pred = retrained_model(x_train=x_2011, x_test=x_21, y_train=y_2011,
                             alpha=best_alpha, regression_method=params.method.__name__)

    y_pred_final = pd.DataFrame(y_pred, columns=['y_pred'])

    # final evaluation tests to see accuracy of prediction
    y_true = df21[params.target_column]
    evaluation_scores = evaluate_model(y_true, y_pred)

    # dictionaries to store outputs from model tests and outputs
    dataframes = {
        'modified_df_11': modified_df_11,
        'modified_df_21': modified_df_21,
        'correlations': correlations,
        'iid': iid,
        'vif_results_1': vif_results_1,
        'vif_results_2': vif_results_2,
        'y_pred_final': y_pred_final,
        'evaluation_scores': evaluation_scores
    }

    filenames = {
        'modified_df_11': 'modified_df_11.csv',
        'modified_df_21': 'modified_df_21.csv',
        'correlations': 'correlations.csv',
        'iid': 'iid.csv',
        'vif_results_1': 'vif_results_1.csv',
        'vif_results_2': 'vif_results_2.csv',
        'y_pred_final': 'y_pred_final.csv',
        'evaluation_scores': 'evaluation_scores.csv',
    }

    save_dataframes_to_folder(dataframes, filenames, params.folder)

    return


if __name__ == "__main__":
    """PATH_2011 and PATH_2021 are current links to a local drive. These of 
       are included to keep functionality of the model. These paths would be 
       replaces for your specific use of the model."""
    params = CarInputs2(x=Path(r"E:\caf.ml\data_process_function\test_data\cb_tfn_v12_smallerversion.csv"),
                        y=None,
                        folder_path=None,
                        index1='SurveyYear',
                        index2=None,
                        wide_format=None,
                        variable_name=None,
                        value_name=None,
                        method=Ridge,
                        target_column='weighted_trips',
                        folder=Path(r"E:\caf.ml\data_process_function\car_model_results"))
    output_folder = r"E:\caf.ml\data_process_function\test_data\output"
    reg_method = Lasso
    custom_regression_methods = [Lasso, ElasticNet]
# path 2011 r"E:\TRSE\data\final\sorted_data\data_no_geography\sum_car\2021finalsumcar.csv"
# path 2021 r"E:\TRSE\data\final\sorted_data\data_no_geography\sum_car\2021finalsumcar.csv"
    main(params, output_folder, reg_method, custom_regression_methods)
