""" ACTUAL MODEL
Created on: 27/07/2023
Updated on: 08/09/2023

Original author: Adil Zaheer
Last update made by: Adil Zaheer
Other updates made by: Isaac Scott

File purpose: Land Value uplift model. Forecast rateable value

"""
# IMPORTS
import pandas as pd
import os
import logging
from pathlib import Path
import inputs
from inputs import *
from LVU import selection, model

# LOG CREATION
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level to capture (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Send log messages to the console
        logging.FileHandler("LVU_LOG.log"),  # Save log messages to a file
    ],
)
LOG = logging.getLogger(__name__)


def convert_string(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def main(params: LvuInputs):
    """
    Run time: approximately 18 minutes
    The main function carries out the following steps:
    1) Dataframes are read in. These are a combination of csv's,
    specific NTEM data or custom x data.
    2) The read in X variables are then transformed into an appropriate
    format. This format contains all of the x variables in an orientation
    best suited to the model.
    3) The variables are then processed by being transformed to contain
    both historical and future x values. The x variables are also scaled using
    the standard scaler module (centres data around zero with a standard
    deviation of 1).
    4) Correlations and importance's calculated. These are outputted to the
    specified output folder and help provide context on how valid the variables
    inputted are for forecast modelling.
    5) Using extratrees regression algorithm, the x variables are ranked and
    are included in the final model if they have an adequate R2 value.
    6) Model evaluation is conducted in order to assess the validity of the
    final model (Mean standard error and R2)
    7) The final model is called and the y prediction is made. This is
    outputted to the output folder in CSV format.
    """
    # 1) read dataframes in
    dataframes = []
    if params.folder_path is not None:
        dataframes = read_multiple_files(params.folder_path)
    ntem_data = []
    if params.ntem_data is not None:
        ntem_data = process_ntem(params.ntem_data)
    custom_x = []
    if params.custom_x is not None:
        for x in params.custom_x:
            new_x = XVar(
                geog_col=x.geog_col,
                year_col=x.year_col,
                val_name=x.val_name,
                dataframe=x.load_custom_file(),
            )
            custom_x.append(new_x)

    x_data = dataframes + ntem_data + custom_x
    y = pd.read_csv(params.y_data_path, index_col=0)
    if len(y.columns) == 1:
        y.columns = [params.base_year]
    else:
        y.columns = y.columns.astype(int)
    # 2) get x variables in a appropriate format
    final_x_data = x_variable_processing(x_data)

    # 3) process variables for ML process
    x = tuning_features(final_x_data).dropna()
    hist_years = y.columns
    x_hist = x.loc[:, hist_years, :].copy()
    x_pred = x.loc[:, params.target_years, :]
    # y data comeins in wide format. Here it is stacked to add to the index level
    # index names are set to match x, and then it is joined to x to match index
    # order
    y = y.stack()
    y.index.names = x.index.names

    # 4) Begin statistical analysis of x variables
    LOG.info("Filtering X features by correlation and model importance.")
    x_hist_corr_imp = selection.filter_by_corr_importance(x_hist, y, 0.95)
    LOG.info(
        f"The following X features have been dropped and won't be included in the final model:"
        f"{x_hist_corr_imp.dropped_cols}"
    )
    x_hist_corr_imp.corr.to_csv(params.output_folder / "x_correlations.csv")
    x_hist_corr_imp.importances.to_csv(params.output_folder / "x_importances.csv")
    LOG.info(
        "X correlations and importances have been written to the output folder."
        " It is strongly recommended that these are checked by the user."
    )

    # 5) Begin ranking of dependent variables (x)
    if params.model is None:
        model_name = selection.select_model(x_hist_corr_imp.x, y)
    else:
        model_name = params.model
    if params.model_params is None:
        LOG.info(
            "Performing grid search to determine the best parameters for the model. This can"
            " take some time."
        )
        model_params = selection.select_param(x_hist_corr_imp.x, y, model_name)
    else:
        model_params = params.model_params
    if model_params is None:
        LOG.warning(
            "No model had an acceptable accuracy, so the model is being stopped."
        )
        return None
    LOG.info("Best params for model determined to be: %s", model_params)
    ml_mod = inputs.Models[model_name].value(**model_params)

    # 6) Internal model analysis and validation
    mean_mse, y_pred_list, mean_actual_r_squared = model.cross_validation(
        x_hist_corr_imp.x, y, params.folds, ml_mod
    )
    y_pred = pd.DataFrame(y_pred_list, columns=["y_pred"])
    LOG.info(f"R2 score: {mean_actual_r_squared}")
    LOG.info(f"MSE: {mean_mse}")

    # 7) Forecasting model
    ml_mod.fit(x_hist_corr_imp.x, y)
    prediction = pd.DataFrame(
        ml_mod.predict(x_pred[x_hist.columns]), index=x_pred.index, columns=["y"]
    )
    params.model_params = model_params
    params.model = model_name
    params.r2 = mean_actual_r_squared
    params.mse = mean_mse
    return prediction, params


if __name__ == "__main__":
    """
    This forecast model uses three other scripts:
    - Selection.py
    - Model.py
    - Loading.py
    All can be found in the LVU folder in Normits_ML on GitHub.

    data_path = path to ntem based data
    y_data_path = path to your y data (variable to forecast)

    custom_x: (information that can be found in your data's csv)
    dataframe = path for your x data that will be used in the forecast
    geog_col = geography of the x data
    year_col = name of the year column of the x data
    val_name = x data column name

    base_year = final year of your historical data (training data)
    target_year = the year which you would like to forecast to (this must
    be a year in which you have x variables data up to)
    output_folder = path to where you want model outputs to go
    """
    ntem = NtemInputInfo(
        data_path=r"I:\Transfer\IS\EVCI_inputs\ntem.csv",
        cat_cols=["PlanningDataType"],
        zone_col="ZoneID",
    )
    y_dir = Path(r"I:\Transfer\IS\EVCI_inputs\y")
    for file in os.listdir(y_dir):
        if file.endswith(".csv"):
            params = LvuInputs(
                ntem_data=ntem,
                y_data_path=y_dir / file,
                custom_x=[
                    XVar(
                        dataframe=Path(
                            r"I:\Transfer\IS\EVCI_inputs\gva_inc_forecast.csv"
                        ),
                        geog_col="ZoneID",
                        year_col="year",
                        val_name="gva",
                    )
                ],
                folds=10,
                base_year=2018,
                target_years=[2020, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060],
                output_folder=Path(r"I:\Transfer\IS\EVCI_inputs\outputs"),
            )
            yml_name = file.split(".")[0] + ".yml"
            if os.path.isfile(params.output_folder / yml_name):
                previous = LvuInputs.load_yaml(params.output_folder / yml_name)
                if params.equal_no_return(previous):
                    params.model = previous.model
                    model_params = previous.model_params
                    for key, val in model_params.items():
                        model_params[key] = convert_string(val)
                    params.model_params = model_params
            prediction, model_params = main(params)
            if prediction is None:
                continue
            prediction.to_csv(params.output_folder / file)

            params.save_yaml(params.output_folder / yml_name)
