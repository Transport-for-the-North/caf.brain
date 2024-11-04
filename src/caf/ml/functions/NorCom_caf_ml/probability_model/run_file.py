import sys
from pathlib import Path
from caf.ml.functions.NorCom_caf_ml.probability_model.inputs import NorCom_probability_model_inputs_cafml
from caf.ml.functions.NorCom_caf_ml.probability_model.main import main
sys.path.extend(r"C:\Users\Liberty\Documents\GitHub\caf.ml\src")

if __name__ == "__main__":
    """
    classified_build: Path(r) , always a path directly to the data that will 
                      be modelled. The data for NorCom is typically the 
                      classified build. The data should contain all relevant
                      columns including the column that is being predicted. 
                      The data will be processed and split into relevant
                      training and test sets but the raw input should be the 
                      original dataset in its entirety. 
    output_folder: Path(r), This should be a path to an empty folder where
                   all model outputs are to be written to. 
    target_column: string, this should be a column title in the input data
                   (classified build). This is what the model will be 
                   predicting. 
    index_columns: list, this should be a list of string(s). These are column 
                   titles in the input data that should be indexed. These
                   are relevant to the order of the data but not to be 
                   modelled. e.g LSOA ID. 
    categorical_features: list, this should be a list of string(s). These are 
                          column titles in the input data that are 
                          categorical. 
    numerical_features: list, this should be a list of string(s). These are 
                          column titles in the input data that are 
                          numerical. 
    weight_column: string, this should be a column title in the input data
                   (classified build). This represents the weight to be applied
                   to the model to account for missing trips in the classified
                   build. 
    training_year: string, this should be a year value in the input data. 
                   This will be used to distinguish between the training, test
                   and validation data which is all created from the 
                   input data. This requires the data to have a time component.
                   Original caf.ml model has cross section prediction 
                   capabilities. 
    column_name_to_drop_rows: list, list of strings that represent column 
                              names. These columns will be linked to specific
                              rows that are to be dropped. Leave as default,
                              ['tfn_at'].
                              
    value_in_row: list, list of strings that represent a specific value in a
                  row. The row and value is linked to the column specified in
                  column_name_to_drop_rows. Leave as default, ['20'].
    stats_model: yes or None. If yes, a basic stats models logistic regression
                 will be ran. 
    sklearn_model: yes or None. If yes, a basic sklearn logistic regression
                 will be ran. 
    svm_adaptation: yes or None. If yes, a basic support vector machine using 
                    sklearn will be ran. 
    cafml_version: yes (string) or None. If yes, a version of caf.ml will be ran. 
    improve_data: yes (string) or None. If Yes, modifying_data will be ran.
                  This will be ran as default is there are any issues present
                  in the data. This function results in interaction terms
                  and polynomial feature being added. This is in addition to
                  mca being added in order to attempt to fix the issues 
                  present in the data. MCA will also be done which aims
                  to fix the issues in a way similar to PCA. 
    model_to_use: string, either: gb, rf, dt, svm , svm_binary, logit_l1, 
                  logit_l2, logit_elastic_net, logit_multinomial or None. 
                  These link to the cafml_version argument only. This allows 
                  the choice of gradient boosting, random forest, decision tree 
                  ,support vector machine and logit variations being used in 
                  the caf.ml model. 
    binary_prediction: If prediction is binary, 0 vs 1 or 1 vs 2 then this 
                       must be specified as a string. If left as None then a
                       multiclass prediction will be made (0 vs 1 vs 2+). 
    """
    params = NorCom_probability_model_inputs_cafml(
                                                   classified_build=Path(r"E:\2024 work streams\norcom caf.ml integration\FINAL\new_functions_testing\cb_tfn_v15.csv"),
                                                   output_folder=Path(r"E:\2024 work streams\norcom caf.ml integration\redo_norcom\cafml\0vs1\logit_l1"),
                                                   target_column='numcarvan',
                                                   index_columns=['householdid', 'surveyyear', 'hholdua_b01id'],
                                                   categorical_features=['tfn_at', 'hh_child', 'ns', 'hholdnumadults'],
                                                   numerical_features=None,
                                                   weight_column='w2',
                                                   training_year='2021',
                                                   column_name_to_drop_rows=['tfn_at'],
                                                   value_in_row=['20'],
                                                   stats_model=None,
                                                   sklearn_model=None,
                                                   svm_adaptation=None,
                                                   cafml_version='Yes',
                                                   improve_data='yes',
                                                   model_choice='logit_l1',
                                                   binary_prediction='0vs1'


    )
    main(params)
