# -*- coding: utf-8 -*-
"""
Created on: 1/21/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from caf.ml.feature_selection.feature_selection_functions import (rf_feature_selection,
                                                                  combine_results,
                                                                  analyse_feature_importance)


def main_feature_selection(train,
                           test,
                           target_column,
                           cv,
                           regression_method,
                           weight_column,
                           binary_prediction,
                           output,
                           skip_feature_selection,
                           intensive_feature_selection):

    if skip_feature_selection:
        return train, test

    if intensive_feature_selection:
        train_final = rf_feature_selection(data=train,
                                           target_column=target_column,
                                           cv=cv,
                                           regression_method=regression_method,
                                           weight_column=weight_column,
                                           binary_prediction=binary_prediction)

        test_final = combine_results(train_final=train_final,
                                     target_column=target_column,
                                     weight_column=weight_column,
                                     test=test)

        return train_final, test_final

    else:
        train_final = analyse_feature_importance(train_transformed=train,
                                                 target_column=target_column,
                                                 weight_column=weight_column,
                                                 output_path=output)

        test_final = combine_results(train_final=train_final,
                                     target_column=target_column,
                                     weight_column=weight_column,
                                     test=test)

        return train_final, test_final
