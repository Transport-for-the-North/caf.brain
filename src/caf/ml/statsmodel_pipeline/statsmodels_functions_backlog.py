# -*- coding: utf-8 -*-
"""
Created on: 1/30/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position


# def select_model_stats(train: pd.DataFrame,
#                        target_column: str,
#                        weight_column: str,
#                        models_to_test: list[Models],
#                        output_folder: Path):
#
#     weight = None
#     y = train[target_column]
#     x = train.drop(columns=target_column)
#     if weight_column in train.columns:
#         weight = x[weight_column]
#         x = x.drop(columns=weight_column)
#
#     x_with_const = sm.add_constant(x)
#     acc = {}
#     best_score = float('-inf')
#     best_model = None
#
#     for model_enum in models_to_test:
#         model_name = model_enum.name
#         print(f"Testing model: {model_name}")
#
#         # statsmodels
#         scores_r2, scores_mse = score_statsmodel(
#             model_enum=model_enum,
#             x=x_with_const,
#             y=y,
#             weight=weight
#         )
#         mean_score = scores_r2.mean()
#         acc[model_enum] = {'R-squared': scores_r2.mean(), 'MSE': scores_mse.mean()}
#
#         if mean_score > best_score:
#             best_score = mean_score
#             model_class, params = model_enum.value
#             x_init = sm.add_constant(x)
#             best_model = model_class(y, x_init)
#
#     print(f"Best model: {best_model}")
#     print(f"Best model score: {best_score}")
#     evaluation_df = pd.DataFrame.from_dict(acc, orient='index')
#
#     output_filename = 'model_algorithm_evaluation.csv'
#     output_path = os.path.join(output_folder, output_filename)
#     evaluation_df.to_csv(output_path, index=True)
#
#     return best_model
#
#
# def score_statsmodel(model_enum, x, y, weight):
#     kf = KFold(n_splits=3, shuffle=True, random_state=42)
#     model_class, params = model_enum.value
#     is_classification = hasattr(model_class, 'predict_proba')
#
#     scores_primary = []  # r² for regression, F1 for classification
#     scores_secondary = [] # MSE for regression, AUC for classification
#     fit_params = params.copy()
#     fit_params.update({
#         'maxiter': 1000,
#         'disp': False
#     })
#
#     for fold_num, (train_idx, test_idx) in enumerate(kf.split(x), 1):
#         try:
#             x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
#             y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#
#             if weight is not None and train_idx is not None:
#                 w_train = weight.iloc[train_idx]
#                 model = model_class(y_train, x_train, freq_weights=w_train)
#             else:
#                 model = model_class(y_train, x_train)
#
#             fit_model = model.fit(**fit_params)
#
#             if is_classification:
#                 y_pred_proba = fit_model.predict(x_test)
#                 y_pred = (y_pred_proba > 0.5).astype(int)
#                 f1 = f1_score(y_test, y_pred)
#                 auc = roc_auc_score(y_test, y_pred_proba)
#                 scores_primary.append(f1)
#                 scores_secondary.append(auc)
#             else:
#                 y_pred = fit_model.predict(x_test)
#                 mse = np.mean((y_test - y_pred) ** 2)
#                 r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
#                 scores_primary.append(r2)
#                 scores_secondary.append(mse)
#
#         except Exception as e:
#             print(f"Warning: Fold {fold_num} failed with error: {str(e)}")
#             # bad scores as model failed
#             scores_primary.append(-999 if is_classification else -np.inf)
#             scores_secondary.append(999)
#             continue
#
#     return np.array(scores_primary), np.array(scores_secondary)
#
#
# def find_coeffs_stats(train,
#                target_column,
#                output_folder,
#                weight_column,
#                model_initialised):
#     x = train.drop(columns=[target_column])
#     y = train[target_column]
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)
#
#     weight = None
#     if weight_column in train.columns:
#         weight_df = x_train[weight_column]
#         weight = weight_df.values.flatten()
#         x_train = x_train.drop(columns=weight_column)
#         x_test = x_test.drop(columns=weight_column)
#
#
#     model_filename = os.path.join(output_folder, 'initial_fitted_model.pkl')
#
#     if os.path.exists(model_filename):
#         model_fit = joblib.load(model_filename)
#         x_test_const = sm.add_constant(x_test)
#         y_pred = model_fit.predict(x_test_const)
#
#     else:
#         x_train_const = sm.add_constant(x_train)
#         x_test_const = sm.add_constant(x_test)
#
#         fit_params = {
#             'maxiter': 1000,
#             'disp': False,
#             'method': 'newton'
#         }
#
#         if isinstance(model_initialised, sm.GLM):
#             model_fit = model_initialised.__class__(
#                 endog=y_train,
#                 exog=x_train_const,
#                 family=model_initialised.family,
#                 weights=weight if weight is not None else None).fit(**fit_params)
#         else:
#             if weight is not None:
#                 model_fit = model_initialised.__class__(
#                     endog=y_train,
#                     exog=x_train_const,
#                     weights=weight).fit(**fit_params)
#             else:
#                 model_fit = model_initialised.__class__(
#                     endog=y_train,
#                     exog=x_train_const).fit(**fit_params)
#         y_pred = model_fit.predict(x_test_const)
#
#         joblib.dump(model_fit, model_filename)
#
#     residuals = y_test - y_pred
#
#     coeff_df = pd.DataFrame({
#         'Feature': ['const'] + list(x_train.columns),
#         'Coefficient': model_fit.params,
#         'Std_Error': model_fit.bse,
#         'T_Value': model_fit.tvalues,
#         'P_Value': model_fit.pvalues
#     })
#     if hasattr(model_fit, 'coef_'):
#         coefficients = model_fit.coef_
#         coefficients = np.squeeze(coefficients)
#
#         if coefficients.ndim == 1:
#             coeff_df = pd.DataFrame({
#                 'Feature': x_train.columns,
#                 'Coefficient': coefficients
#             })
#         else:
#             coeff_df = pd.DataFrame(coefficients.T, columns=x_train.columns)
#             coeff_df.insert(0, 'Feature', x_train.columns)
#
#     if coeff_df is not None:
#         coeff_df.to_csv(os.path.join(output_folder, 'initial_model_coefficients.csv'),
#                         index=False)
#
#     return model_fit, residuals, x_train, x_test, y_train, y_test, coeff_df
#
#
# def statsmodels_feature_selection(coeff_df,
#                                   train_transformed,
#                                   target_column,
#                                   weight_column):
#     df = coeff_df
#     significance_level = 0.05
#     final_cols = df[df['P_Value'] < significance_level]['Feature'].tolist()
#     cols_to_include = final_cols + [weight_column] + [target_column]
#     train_final = train_transformed[cols_to_include]
#
#     return train_final
#
#
# def stats_prediction(model,
#                      test,
#                      target_column,
#                      output_folder,
#                      validation,
#                      weight_column,
#                      binary_prediction):
#
#     if target_column in test.columns:
#         test = test.drop(columns=target_column)
#     if weight_column in test.columns:
#         weight = test[weight_column].values.flatten()
#         test = test.drop(columns=weight_column)
#     else:
#         weight = None
#
#     test_with_const = sm.add_constant(test)
#
#     if binary_prediction is not None:
#         pred_probs = model.predict(test_with_const)
#
#         if len(binary_prediction) == 2:
#             # Binary
#             pred_classes = (pred_probs > 0.5).astype(int)
#             predictions = np.where(pred_classes == 1, binary_prediction[1], binary_prediction[0])
#         else:
#             # Multiclass
#             if pred_probs.ndim > 1:
#                 pred_indices = np.argmax(pred_probs, axis=1)
#             else:
#                 pred_indices = (pred_probs > 0.5).astype(int)
#             predictions = np.array(binary_prediction)[pred_indices]
#
#         if validation is not None:
#             y_true = validation[target_column].values
#             accuracy = accuracy_score(y_true, predictions, sample_weight=weight)
#             print(f'Accuracy: {accuracy}')
#
#             accuracy_df = pd.DataFrame({'accuracy': [accuracy]})
#             accuracy_df.to_csv(os.path.join(output_folder, 'model_performance.csv'))
#
#     else:
#         # Continuous target
#         predictions = model.predict(test_with_const)
#
#         if validation is not None:
#             y_true = validation[target_column].values
#             r2 = r2_score(y_true, predictions, sample_weight=weight)
#             mse = mean_squared_error(y_true, predictions, sample_weight=weight)
#             print(f'R2: {r2}')
#             print(f'MSE: {mse}')
#
#             metrics_df = pd.DataFrame({'r2': [r2], 'mse': [mse]})
#             metrics_df.to_csv(os.path.join(output_folder, 'model_performance.csv'))
#
#     if hasattr(model, 'params'):
#         coeff_df = pd.DataFrame({
#             'Feature': ['const'] + list(test.columns),
#             'Coefficient': model.params,
#             'Std_Error': model.bse,
#             'T_Value': model.tvalues,
#             'P_Value': model.pvalues
#         })
#         coeff_df.to_csv(os.path.join(output_folder, 'final_model_coefficients.csv'), index=False)
#
#     final_predictions = pd.DataFrame({'predicted_target_column': predictions}, index=test.index)
#     final_predictions.to_csv(os.path.join(output_folder, 'final_predictions.csv'))
#
#     return predictions
