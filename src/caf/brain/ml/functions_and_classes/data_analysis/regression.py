# Third Party
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_rainbow


def assumption_analysis(model):
    alpha = 0.05  # Standard alpha level for significance
    linearity_present = False
    normality_present = False
    heteroscedasticity_present = False

    # Rainbow Test for Linearity
    rainbow_statistic, rainbow_p_value = linear_rainbow(model)
    print(f"Rainbow test p-value: {rainbow_p_value}")
    if rainbow_p_value < alpha:
        print("Warning: Rainbow test suggests non-linearity.")
        linearity_present = True

    # Shapiro-Wilk Test for Normality
    shapiro_statistic, shapiro_p_value = shapiro(model.resid)
    print(f"Shapiro-Wilk test p-value: {shapiro_p_value}")
    if shapiro_p_value < alpha:
        print("Warning: Shapiro-Wilk test suggests non-normality of residuals.")
        normality_present = True

    # Breusch-Pagan Test for Heteroscedasticity
    bp_test_statistic, bp_test_p_value, _, _ = het_breuschpagan(model.resid, model.model.exog)
    print(f"Breusch-Pagan test p-value: {bp_test_p_value}")
    if bp_test_p_value < alpha:
        print("Warning: Breusch-Pagan test suggests heteroscedasticity.")
        heteroscedasticity_present = True

    # White's Test for Heteroscedasticity
    white_test_statistic, white_test_p_value, _, _ = het_white(model.resid, model.model.exog)
    print(f"White's test p-value: {white_test_p_value}")
    if white_test_p_value < alpha:
        print("Warning: White's test suggests heteroscedasticity.")
        heteroscedasticity_present = True

    return linearity_present, normality_present, heteroscedasticity_present
