import shap
import pandas as pd

def get_explanation(model, input_df):

    explainer = shap.Explainer(model.predict, input_df)

    shap_values = explainer(input_df)

    return shap_values