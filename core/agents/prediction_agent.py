from langgraph.prebuilt import create_react_agent

from backend.utils.prediction_tools import (
    AMES_classifier,
    BBB_classifier,
    CYP1A2_classifier,
    CYP2C19_classifier,
    CYP2C9_classifier,
    CYP2D6_classifier,
    CYP3A4_classifier,
    PAMPA_classifier,
    PGP_classifier,
    Solubility_regressor,
    Lipophilicity_regressor,
    hERG_classifier,
    predict_repurposedrugs,
)
from core.prompts.prompts import PREDICTION_SYSTEM_PROMPT_ver3


def build_prediction_agent(llm, *, pre_model_hook=None):
    prediction_agent = create_react_agent(
        model=llm,
        tools=[CYP3A4_classifier, 
                CYP2C19_classifier,
                CYP2D6_classifier,
                CYP1A2_classifier,
                CYP2C9_classifier,
                hERG_classifier, 
                AMES_classifier, 
                PGP_classifier, 
                Solubility_regressor, 
                Lipophilicity_regressor,
                PAMPA_classifier,
                BBB_classifier,
                predict_repurposedrugs],
        name='prediction_agent',
        prompt=PREDICTION_SYSTEM_PROMPT_ver3,
        version='v2',
        pre_model_hook=pre_model_hook,
    )

    return prediction_agent
