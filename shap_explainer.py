from model import MultimodalT2DPredictor
import shap
import torch
import numpy as np

# === Model Wrapper for SHAP ===
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Split concatenated input tensor into original modality tensors
        ehr      = x[:, :8]
        lifestyle = x[:, 8:23]
        synthea   = x[:, 23:29]
        pima      = x[:, 29:37]
        cdc       = x[:, 37:58]
        hosp      = x[:, 58:102]
        return self.model(ehr, lifestyle, synthea, pima, cdc, hosp)

# === Main explanation function ===
def explain_prediction(model_path, sample):
    model = MultimodalT2DPredictor(
        ehr_dim=8,
        lifestyle_dim=15,
        synthea_dim=6,
        pima_dim=8,
        cdc_dim=21,
        hosp_dim=44
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    wrapped_model = WrappedModel(model)

    # Concatenate all sample modalities into a single tensor for SHAP
    inputs = torch.cat([
        sample['ehr'],
        sample['lifestyle'],
        sample['synthea'],
        sample['pima'],
        sample['cdc'],
        sample['hosp']
    ], dim=1)

    # === SHAP explanation ===
    explainer = shap.Explainer(wrapped_model, inputs)
    shap_values = explainer(inputs)

    shap.plots.waterfall(shap_values[0])
