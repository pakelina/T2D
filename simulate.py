from model import MultimodalT2DPredictor
import torch
import pandas as pd

def simulate_prediction(model_path, ehr_tensor, life_tensor, syn_tensor, pima_tensor, cdc_tensor, hosp_tensor):
    model = MultimodalT2DPredictor(
        ehr_dim=ehr_tensor.shape[1],
        lifestyle_dim=life_tensor.shape[1],
        synthea_dim=syn_tensor.shape[1],
        pima_dim=pima_tensor.shape[1],
        cdc_dim=cdc_tensor.shape[1],
        hosp_dim=hosp_tensor.shape[1]
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    output = model(ehr_tensor, life_tensor, syn_tensor, pima_tensor, cdc_tensor, hosp_tensor)
    return torch.sigmoid(output).item()

# Dummy inputs for testing
ehr = torch.randn(1, 8)
life = torch.randn(1, 15)
syn = torch.randn(1, 6)
pima = torch.randn(1, 8)
cdc = torch.randn(1, 21)
hosp = torch.randn(1, 44)


risk = simulate_prediction("best_model.pt", ehr, life, syn, pima, cdc, hosp)
print("Predicted Diabetes Risk Score:", risk)
