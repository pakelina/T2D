# test_shap.py

from model import MultimodalT2DPredictor
import numpy as np
from shap_explainer import explain_prediction
import torch

ehr   = torch.randn(1, 8)
life  = torch.randn(1, 15)   # changed from 4 to 15
syn   = torch.randn(1, 6)    # changed from 5 to 6
pima  = torch.randn(1, 8)
cdc   = torch.randn(1, 21)   # changed from 15 to 21
hosp  = torch.randn(1, 44)   # changed from 10 to 44

explain_prediction("best_model.pt", {
    "ehr": ehr,
    "lifestyle": life,
    "synthea": syn,
    "pima": pima,
    "cdc": cdc,
    "hosp": hosp
})
