# whatif.py
import torch
from model import MultimodalT2DPredictor

def run_what_if(model_path, baseline_input, changed_input):
    model = MultimodalT2DPredictor(
        ehr_dim=baseline_input["ehr"].shape[1],
        lifestyle_dim=baseline_input["lifestyle"].shape[1],
        synthea_dim=baseline_input["clinical"].shape[1],
        pima_dim=baseline_input["pima"].shape[1],
        cdc_dim=baseline_input["cdc"].shape[1],
        hosp_dim=baseline_input["hosp"].shape[1]
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    def predict(inputs):
        with torch.no_grad():
            output = model(
                torch.tensor(inputs["ehr"], dtype=torch.float32),
                torch.tensor(inputs["lifestyle"], dtype=torch.float32),
                torch.tensor(inputs["synthea"], dtype=torch.float32),
                torch.tensor(inputs["pima"], dtype=torch.float32),
                torch.tensor(inputs["cdc"], dtype=torch.float32),
                torch.tensor(inputs["hosp"], dtype=torch.float32),
            )
            return torch.sigmoid(output).item()

    return predict(baseline_input), predict(changed_input)
