# model.py
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class MultimodalT2DPredictor(nn.Module):
    def __init__(self, ehr_dim, lifestyle_dim, synthea_dim, pima_dim, cdc_dim, hosp_dim):
        super().__init__()
        self.ehr_encoder = Encoder(ehr_dim)
        self.lifestyle_encoder = Encoder(lifestyle_dim)
        self.synthea_encoder = Encoder(synthea_dim)
        self.pima_encoder = Encoder(pima_dim)
        self.cdc_encoder = Encoder(cdc_dim)
        self.hosp_encoder = Encoder(hosp_dim)

        self.classifier = nn.Sequential(
            nn.Linear(16 * 6, 32), nn.ReLU(),
            nn.Linear(32, 1)  # No sigmoid, use BCEWithLogitsLoss
        )

    def forward(self, ehr_x, life_x, syn_x, pima_x, cdc_x, hosp_x):
        ehr_out = self.ehr_encoder(ehr_x)
        life_out = self.lifestyle_encoder(life_x)
        syn_out = self.synthea_encoder(syn_x)
        pima_out = self.pima_encoder(pima_x)
        cdc_out = self.cdc_encoder(cdc_x)
        hosp_out = self.hosp_encoder(hosp_x)

        combined = torch.cat([ehr_out, life_out, syn_out, pima_out, cdc_out, hosp_out], dim=1)
        return self.classifier(combined)
