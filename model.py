# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalT2DPredictor(nn.Module):
    def __init__(self, ehr_dim, lifestyle_dim, clinical_dim, pima_dim, cdc_dim, hosp_dim):
        super(MultimodalT2DPredictor, self).__init__()

        # Define input branches
        self.ehr_branch = nn.Sequential(
            nn.Linear(ehr_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        self.lifestyle_branch = nn.Sequential(
            nn.Linear(lifestyle_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
        )


        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        self.pima_branch = nn.Sequential(
            nn.Linear(pima_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        self.cdc_branch = nn.Sequential(
            nn.Linear(cdc_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
        )
        self.hosp_branch = nn.Sequential(
            nn.Linear(hosp_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(32 + 16 + 32 + 32 + 16 + 32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Binary classification
        )

    def forward(self, ehr_x, life_x, clin_x, pima_x, cdc_x, hosp_x):
        ehr_out = self.ehr_branch(ehr_x)
        life_out = self.lifestyle_branch(life_x)
        clin_out = self.clinical_branch(clin_x)
        pima_out = self.pima_branch(pima_x)
        cdc_out = self.cdc_branch(cdc_x)
        hosp_out = self.hosp_branch(hosp_x)

        # Concatenate all modalities
        fused = torch.cat([ehr_out, life_out, clin_out, pima_out, cdc_out, hosp_out], dim=1)
        output = self.fusion(fused)
        return output
