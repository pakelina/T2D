# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from data_preprocessing import load_and_prepare_data
from model import MultimodalT2DPredictor


def train_model():
    # Load Data
    train_loader, test_loader = load_and_prepare_data()
    sample = next(iter(train_loader))

    # Get input dimensions
    ehr_dim = sample[0].shape[1]
    lifestyle_dim = sample[1].shape[1]
    clinical_dim = sample[2].shape[1]
    pima_dim = sample[3].shape[1]
    cdc_dim = sample[4].shape[1]
    hosp_dim = sample[5].shape[1]

    # Model & optimizer
    model = MultimodalT2DPredictor(ehr_dim, lifestyle_dim, clinical_dim, pima_dim, cdc_dim, hosp_dim)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.BCEWithLogitsLoss()

    # Training
    best_auc = 0.0
    patience = 7
    counter = 0

    for epoch in range(1, 31):
        model.train()
        total_loss = 0

        for ehr_x, life_x, clin_x, pima_x, cdc_x, hosp_x, y in train_loader:
            optimizer.zero_grad()
            logits = model(ehr_x, life_x, clin_x, pima_x, cdc_x, hosp_x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch:02d}] Train Loss: {total_loss:.4f}")

        # Validation
        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for ehr_x, life_x, clin_x, pima_x, cdc_x, hosp_x, y in test_loader:
                logits = model(ehr_x, life_x, clin_x, pima_x, cdc_x, hosp_x)
                probs = torch.sigmoid(logits).squeeze()
                val_probs += probs.tolist()
                val_true += y.squeeze().tolist()

        val_auc = roc_auc_score(val_true, val_probs)
        print(f"           Val AUC: {val_auc:.4f}")
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # Final Evaluation
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    y_true, y_probs, y_preds = [], [], []

    with torch.no_grad():
        for ehr_x, life_x, clin_x, pima_x, cdc_x, hosp_x, y in test_loader:
            logits = model(ehr_x, life_x, clin_x, pima_x, cdc_x, hosp_x)
            probs = torch.sigmoid(logits).squeeze()
            y_true += y.squeeze().tolist()
            y_probs += probs.tolist()
            y_preds += (probs >= 0.5).int().tolist()

    acc = accuracy_score(y_true, y_preds)
    auc = roc_auc_score(y_true, y_probs)
    print(f"\nFinal Test Accuracy: {acc:.4f}, AUC: {auc:.4f}")


if __name__ == "__main__":
    train_model()
