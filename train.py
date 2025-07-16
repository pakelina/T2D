# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from data_preprocessing import load_and_prepare_data
from model import MultimodalT2DPredictor


def train_model():
    train_loader, test_loader = load_and_prepare_data()
    sample = next(iter(train_loader))
    ehr_dim = sample[0].shape[1]
    lifestyle_dim = sample[1].shape[1]
    synthea_dim = sample[2].shape[1]
    pima_dim = sample[3].shape[1]
    cdc_dim = sample[4].shape[1]
    hosp_dim = sample[5].shape[1]

    model = MultimodalT2DPredictor(ehr_dim, lifestyle_dim, synthea_dim, pima_dim, cdc_dim, hosp_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(1, 51):
        model.train()
        total_loss = 0
        for ehr_x, life_x, syn_x, pima_x, cdc_x, hosp_x, y in train_loader:
            optimizer.zero_grad()
            logits = model(ehr_x, life_x, syn_x, pima_x, cdc_x, hosp_x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}/50, Loss: {total_loss:.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    y_true, y_pred_prob, y_pred_bin = [], [], []

    with torch.no_grad():
        for ehr_x, life_x, syn_x, pima_x, cdc_x, hosp_x, y in test_loader:
            logits = model(ehr_x, life_x, syn_x, pima_x, cdc_x, hosp_x)
            probs = torch.sigmoid(logits).squeeze()
            y_true += y.squeeze().tolist()
            y_pred_prob += probs.tolist()
            y_pred_bin += (probs >= 0.5).int().tolist()

    acc = accuracy_score(y_true, y_pred_bin)
    auc = roc_auc_score(y_true, y_pred_prob)
    print(f"\nTest Accuracy: {acc:.4f}, AUC: {auc:.4f}")


if __name__ == "__main__":
    train_model()