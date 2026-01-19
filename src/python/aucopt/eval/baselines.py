import torch
import torch.nn as nn
import torch.optim as optimiz
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import os

# Optional import depending on availability
try:
    from libauc.losses import AUCMLoss
    from libauc.optimizers import PESG
    LIBAUC_AVAILABLE = True
except ImportError:
    LIBAUC_AVAILABLE = False

def evaluate_pytorch_bce(X_train, X_test, y_train, y_test, device="cpu"):
    class LinearModel(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.linear = nn.Linear(d, 1, bias=False)
        def forward(self, x):
            return self.linear(x)

    try:
        model = LinearModel(X_train.shape[1]).to(device)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        optimizer = optimiz.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(100):
            optimizer.zero_grad()
            loss = loss_fn(model(X_train_tensor), y_train_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = model(X_test_tensor).squeeze().cpu().numpy()
            auc = roc_auc_score(y_test, scores)
            return auc, scores
    except Exception as e:
        return f"Error: {e}"

def evaluate_libauc(X_train, X_test, y_train, y_test, device="cpu"):
    if not LIBAUC_AVAILABLE:
        return "LibAUC not available"

    class LinearModel(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.linear = nn.Linear(d, 1, bias=False)
        def forward(self, x):
            return self.linear(x)

    try:
        model = LinearModel(X_train.shape[1]).to(device)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        loss_fn = AUCMLoss(imratio=y_train.mean())
        optimizer = PESG(params=model.parameters(),
                 loss_fn=loss_fn,
                 a=loss_fn.a, b=loss_fn.b, alpha=loss_fn.alpha,
                 lr=0.01, weight_decay=1e-5, imratio=y_train.mean())


        model.train()
        for _ in range(100):
            y_pred = model(X_train_tensor)
            loss = loss_fn(y_pred, y_train_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = model(X_test_tensor).squeeze().cpu().numpy()
            auc = roc_auc_score(y_test, scores)
            return auc, scores
    except Exception as e:
        return f"Error: {e}"