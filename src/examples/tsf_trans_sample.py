#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/14/2025
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# Generate Synthetic Time Series Data
def generate_synthetic_data(seq_length=20, num_samples=1000):
    x = np.sin(np.linspace(0, 100, num_samples))  # Example sine wave
    sequences, targets = [], []
    for i in range(len(x) - seq_length):
        sequences.append(x[i:i + seq_length])
        targets.append(x[i + seq_length])
    return np.array(sequences), np.array(targets)


# Define a Lightweight Transformer Model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, ff_hidden_dim, num_layers, seq_length, output_dim):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, d_model))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_hidden_dim
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.embedding(src) + self.positional_encoding
        src = src.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)
        encoded = self.encoder(src)
        output = self.fc_out(encoded[-1])  # Use the last time step's output
        return output


# Hyperparameters
seq_length = 20
d_model = 64
num_heads = 4
ff_hidden_dim = 128
num_layers = 2
batch_size = 32
epochs = 10
learning_rate = 0.001

# Data Preparation
x, y = generate_synthetic_data(seq_length=seq_length)
x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(x, y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model Initialization
model = TimeSeriesTransformer(input_dim=1, d_model=d_model, num_heads=num_heads,
                              ff_hidden_dim=ff_hidden_dim, num_layers=num_layers,
                              seq_length=seq_length, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in data_loader:
        batch_x = batch_x.unsqueeze(-1)  # Add feature dimension
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(data_loader):.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    # test_seq = torch.tensor(x[:1], dtype=torch.float32).unsqueeze(-1)
    test_seq = x[:1].clone().detach().unsqueeze(-1)
    prediction = model(test_seq)
    print(f"Predicted: {prediction.squeeze().item()}, Actual: {y[0].item()}")