#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/12/2025
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

'''
-->OBJECTIVE: Combine the Iris dataset DataLoader with the Transformer Encoder for training and evaluation. 
This step connects the data pipeline to the Transformer Encoder and prepares for training.
---------------------------------------------
1. Training Workflow

Key Steps
	1.	Load Data:
	•	Use the train_loader and test_loader from the load_iris_data function.
	2.	Initialize the Transformer Encoder:
	•	Set up the model parameters, optimizer, and loss function.
	3.	Training Loop:
	•	Train the Transformer Encoder using the training data.
	4.	Evaluation:
	•	Test the model on the test data to assess performance.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.data_utils import load_iris_data
from src.foundation.transformer_encoder_stack import TransformerEncoder

def train_transformer_encoder(data_path, batch_size=32, epochs=10, learning_rate=1e-3, device="cpu"):
    """
    Trains the Transformer Encoder on the Iris dataset.

    Args:
        data_path (str): Path to the Iris dataset CSV file.
        batch_size (int): Batch size for training and testing.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Computation device ("cpu", "cuda", "mps").
    """
    # Load Data
    train_loader, test_loader = load_iris_data(data_path, batch_size=batch_size)

    # Model Parameters
    input_dim = 4       # Iris dataset has 4 features
    d_model = 64        # Dimension of the model
    num_heads = 8       # Number of attention heads
    ff_hidden_dim = 128 # Hidden dimension of the feedforward network
    num_layers = 3      # Number of encoder blocks

    # Initialize Model, Loss, and Optimizer
    model = TransformerEncoder(input_dim, d_model, num_heads, ff_hidden_dim, num_layers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, labels) in enumerate(train_loader):
            # Reshape input data to include sequence length dimension
            data = data.unsqueeze(1)  # Shape: (batch_size, seq_len=1, input_dim)
            data, labels = data.to(device), labels.to(device)

            # Forward Pass
            outputs = model(data)[:, 0, :]  # Use [CLS]-like token for classification
            loss = criterion(outputs, labels)

            # Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.unsqueeze(1)  # Reshape for sequence dimension
            data, labels = data.to(device), labels.to(device)

            # Forward Pass
            outputs = model(data)[:, 0, :]
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.4f}%")