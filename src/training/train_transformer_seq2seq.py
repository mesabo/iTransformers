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
from src.utils.data_utils import load_iris_seq2seq_data
from src.models.transformer_model import TransformerModel

def train_transformer_seq2seq(data_path, batch_size=32, epochs=2, learning_rate=1e-3, device="cpu"):
    """
    Trains the Transformer Model on the Iris dataset for Seq2Seq tasks.

    Args:
        data_path (str): Path to the Iris dataset CSV file.
        batch_size (int): Batch size for training and testing.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Computation device ("cpu", "cuda", "mps").
    """
    # Load Data
    train_loader, test_loader = load_iris_seq2seq_data(data_path, batch_size=batch_size)

    # Model Parameters
    input_dim = 4        # Iris dataset has 4 features
    output_dim = 3       # One-hot encoded labels (3 classes)
    d_model = 64         # Dimension of the model
    num_heads = 8        # Number of attention heads
    ff_hidden_dim = 128  # Hidden dimension of the feedforward network
    num_encoder_layers = 3
    num_decoder_layers = 3

    # Initialize Model, Loss, and Optimizer
    model = TransformerModel(input_dim, output_dim, d_model, num_heads, ff_hidden_dim, num_encoder_layers, num_decoder_layers).to(device)
    print(f"Model: {model}")

    criterion = nn.MSELoss()  # Using Mean Squared Error for one-hot outputs
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, labels in train_loader:
            # Add seq_len dimension
            data = data.unsqueeze(1).to(device)  # Shape: (batch_size, seq_len=1, input_dim)
            labels = labels.unsqueeze(1).to(device)  # Shape: (batch_size, seq_len=1, output_dim)

            # Forward Pass
            outputs = model(data, labels)

            # Compute loss
            loss = criterion(outputs.squeeze(1), labels.squeeze(1))

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
            data = data.unsqueeze(1).to(device)  # Add seq_len dimension
            labels = labels.unsqueeze(1).to(device)  # Add seq_len dimension

            outputs = model(data, labels)
            predicted = torch.argmax(outputs, dim=-1)  # Shape: (batch_size, seq_len)
            targets = torch.argmax(labels, dim=-1)  # Shape: (batch_size, seq_len)

            # Flatten batch and sequence dimensions for comparison
            correct += (predicted.view(-1) == targets.view(-1)).sum().item()
            total += targets.numel()  # Total number of elements in targets

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
