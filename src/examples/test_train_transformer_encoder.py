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
from pathlib import Path
from src.training.train_transformer_encoder import train_transformer_encoder

def test_train_transformer_encoder():
    data_path = Path(__file__).resolve().parent.parent.parent / "data/Iris/iris.csv"
    batch_size = 32
    epochs = 10
    learning_rate = 1e-3
    device = "mps"  # Change to "cuda" or "cpu" as needed

    # Train and evaluate the Transformer Encoder
    train_transformer_encoder(str(data_path), batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, device=device)

# if __name__ == "__main__":
#     test_train_transformer_encoder()