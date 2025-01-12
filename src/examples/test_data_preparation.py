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

from pathlib import Path
from src.utils.data_utils import load_iris_data

def test_data_preparation():
    # Dynamically resolve the absolute path to the dataset
    data_path = Path(__file__).resolve().parent.parent.parent / "data/Iris/iris.csv"
    batch_size = 16

    # Load DataLoaders
    train_loader, test_loader = load_iris_data(str(data_path), batch_size=batch_size)

    # Test the DataLoader
    print("Training DataLoader:")
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"Data Shape: {data.shape}")  # Expected: (batch_size, num_features)
        print(f"Labels Shape: {labels.shape}")  # Expected: (batch_size,)
        break  # Test one batch

    print("\nTesting DataLoader:")
    for batch_idx, (data, labels) in enumerate(test_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"Data Shape: {data.shape}")
        print(f"Labels Shape: {labels.shape}")
        break

if __name__ == "__main__":
    test_data_preparation()