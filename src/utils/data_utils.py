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
-->OBJECTIVE: Prepare the Iris dataset for the Transformer Encoder. This involves:
	1.	Loading and preprocessing the data.
	2.	Splitting the data into training and testing sets.
	3.	Creating a PyTorch DataLoader for efficient batching.
---------------------------------------------
'''

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, Dataset

class IrisDataset(Dataset):
    """
    Custom PyTorch Dataset for the Iris dataset.
    """
    def __init__(self, data, labels):
        """
        Args:
            data (torch.Tensor): Input features (num_samples, num_features).
            labels (torch.Tensor): Target labels (num_samples,).
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_iris_data(data_path, batch_size=32):
    """
    Loads and preprocesses the Iris dataset, returning DataLoaders for training and testing.

    Args:
        data_path (str): Path to the Iris dataset CSV file.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        tuple: Training DataLoader, Testing DataLoader.
    """
    # Load the dataset
    df = pd.read_csv(data_path)
    features = df.iloc[:, :-1].values  # Input features
    labels = df.iloc[:, -1].values     # Target labels

    # Encode string labels as integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Standardize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Convert to PyTorch tensors
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Create PyTorch Datasets
    train_dataset = IrisDataset(X_train, y_train)
    test_dataset = IrisDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader