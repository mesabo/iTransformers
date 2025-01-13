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
-->OBJECTIVE: Build the Transformer Encoder by stacking multiple encoder blocks. This forms the backbone of the Transformer architecture.
---------------------------------------------
1. Design of Full Transformer Encoder

Components
	1.	Input Embedding:
	•	Converts input tokens/features into high-dimensional vectors.
	2.	Positional Encoding:
	•	Adds positional information to embeddings.
	3.	Stack of Encoder Blocks:
	•	Sequentially applies multiple Transformer Encoder Blocks.
'''

import torch
import torch.nn as nn
from src.foundation.positional_encoding import PositionalEncoding
from src.foundation.transformer_encoder import TransformerEncoderBlock


class TransformerEncoder(nn.Module):
    """
    Implements the full Transformer Encoder.
    """

    def __init__(self, input_dim, d_model, num_heads, ff_hidden_dim, num_layers, max_len=5000, dropout=0.1):
        """
        Args:
            input_dim (int): Dimension of input features (e.g., input embedding size).
            d_model (int): Dimension of the model (input/output feature size).
            num_heads (int): Number of attention heads.
            ff_hidden_dim (int): Hidden dimension of the feedforward network.
            num_layers (int): Number of encoder blocks.
            max_len (int): Maximum sequence length.
            dropout (float): Dropout probability.
        """
        super(TransformerEncoder, self).__init__()

        # Input Embedding Layer
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Stack of Transformer Encoder Blocks
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, input_dim).
            mask (torch.Tensor, optional): Attention mask (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Output of the encoder stack (batch_size, seq_len, d_model).
        """
        # Debug original shape
        print(f"Original Input Shape: {x.shape}")

        # Apply input embedding and positional encoding
        x = self.input_embedding(x)  # Shape: (batch_size, seq_len, d_model)
        print(f"Shape after Embedding: {x.shape}")

        x = self.positional_encoding(x)  # Shape: (batch_size, seq_len, d_model)
        print(f"Shape after Positional Encoding: {x.shape}")

        x = self.dropout(x)

        # Pass through each encoder block
        for layer in self.layers:
            x = layer(x, mask)

        return x