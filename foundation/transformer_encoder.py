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
-->OBJECTIVE: Build the Transformer Encoder Block, which combines:
	1.	Multi-Head Attention.
	2.	Feedforward Neural Network (FFN).
	3.	Layer normalization and residual connections.
---------------------------------------------
1. Transformer Encoder Block Design

Components
	1.	Multi-Head Attention:
	•	Extracts relationships between tokens in the sequence.
	2.	Feedforward Network (FFN):
	•	Applies point-wise transformations to each token.
	3.	Layer Normalization:
	•	Stabilizes training by normalizing the outputs.
	4.	Residual Connections:
	•	Helps propagate information across layers.
'''

import torch
import torch.nn as nn
from foundation.multi_head_attention import MultiHeadAttention


class TransformerEncoderBlock(nn.Module):
    """
    Implements a single Transformer Encoder block.
    """

    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        """
        Args:
            d_model (int): Dimension of the model (input/output feature size).
            num_heads (int): Number of attention heads.
            ff_hidden_dim (int): Hidden dimension of the feedforward network.
            dropout (float): Dropout probability.
        """
        super(TransformerEncoderBlock, self).__init__()

        # Multi-Head Attention
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)

        # Feedforward Network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, d_model),
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Attention mask (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Output of the encoder block (batch_size, seq_len, d_model).
        """
        # Multi-Head Attention + Residual Connection
        attn_output, _ = self.multi_head_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feedforward Network + Residual Connection
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x