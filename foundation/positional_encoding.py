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
-->OBJECTIVE: Add Positional Encoding to the input embeddings to provide the Transformer with information about 
the sequence order. Since Transformers don’t have recurrence or convolution, positional encoding is crucial for 
handling sequential data like the Iris dataset.
---------------------------------------------
1. Positional Encoding

Key Concept

Positional encoding adds a unique representation for each position in the sequence using sine and cosine functions of varying frequencies.

Formula


PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)


PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)


Where:
	•	pos: Position in the sequence.
	•	i: Dimension index.
	•	d_{\text{model}}: Total dimensionality of the model.
'''

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model (int): Dimension of the model (input embeddings).
            max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()

        # Create a matrix of [max_len, d_model] with positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Tensor with positional encodings added (batch_size, seq_len, d_model).
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x