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
-->Objective: Build and test a Multi-Head Attention Layer, which enables the Transformer to focus on different parts of 
the input sequence simultaneously.

1. Multi-Head Attention Layer

Key Concept

Multi-Head Attention splits the input queries, keys, and values into multiple heads, applies scaled dot-product attention to each, and concatenates the results.

Formula


\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O

Where:
	•	Each head computes:

\text{head}_i = \text{Attention}(Q W^Q_i, K W^K_i, V W^V_i)

	•	W^Q, W^K, W^V, W^O are learned weight matrices.
'''

import torch
import torch.nn as nn
from foundation.self_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention.
    """

    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model (int): Dimension of the model (input/output feature size).
            num_heads (int): Number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model)

        # Attention layer
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q (torch.Tensor): Query matrix (batch_size, seq_len, d_model).
            K (torch.Tensor): Key matrix (batch_size, seq_len, d_model).
            V (torch.Tensor): Value matrix (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Mask for attention (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Output of multi-head attention (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = Q.size()

        # Project Q, K, V
        Q = self.w_q(Q)  # (batch_size, seq_len, d_model)
        K = self.w_k(K)  # (batch_size, seq_len, d_model)
        V = self.w_v(V)  # (batch_size, seq_len, d_model)

        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Apply scaled dot-product attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Apply final linear projection
        output = self.w_o(attn_output)

        return output, attn_weights