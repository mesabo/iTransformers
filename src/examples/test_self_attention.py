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

import torch
from src.foundation.self_attention import ScaledDotProductAttention

def test_scaled_dot_product_attention():
    try:
        d_k = 64  # Dimension of keys
        batch_size = 2
        seq_len = 5
        d_v = 64  # Dimension of values

        # Generate random Q, K, V matrices
        Q = torch.rand(batch_size, seq_len, d_k)
        K = torch.rand(batch_size, seq_len, d_k)
        V = torch.rand(batch_size, seq_len, d_v)

        # Instantiate attention layer
        attention = ScaledDotProductAttention(d_k)

        # Forward pass
        output, attn_weights = attention(Q, K, V)

        print("Output Shape:", output.shape)  # Expected: (batch_size, seq_len, d_v)
        print("Attention Weights Shape:", attn_weights.shape)  # Expected: (batch_size, seq_len, seq_len)
    except Exception as e:
        print("Error occurred during self-attention test:", e)
        raise
