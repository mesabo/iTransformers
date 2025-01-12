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
Implement the Scaled Dot-Product Attention mechanism, which forms the foundation of Transformers.
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Implements Scaled Dot-Product Attention.
    """

    def __init__(self, d_k):
        """
        Args:
            d_k (int): Dimension of the Key vectors.
        """
        super(ScaledDotProductAttention, self).__init__()
        self.scale = 1.0 / (d_k ** 0.5)  # Scaling factor

    def forward(self, Q, K, V, mask=None):
        """
        Computes the scaled dot-product attention.

        Args:
            Q (torch.Tensor): Query matrix (batch_size, seq_len, d_k).
            K (torch.Tensor): Key matrix (batch_size, seq_len, d_k).
            V (torch.Tensor): Value matrix (batch_size, seq_len, d_v).
            mask (torch.Tensor, optional): Mask for attention (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Attention output (batch_size, seq_len, d_v).
            torch.Tensor: Attention weights (batch_size, seq_len, seq_len).
        """
        # Compute scores (QK^T)
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        scores = scores * self.scale  # Scale scores

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # Compute weighted sum of values
        output = torch.matmul(attn_weights, V)  # (batch_size, seq_len, d_v)

        return output, attn_weights