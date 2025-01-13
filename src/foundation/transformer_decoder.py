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
-->OBJECTIVE: Leverage the Transformer Encoder for sequence-to-sequence tasks. We’ll:
	1.	Add a Decoder to create a full Transformer model.
	2.	Train and test the model on a simple sequence-to-sequence task (e.g., addition of two numbers, toy translation).
---------------------------------------------
Steps to Expand

4.1 Build the Transformer Decoder

We’ll implement the Transformer Decoder with:
	1.	Masked Multi-Head Attention: Prevents tokens from attending to future tokens in the output sequence.
	2.	Cross-Attention: Allows the decoder to focus on encoder outputs.
'''

import torch
import torch.nn as nn
from src.foundation.multi_head_attention import MultiHeadAttention

class TransformerDecoderBlock(nn.Module):
    """
    Implements a single Transformer Decoder block.
    """
    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        """
        Args:
            d_model (int): Dimension of the model (input/output feature size).
            num_heads (int): Number of attention heads.
            ff_hidden_dim (int): Hidden dimension of the feedforward network.
            dropout (float): Dropout probability.
        """
        super(TransformerDecoderBlock, self).__init__()

        # Masked Multi-Head Attention
        self.masked_attn = MultiHeadAttention(d_model, num_heads)

        # Cross-Attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads)

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
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt (torch.Tensor): Target sequence (batch_size, tgt_seq_len, d_model).
            memory (torch.Tensor): Encoder output (batch_size, src_seq_len, d_model).
            tgt_mask (torch.Tensor, optional): Mask for target sequence (tgt_seq_len, tgt_seq_len).
            memory_mask (torch.Tensor, optional): Mask for encoder output (src_seq_len, tgt_seq_len).

        Returns:
            torch.Tensor: Output of the decoder block (batch_size, tgt_seq_len, d_model).
        """
        # Masked Multi-Head Attention + Residual Connection
        tgt2, _ = self.masked_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))

        # Cross-Attention + Residual Connection
        tgt2, _ = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))

        # Feedforward Network + Residual Connection
        tgt2 = self.feedforward(tgt)
        tgt = self.norm3(tgt + self.dropout(tgt2))

        return tgt


class TransformerDecoder(nn.Module):
    """
    Implements the full Transformer Decoder.
    """
    def __init__(self, d_model, num_heads, ff_hidden_dim, num_layers, dropout=0.1):
        """
        Args:
            d_model (int): Dimension of the model (input/output feature size).
            num_heads (int): Number of attention heads.
            ff_hidden_dim (int): Hidden dimension of the feedforward network.
            num_layers (int): Number of decoder blocks.
            dropout (float): Dropout probability.
        """
        super(TransformerDecoder, self).__init__()

        # Stack of Transformer Decoder Blocks
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final Layer Normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt (torch.Tensor): Target sequence (batch_size, tgt_seq_len, d_model).
            memory (torch.Tensor): Encoder output (batch_size, src_seq_len, d_model).
            tgt_mask (torch.Tensor, optional): Mask for target sequence (tgt_seq_len, tgt_seq_len).
            memory_mask (torch.Tensor, optional): Mask for encoder output (src_seq_len, tgt_seq_len).

        Returns:
            torch.Tensor: Output of the decoder (batch_size, tgt_seq_len, d_model).
        """
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)

        output = self.norm(tgt)

        return output