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

'''
Integrate the encoder and decoder to form a complete Transformer.
'''

import torch
import torch.nn as nn

from src.foundation.transformer_decoder import TransformerDecoder
from src.foundation.transformer_encoder_stack import TransformerEncoder


class TransformerModel(nn.Module):
    """
    Implements the full Transformer model for sequence-to-sequence tasks.
    """

    def __init__(self, input_dim, output_dim, d_model, num_heads, ff_hidden_dim, num_encoder_layers, num_decoder_layers,
                 dropout=0.1):
        """
        Args:
            input_dim (int): Input dimension (e.g., input embedding size).
            output_dim (int): Output dimension (e.g., output embedding size).
            d_model (int): Dimension of the model (input/output feature size).
            num_heads (int): Number of attention heads.
            ff_hidden_dim (int): Hidden dimension of the feedforward network.
            num_encoder_layers (int): Number of encoder layers.
            num_decoder_layers (int): Number of decoder layers.
            dropout (float): Dropout probability.
        """
        super(TransformerModel, self).__init__()

        # Encoder
        self.encoder = TransformerEncoder(input_dim, d_model, num_heads, ff_hidden_dim, num_encoder_layers, int(5000),
                                          dropout)
        # Target Embedding
        self.tgt_embedding = nn.Linear(output_dim, d_model)

        # Decoder
        self.decoder = TransformerDecoder(d_model, num_heads, ff_hidden_dim, num_decoder_layers, dropout)

        # Output projection
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Args:
            src (torch.Tensor): Source sequence (batch_size, src_seq_len, input_dim).
            tgt (torch.Tensor): Target sequence (batch_size, tgt_seq_len, output_dim).
            src_mask (torch.Tensor, optional): Mask for source sequence (src_seq_len, src_seq_len).
            tgt_mask (torch.Tensor, optional): Mask for target sequence (tgt_seq_len, tgt_seq_len).
            memory_mask (torch.Tensor, optional): Mask for encoder output (src_seq_len, tgt_seq_len).

        Returns:
            torch.Tensor: Output of the Transformer (batch_size, tgt_seq_len, output_dim).
        """
        print(f"src shape: {src.shape}")  # Should be (batch_size, seq_len, input_dim)
        print(f"tgt shape before embedding: {tgt.shape}")  # Should be (batch_size, seq_len, output_dim)

        # Ensure `tgt` has a seq_len dimension
        if tgt.dim() == 2:
            tgt = tgt.unsqueeze(1)  # Add seq_len dimension

        memory = self.encoder(src, src_mask)  # Encoder output
        # Embed the target tensor
        tgt = self.tgt_embedding(tgt)  # Shape: (batch_size, tgt_seq_len=1, d_model)
        print(f"tgt shape after embedding: {tgt.shape}")  # Should be (batch_size, seq_len, d_model)

        output = self.decoder(tgt, memory, tgt_mask, memory_mask)  # Decoder output
        return self.fc_out(output)
