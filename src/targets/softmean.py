"""Digit-accuracy target.

Source: notebooks/4digit/epochs_100/seed_0/softmean.ipynb, cell 17.
"""
from src.data import PAD_TOKEN

import torch.nn.functional as F

def Q_LOSS(q_halt, f_out, y_true, pad_token=PAD_TOKEN):
    preds = f_out.argmax(dim=-1)                     # (B, L)
    correct = (preds == y_true).float()              # (B, L)
    non_pad = (y_true != pad_token).float()          # (B, L)
    target = (correct * non_pad).sum(dim=-1, keepdim=True) / (non_pad.sum(dim=-1, keepdim=True) + 1e-8)
    return F.binary_cross_entropy_with_logits(q_halt, target, reduction='none')
