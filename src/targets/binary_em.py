"""Exact-match target.

Source: notebooks/4digit/epochs_100/seed_0/binary_em.ipynb, cell 16.
"""
from src.data import PAD_TOKEN

import torch.nn.functional as F

def Q_LOSS(q_halt, f_out, y_true, pad_token=PAD_TOKEN):
    preds = f_out.argmax(dim=-1)                                               # (B, L)
    non_pad = (y_true != pad_token)                                            # (B, L)
    # Only non-padding positions must match.
    target = ((preds == y_true) | ~non_pad).all(dim=-1, keepdim=True).float()  # (B, 1)
    return F.binary_cross_entropy_with_logits(q_halt, target, reduction='none')
