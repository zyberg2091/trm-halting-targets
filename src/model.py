"""Source: notebooks/4digit/epochs_100/seed_0/softmean.ipynb, cell 6."""
import torch
from torch import nn

class TinyRModel(nn.Module):
  def __init__(self, hidden_size, vocab_size, inp_seq_len, out_seq_len):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                              nn.ReLU(),
                              nn.Linear(hidden_size, hidden_size))

    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.input_proj = nn.Linear(inp_seq_len * hidden_size, hidden_size)


    self.out = nn.Linear(hidden_size, out_seq_len * vocab_size)
    self.halt = nn.Linear(hidden_size, 1)



  def latent(self, x, y, z, n=6):
      # x enters the z updates only.
      for i in range(n):
          z = self.net(x + y + z)

      y = self.net(y + z)           # Reuse net for y

      return y, z



  def forward(self, x, y, z, T, n=6):

    x = self.embedding(x)          # (B, L, H)
    x = x.view(x.size(0), -1)              # Flatten the sequence, keeping the batch axis
    x = self.input_proj(x)                 # (B, H)

    # Backpropagate through the last latent call only.
    with torch.no_grad():
      for j in range(T-1):
        y, z= self.latent(x,y,z,n)

    y, z = self.latent(x, y, z, n)

    f_out = self.out(y)

    halt= self.halt(y) # Raw logit; the threshold is applied in train.py

    return f_out, halt, y, z
