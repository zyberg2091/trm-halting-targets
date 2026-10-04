"""Build the addition dataset and loaders on import.

Source: notebooks/8digit/epochs_100/seed_0/softmean.ipynb, cells 11--13.
Cell 11 generates operands; 12 computes sums and carries; 13 defines
AdditionDataset, the split, and loaders. SEED and NUM_DIGITS come from config.
See CODE_PROVENANCE.md for the packaging changes.
"""
from src.config import SEED, NUM_DIGITS

import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split

random.seed(SEED)
torch.manual_seed(SEED)

def generate_pair(num_digits):
    a = random.randint(10**(num_digits-1), 10**num_digits - 1)
    b = random.randint(10**(num_digits-1), 10**num_digits - 1)
    return str(a), str(b)


operands_a, operands_b = [], []

num_digits = NUM_DIGITS

for sample_idx in range(50000):
    num_a, num_b = generate_pair(num_digits=num_digits)

    operands_a.append(num_a)
    operands_b.append(num_b)

dataset = []

for num_a, num_b in zip(operands_a, operands_b):

    input_sequence = list(num_a) + ['+'] + list(num_b)

    rev_a = list(reversed(num_a))
    rev_b = list(reversed(num_b))

    num_digits = len(num_a)

    result_digits = []
    carry_sequence = []

    carry = 0
    current_chain_length = 0
    max_carry_chain = 0

    for digit_a, digit_b in zip(rev_a, rev_b):

        digit_sum = int(digit_a) + int(digit_b) + carry

        output_digit = digit_sum % 10
        new_carry = digit_sum // 10

        result_digits.append(output_digit)
        carry_sequence.append(new_carry)

        # This label counts consecutive outgoing carries. It does not measure
        # the minimum number of dependent operations needed to solve the sum.
        if new_carry == 1:
            current_chain_length += 1
            max_carry_chain = max(max_carry_chain, current_chain_length)
        else:
            current_chain_length = 0

        carry = new_carry

    if carry > 0:
        result_digits.append(carry)

    final_output = list(reversed(result_digits))

    data = {
        'input': input_sequence,
        'output': final_output,
        'num_digits': num_digits,
        'carry_sequence': carry_sequence,
        'max_carry_chain': max_carry_chain
    }

    dataset.append(data)

print(dataset[0], len(dataset))

import torch
from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PAD_TOKEN = 11
PLUS_TOKEN = 10

inp_seq_len = 2 * num_digits + 1     # Operand A, '+', operand B
out_seq_len = num_digits + 1         # The sum can have one extra digit
n_chain     = num_digits + 1         # Buckets 0 through num_digits

class AdditionDataset(Dataset):
    def __init__(self, dataset_list, max_input_len=inp_seq_len, max_output_len=out_seq_len):
        self.data = dataset_list
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def pad(self, seq, max_len, pad_value=PAD_TOKEN):
        return seq + [pad_value] * (max_len - len(seq))

    def encode_input(self, input_seq):
        return [
            int(x) if x != '+' else PLUS_TOKEN
            for x in input_seq
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        input_ids = self.encode_input(sample['input'])
        labels    = sample['output']

        input_ids = self.pad(input_ids, self.max_input_len)
        labels    = self.pad(labels, self.max_output_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "carry": torch.tensor(sample['max_carry_chain'], dtype=torch.long)
        }



hidden_size = 256

batch_size  = 32

vocab_size = 12

dataset = AdditionDataset(dataset)

val_size = 5000
train_size = len(dataset) - val_size

torch.manual_seed(SEED)
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=batch_size)
