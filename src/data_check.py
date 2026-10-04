"""Notebook subset-count check; prints on import.

Source: notebooks/4digit/epochs_100/seed_0/softmean.ipynb, cell 23.
"""
from src.data import dataset

# Retained notebook filters. Both <= 2-digit subsets are empty here.
easy_dataset = [d for d in dataset.data if d['num_digits'] <= 2 and d['max_carry_chain'] == 0]
print(f"Easy examples: {len(easy_dataset)}")

medium_dataset = [d for d in dataset.data if d['num_digits'] <= 2]
print(f"Medium examples: {len(medium_dataset)}")

# All examples for a 4-digit run; none for an 8-digit run.
hard_dataset = [d for d in dataset.data if d['num_digits'] == 4]
print(f"Hard examples: {len(hard_dataset)}")
