import sys

if not "." in sys.path:
    sys.path.append(".")
import unittest
import torch
from torch.utils.data import Dataset, DataLoader
import os

from datasets import LitNextTokenPredictionDataModule, NextTokenPredictionDataset


class Test_datasets(unittest.TestCase):
    def setUp(self):
        train_data = "./data/tinyshakespeare.txt"
        tokenizer_path = "./data/tiny_shakespeare.model"
        self.NTPDataModule = LitNextTokenPredictionDataModule(
            train_data=train_data,
            val_data=None,
            test_data=None,
            tokenizer_path=tokenizer_path,
            context_len=8,
            batch_size=4,
        )
        self.NTPDataset = NextTokenPredictionDataset(
            text_path=train_data,
            tokenizer_path=tokenizer_path,
            context_len=8,
        )

    def test_NextTokenPredictionDataset(self):
        print("---test_NextTokenPredictionDataset---")
        indices = torch.randint(0, len(self.NTPDataset), (4,))
        for idx in indices:
            sequence, label = self.NTPDataset[idx]
            print(f"sequence: {sequence}")
            print(f"label: {label}")
            print("++++++++")
        print()

    def test_train_dataloader(self):
        print("---test_train_dataloader---")
        train_loader = self.NTPDataModule.train_dataloader()
        batch = next(iter(train_loader))
        sequence, label = batch
        # print(f'sequence: {sequence.shape}\n{sequence}')
        # print(f'label: {label.shape}\n{label}')
        print(f"sequence: \n{sequence}")
        print(f"label: \n{label}")
        print()

    def test_val_dataloader(self):
        print("---test_val_dataloader---")
        val_loader = self.NTPDataModule.val_dataloader()
        batch = next(iter(val_loader))
        sequence, label = batch
        # print(f'sequence: {sequence.shape}\n{sequence}')
        # print(f'label: {label.shape}\n{label}')
        print(f"sequence: \n{sequence}")
        print(f"label: \n{label}")
        print()

    def test_test_dataloader(self):
        print("---test_test_dataloader---")
        test_loader = self.NTPDataModule.test_dataloader()
        batch = next(iter(test_loader))
        sequence, label = batch
        # print(f'sequence: {sequence.shape}\n{sequence}')
        # print(f'label: {label.shape}\n{label}')
        print(f"sequence: \n{sequence}")
        print(f"label: \n{label}")
        print()


if __name__ == "__main__":
    unittest.main()
