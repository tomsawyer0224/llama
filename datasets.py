import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
import os, pathlib
from models import Tokenizer


class NextTokenPredictionDataset(Dataset):
    def __init__(self, text_path: str, tokenizer_path: str, context_len: int = 128):
        """
        args:
            text_path: path to text folder
            tokenizer: path to tokenizer model
            context_len: max sequence length
        rerturns: None
        """
        super().__init__()
        tokenizer = Tokenizer(tokenizer_path)
        # with open('text_path', 'r') as f:
        # text = f.read()
        text = pathlib.Path(text_path).read_text()
        self.all_tokens = tokenizer.encode(text)
        total_len = len(self.all_tokens)
        if total_len % context_len == 0:
            self.n_samples = total_len // context_len - 1
        else:
            self.n_samples = total_len // context_len
        self.total_len = self.n_samples * context_len
        self.context_len = context_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sequence = self.all_tokens[
            idx * self.context_len : (idx + 1) * self.context_len
        ]
        label = self.all_tokens[
            idx * self.context_len + 1 : (idx + 1) * self.context_len + 1
        ]
        return torch.tensor(sequence), torch.tensor(label)


class LitNextTokenPredictionDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data: str | Dataset,
        val_data: str | Dataset = None,
        test_data: str | Dataset = None,
        tokenizer_path: str = None,
        context_len: int = 128,
        batch_size: int = 16,
    ) -> None:
        """
        args:
            train_data: path/to/train_data or torch Dataset
            val_data: path/to/val_data or torch Dataset
            test_data: path/to/test_data or torch Dataset
            text_path: path to text folder
            tokenizer: path to tokenizer model
            context_len: max sequence length
        """
        assert isinstance(
            train_data, (str, Dataset)
        ), "train_data should be a str or a torch Dataset"
        assert tokenizer_path is not None, f"must provide tokenizer_path"
        if val_data:
            assert isinstance(
                val_data, (str, Dataset)
            ), "val_data should be a str or a torch Dataset"
        if test_data:
            assert isinstance(
                test_data, (str, Dataset)
            ), "test_data should be a str or a torch Dataset"
        super().__init__()

        if isinstance(train_data, str):
            train_data = NextTokenPredictionDataset(
                text_path=train_data,
                tokenizer_path=tokenizer_path,
                context_len=context_len,
            )

        if val_data:
            if isinstance(val_data, str):
                val_data = NextTokenPredictionDataset(
                    text_path=val_data,
                    tokenizer_path=tokenizer_path,
                    context_len=context_len,
                )

        if test_data:
            if isinstance(test_data, str):
                test_data = NextTokenPredictionDataset(
                    text_path=test_data,
                    tokenizer_path=tokenizer_path,
                    context_len=context_len,
                )

        if val_data and not test_data:
            train_data, test_data = torch.utils.data.random_split(
                dataset=train_data,
                lengths=[0.8, 0.2],
                generator=torch.Generator().manual_seed(42),
            )
        if not val_data and test_data:
            train_data, val_data = torch.utils.data.random_split(
                dataset=train_data,
                lengths=[0.8, 0.2],
                generator=torch.Generator().manual_seed(42),
            )
        if not val_data and not test_data:
            train_data, val_data, test_data = torch.utils.data.random_split(
                dataset=train_data,
                lengths=[0.8, 0.1, 0.1],
                generator=torch.Generator().manual_seed(42),
            )
        self.train_dataset = train_data
        self.val_dataset = val_data
        self.test_dataset = test_data
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )
