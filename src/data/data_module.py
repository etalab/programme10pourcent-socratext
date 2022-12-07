"""
Tickets Dataset Module.
"""
from typing import List

import random
import pytorch_lightning as pl
from albumentations import Compose
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from mappings import label2id
from data.formatter import LabelStudioJsonFormatter


class TicketsDataset(Dataset):
    """
    Tickets dataset.
    """

    def __init__(self, formatted_data: List, processor=None, max_length=512):
        """
        Args:
            formatted_data (List): List of formatted Label Studio data
                (paths and word-level annotations (words, labels, boxes).
            processor (LayoutLMv2Processor): Processor to prepare the
                text + image.
        """
        self.formatted_data = formatted_data
        self.processor = processor

    def __len__(self):
        return len(self.formatted_data)

    def __getitem__(self, idx):
        # First, take an image
        image_data = self.formatted_data[idx]

        # Get path and image
        path = image_data["path"]
        image = Image.open(path)
        image = ImageOps.exif_transpose(image)

        # Get word-level annotations
        words = image_data["words"]
        boxes = image_data["boxes"]
        word_labels = image_data["labels"]

        assert len(words) == len(boxes) == len(word_labels)

        word_labels = [label2id[label] for label in word_labels]
        # Use processor to prepare everything
        encoded_inputs = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Remove batch dimension
        for k, v in encoded_inputs.items():
            encoded_inputs[k] = v.squeeze()

        assert encoded_inputs.input_ids.shape == torch.Size([512])
        assert encoded_inputs.attention_mask.shape == torch.Size([512])
        assert encoded_inputs.token_type_ids.shape == torch.Size([512])
        assert encoded_inputs.bbox.shape == torch.Size([512, 4])
        assert encoded_inputs.image.shape == torch.Size([3, 224, 224])
        assert encoded_inputs.labels.shape == torch.Size([512])

        return encoded_inputs


class TicketsDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning Data Module for the Tickets dataset.
    """

    def __init__(
        self,
        data: List,
        test_data: List,
        processor=None,
        transforms_preprocessing: Compose = None,
        transforms_augmentation: Compose = None,
        batch_size: int = 2,
        num_workers: int = 4,
    ):
        """
        Data Module initialization.

        Args:
            data (List): Train/validation data in the form of a raw
                Label Studio annotations json.
            test_data (List): Test data in the form of a raw Label
                Studio annotations json.
            transforms_preprocessing (Optional[Compose]): Compose object
                from albumentations applied
                on validation an test dataset.
            transforms_augmentation (Optional[Compose]): Compose object
                from albumentations applied on training dataset.
            batch_size (int): Define batch size.
            num_workers (int): Define number of workers to process data.
        """
        super().__init__()
        self.formatter = LabelStudioJsonFormatter()

        self.data = self.formatter.format_data(data)
        self.data = self.formatter.filter_data(self.data)

        self.test_data = self.formatter.format_data(test_data)
        self.test_data = self.formatter.filter_data(self.test_data)

        self.processor = processor

        self.transforms_preprocessing = transforms_preprocessing
        self.transforms_augmentation = transforms_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.setup()

    def setup(self, stage: str = None) -> None:
        """
        Start training, validation and test datasets.

        Args:
            stage (Optional[str]): Used to separate setup logic
                for trainer.fit and trainer.test.
        """
        n_samples = len(self.data)
        random.shuffle(self.data)
        train_slice = slice(0, int(n_samples * 0.8))
        val_slice = slice(int(n_samples * 0.8), n_samples)

        self.train_dataset = TicketsDataset(
            self.data[train_slice], processor=self.processor
        )
        self.val_dataset = TicketsDataset(
            self.data[val_slice], processor=self.processor
        )
        self.test_dataset = TicketsDataset(
            self.test_data,
            processor=self.processor
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        """
        Create Dataloader.

        Returns: DataLoader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        """
        Create Dataloader.

        Returns: DataLoader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        """
        Create Dataloader.

        Returns: DataLoader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
