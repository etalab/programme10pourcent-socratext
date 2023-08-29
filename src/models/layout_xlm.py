"""
LayoutXLM Module.
"""
import pytorch_lightning as pl
from transformers import LayoutLMv2ForTokenClassification, AdamW
from mappings import label2id
from torch import optim


class LayoutXLMModule(pl.LightningModule):
    """
    Pytorch Lightning Module for LayoutXLM.
    """

    def __init__(
        self,
        initial_lr: float = 0.001
    ):
        """
        Initialize LayoutXLMModule.

        Args:
            initial_lr (float): Initial learning rate.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = LayoutLMv2ForTokenClassification.from_pretrained(
            "microsoft/layoutxlm-base",
            num_labels=len(label2id)
        )

        self.initial_lr = initial_lr

    def forward(self, batch):
        """
        Perform forward-pass.

        Args:
            batch (tensor): Batch of images to perform forward-pass.

        Returns: Model prediction.
        """
        # TODO : better implementation of this function
        # Get the inputs
        input_ids = batch['input_ids']
        bbox = batch['bbox']
        image = batch['image']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        labels = batch['labels']

        outputs = self.model(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        """
        Get training step.

        Args:
            batch (List): Data for training.
            batch_idx (int): Batch index.

        Returns: Tensor.
        """
        # Get the inputs
        input_ids = batch['input_ids']
        bbox = batch['bbox']
        image = batch['image']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        labels = batch['labels']

        outputs = self.model(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        loss = outputs.loss

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Get validation step.

        Args:
            batch (List): Data for validating.
            batch_idx (int): Batch index.

        Returns: Tensor.
        """
        # Get the inputs
        input_ids = batch['input_ids']
        bbox = batch['bbox']
        image = batch['image']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        labels = batch['labels']

        outputs = self.model(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        loss = outputs.loss

        self.log('validation_loss', loss)
        # Log other performance metrics as well as outputs ?
        return loss

    def test_step(self, batch, batch_idx):
        """
        Get test step.

        Args:
            batch (List): Data for training.
            batch_idx (int): Batch index.

        Returns: Tensor.
        """
        # Get the inputs
        input_ids = batch['input_ids']
        bbox = batch['bbox']
        image = batch['image']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        labels = batch['labels']

        outputs = self.model(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        loss = outputs.loss

        self.log('test_loss', loss)
        # Log other performance metrics as well as outputs ?
        return loss

    def configure_optimizers(self):
        """
        Configure optimizer for Pytorch lightning.

        Returns: Optimizer and scheduler for Pytorch lightning.
        """
        optimizer = AdamW(self.model.parameters(), lr=self.initial_lr)

        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            'monitor': 'validation_loss',
            'interval': 'epoch'
        }

        return [optimizer], [scheduler]
