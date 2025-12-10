"""
Unit tests for training components: TextDataset and Trainer.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from train import TextDataset, Trainer
from model import GPTModel
from tokenizer import CharTokenizer


class TestTextDataset:
    """Tests for the TextDataset class."""

    def test_len_returns_positive(self, train_dataset):
        assert len(train_dataset) > 0

    def test_len_accounts_for_block_size(self, longer_text, tokenizer_large):
        block_size = 16
        dataset = TextDataset(longer_text, tokenizer_large, block_size=block_size)
        encoded_len = len(tokenizer_large.encode(longer_text))
        expected_len = max(1, encoded_len - block_size)
        assert len(dataset) == expected_len

    def test_getitem_returns_tuple(self, train_dataset):
        x, y = train_dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_getitem_shapes_match_block_size(self, longer_text, tokenizer_large):
        block_size = 16
        dataset = TextDataset(longer_text, tokenizer_large, block_size=block_size)
        x, y = dataset[0]
        assert x.shape == (block_size,)
        assert y.shape == (block_size,)

    def test_target_is_shifted_input(self, train_dataset):
        """y should be x shifted by 1 position."""
        x, y = train_dataset[0]
        # Get the underlying data to verify shift
        idx = 0
        block_size = train_dataset.block_size
        chunk = train_dataset.data[idx : idx + block_size + 1]
        expected_x = chunk[:-1]
        expected_y = chunk[1:]
        assert torch.equal(x, expected_x)
        assert torch.equal(y, expected_y)

    def test_dtype_is_long(self, train_dataset):
        x, y = train_dataset[0]
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_stores_tokenizer_reference(self, train_dataset, tokenizer_large):
        assert train_dataset.tokenizer is tokenizer_large


class TestTrainerInit:
    """Tests for Trainer initialization."""

    @pytest.fixture
    def trainer_components(self, longer_text, tokenizer_large, model_config):
        """Create components needed for Trainer."""
        dataset = TextDataset(longer_text, tokenizer_large, block_size=16)
        model = GPTModel(vocab_size=tokenizer_large.vocab_size, **model_config)
        return model, dataset

    def test_trainer_init_creates_optimizer(self, trainer_components):
        model, dataset = trainer_components
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            batch_size=4,
            num_epochs=1,
        )
        assert trainer.optimizer is not None

    def test_trainer_init_creates_scheduler(self, trainer_components):
        model, dataset = trainer_components
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            batch_size=4,
            num_epochs=1,
        )
        assert trainer.scheduler is not None

    def test_trainer_init_creates_data_loader(self, trainer_components):
        model, dataset = trainer_components
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            batch_size=4,
            num_epochs=1,
        )
        assert trainer.train_loader is not None

    def test_trainer_init_creates_checkpoint_dir(self, trainer_components):
        model, dataset = trainer_components
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            trainer = Trainer(
                model=model,
                train_dataset=dataset,
                batch_size=4,
                num_epochs=1,
                checkpoint_dir=str(checkpoint_dir),
            )
            assert checkpoint_dir.exists()


class TestTrainerTraining:
    """Tests for actual training functionality."""

    @pytest.fixture
    def trainer(self, longer_text, tokenizer_large, model_config):
        dataset = TextDataset(longer_text, tokenizer_large, block_size=16)
        model = GPTModel(vocab_size=tokenizer_large.vocab_size, **model_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Trainer(
                model=model,
                train_dataset=dataset,
                batch_size=4,
                num_epochs=1,
                checkpoint_dir=tmpdir,
            )

    def test_train_epoch_returns_loss(self, trainer):
        loss = trainer.train_epoch(epoch=0)
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_epoch_updates_parameters(self, trainer):
        # Get initial parameter values
        initial_params = [p.clone() for p in trainer.model.parameters()]

        trainer.train_epoch(epoch=0)

        # Check at least some parameters changed
        params_changed = False
        for initial, current in zip(initial_params, trainer.model.parameters()):
            if not torch.equal(initial, current):
                params_changed = True
                break
        assert params_changed

    @pytest.mark.slow
    def test_training_decreases_loss(self, longer_text, tokenizer_large, model_config):
        """Loss should generally decrease over multiple epochs."""
        dataset = TextDataset(longer_text, tokenizer_large, block_size=16)
        model = GPTModel(vocab_size=tokenizer_large.vocab_size, **model_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                train_dataset=dataset,
                batch_size=8,
                num_epochs=5,
                checkpoint_dir=tmpdir,
            )

            losses = []
            for epoch in range(5):
                loss = trainer.train_epoch(epoch)
                losses.append(loss)

            # Final loss should be less than initial
            assert losses[-1] < losses[0]


class TestTrainerCheckpoints:
    """Tests for checkpoint saving and loading."""

    @pytest.fixture
    def trainer_with_tmpdir(self, longer_text, tokenizer_large, model_config):
        dataset = TextDataset(longer_text, tokenizer_large, block_size=16)
        model = GPTModel(vocab_size=tokenizer_large.vocab_size, **model_config)
        tmpdir = tempfile.mkdtemp()
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            batch_size=4,
            num_epochs=1,
            checkpoint_dir=tmpdir,
        )
        return trainer, tmpdir

    def test_save_checkpoint_creates_file(self, trainer_with_tmpdir):
        trainer, tmpdir = trainer_with_tmpdir
        checkpoint_path = trainer.save_checkpoint(epoch=1, loss=1.0)
        assert Path(checkpoint_path).exists()

    def test_save_checkpoint_contains_required_keys(self, trainer_with_tmpdir):
        trainer, tmpdir = trainer_with_tmpdir
        checkpoint_path = trainer.save_checkpoint(epoch=1, loss=1.0)
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        required_keys = [
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "loss",
            "model_config",
        ]
        for key in required_keys:
            assert key in checkpoint

    def test_save_best_model_on_improvement(self, trainer_with_tmpdir):
        trainer, tmpdir = trainer_with_tmpdir
        trainer.best_val_loss = float("inf")

        trainer.save_checkpoint(epoch=1, loss=0.5)

        best_path = Path(tmpdir) / "best_model.pt"
        assert best_path.exists()

    def test_no_best_model_without_improvement(self, trainer_with_tmpdir):
        trainer, tmpdir = trainer_with_tmpdir
        trainer.best_val_loss = 0.1  # Already very low

        trainer.save_checkpoint(epoch=1, loss=0.5)

        # best_model.pt should not be created since loss didn't improve
        best_path = Path(tmpdir) / "best_model.pt"
        assert not best_path.exists()


class TestTrainerValidation:
    """Tests for validation functionality."""

    @pytest.fixture
    def trainer_with_val(self, longer_text, tokenizer_large, model_config):
        # Split text for train/val
        split_idx = len(longer_text) // 2
        train_text = longer_text[:split_idx]
        val_text = longer_text[split_idx:]

        train_dataset = TextDataset(train_text, tokenizer_large, block_size=16)
        val_dataset = TextDataset(val_text, tokenizer_large, block_size=16)
        model = GPTModel(vocab_size=tokenizer_large.vocab_size, **model_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            yield Trainer(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=4,
                num_epochs=1,
                checkpoint_dir=tmpdir,
            )

    def test_validate_returns_loss(self, trainer_with_val):
        val_loss = trainer_with_val.validate()
        assert isinstance(val_loss, float)
        assert val_loss > 0

    def test_validate_without_val_dataset_returns_none(
        self, longer_text, tokenizer_large, model_config
    ):
        dataset = TextDataset(longer_text, tokenizer_large, block_size=16)
        model = GPTModel(vocab_size=tokenizer_large.vocab_size, **model_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                train_dataset=dataset,
                val_dataset=None,
                batch_size=4,
                num_epochs=1,
                checkpoint_dir=tmpdir,
            )
            val_loss = trainer.validate()
            assert val_loss is None
