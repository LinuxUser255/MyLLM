"""
Shared pytest fixtures for MyLLM tests.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import GPTModel, get_device
from tokenizer import CharTokenizer
from train import TextDataset, Trainer


# --- Sample Data Fixtures ---

@pytest.fixture
def sample_text():
    """Minimal text for testing tokenizer and dataset."""
    return "Hello world! This is a test. The quick brown fox jumps over the lazy dog."


@pytest.fixture
def longer_text():
    """Longer text for training-related tests."""
    return """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them.
    """ * 10  # Repeat to get enough data


# --- Tokenizer Fixtures ---

@pytest.fixture
def tokenizer(sample_text):
    """CharTokenizer with vocabulary built from sample_text."""
    tok = CharTokenizer()
    tok.build_vocab(sample_text)
    return tok


@pytest.fixture
def tokenizer_large(longer_text):
    """CharTokenizer with vocabulary built from longer_text."""
    tok = CharTokenizer()
    tok.build_vocab(longer_text)
    return tok


# --- Model Fixtures ---

@pytest.fixture
def model_config():
    """Small model configuration for fast tests."""
    return {
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 2,
        "max_seq_len": 16,
        "d_ff": 64,
        "dropout": 0.0,  # Disable dropout for deterministic tests
    }


@pytest.fixture
def small_model(tokenizer, model_config):
    """Small GPTModel instance for testing."""
    return GPTModel(
        vocab_size=tokenizer.vocab_size,
        **model_config
    )


@pytest.fixture
def device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --- Dataset Fixtures ---

@pytest.fixture
def train_dataset(longer_text, tokenizer_large):
    """TextDataset for training tests."""
    return TextDataset(longer_text, tokenizer_large, block_size=16)


# --- Pytest Configuration ---

def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
