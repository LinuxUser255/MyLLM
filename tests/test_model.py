"""
Unit tests for the GPTModel and its components.
"""

import pytest
import torch

from model import (
    GPTModel,
    TransformerBlock,
    MultiHeadAttention,
    FeedForward,
    get_device,
)


class TestGetDevice:
    """Tests for device selection."""

    def test_get_device_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_get_device_returns_valid_type(self):
        device = get_device()
        assert device.type in ("cpu", "cuda", "mps")


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention module."""

    @pytest.fixture
    def attention(self):
        return MultiHeadAttention(d_model=32, n_heads=4, dropout=0.0)

    def test_output_shape(self, attention):
        batch_size, seq_len, d_model = 2, 8, 32
        x = torch.randn(batch_size, seq_len, d_model)
        output = attention(x)
        assert output.shape == (batch_size, seq_len, d_model)

    def test_accepts_mask(self, attention):
        x = torch.randn(2, 8, 32)
        mask = torch.tril(torch.ones(8, 8)).unsqueeze(0).unsqueeze(0)
        output = attention(x, mask=mask)
        assert output.shape == x.shape

    def test_d_model_must_be_divisible_by_n_heads(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=32, n_heads=5)

    def test_scale_factor_is_correct(self, attention):
        expected_scale = (32 / 4) ** 0.5  # sqrt(d_k)
        assert abs(attention.scale - expected_scale) < 1e-6


class TestFeedForward:
    """Tests for FeedForward module."""

    @pytest.fixture
    def ff(self):
        return FeedForward(d_model=32, d_ff=128, dropout=0.0)

    def test_output_shape(self, ff):
        batch_size, seq_len, d_model = 2, 8, 32
        x = torch.randn(batch_size, seq_len, d_model)
        output = ff(x)
        assert output.shape == (batch_size, seq_len, d_model)

    def test_expansion_ratio(self, ff):
        assert ff.linear1.out_features == 128
        assert ff.linear2.in_features == 128


class TestTransformerBlock:
    """Tests for TransformerBlock module."""

    @pytest.fixture
    def block(self):
        return TransformerBlock(d_model=32, n_heads=4, d_ff=128, dropout=0.0)

    def test_output_shape(self, block):
        batch_size, seq_len, d_model = 2, 8, 32
        x = torch.randn(batch_size, seq_len, d_model)
        output = block(x)
        assert output.shape == (batch_size, seq_len, d_model)

    def test_has_layer_norms(self, block):
        assert hasattr(block, "ln1")
        assert hasattr(block, "ln2")

    def test_residual_connection_effect(self, block):
        """Output should differ from input due to residual + transformation."""
        x = torch.randn(2, 8, 32)
        output = block(x)
        assert not torch.allclose(output, x)


class TestGPTModel:
    """Tests for the main GPTModel class."""

    def test_output_shape(self, small_model, tokenizer):
        batch_size, seq_len = 2, 8
        vocab_size = tokenizer.vocab_size
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits = small_model(x)

        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_count_parameters_returns_positive(self, small_model):
        param_count = small_model.count_parameters()
        assert param_count > 0

    def test_weight_tying(self, small_model):
        """lm_head weights should be same object as token_embedding weights."""
        assert small_model.lm_head.weight is small_model.token_embedding.weight

    def test_causal_mask_shape(self, small_model):
        seq_len = 8
        mask = small_model.create_causal_mask(seq_len, torch.device("cpu"))
        assert mask.shape == (1, 1, seq_len, seq_len)

    def test_causal_mask_is_lower_triangular(self, small_model):
        seq_len = 4
        mask = small_model.create_causal_mask(seq_len, torch.device("cpu"))
        expected = torch.tril(torch.ones(seq_len, seq_len))
        assert torch.equal(mask.squeeze(), expected)

    def test_forward_with_single_token(self, small_model, tokenizer):
        x = torch.randint(0, tokenizer.vocab_size, (1, 1))
        logits = small_model(x)
        assert logits.shape == (1, 1, tokenizer.vocab_size)


class TestGPTModelGenerate:
    """Tests for the generate method."""

    def test_generate_increases_sequence_length(self, small_model, tokenizer):
        initial_len = 4
        new_tokens = 10
        x = torch.randint(0, tokenizer.vocab_size, (1, initial_len))

        generated = small_model.generate(x, max_new_tokens=new_tokens)

        assert generated.shape[1] == initial_len + new_tokens

    def test_generate_preserves_batch_size(self, small_model, tokenizer):
        batch_size = 3
        x = torch.randint(0, tokenizer.vocab_size, (batch_size, 4))

        generated = small_model.generate(x, max_new_tokens=5)

        assert generated.shape[0] == batch_size

    def test_generate_with_temperature(self, small_model, tokenizer):
        x = torch.randint(0, tokenizer.vocab_size, (1, 4))
        # Should not raise
        generated = small_model.generate(x, max_new_tokens=5, temperature=0.5)
        assert generated.shape[1] == 9

    def test_generate_with_top_k(self, small_model, tokenizer):
        x = torch.randint(0, tokenizer.vocab_size, (1, 4))
        generated = small_model.generate(x, max_new_tokens=5, top_k=5)
        assert generated.shape[1] == 9

    def test_generate_crops_context_when_exceeding_max_seq_len(
        self, small_model, tokenizer
    ):
        """Generate should handle inputs longer than max_seq_len."""
        max_seq_len = small_model.max_seq_len
        x = torch.randint(0, tokenizer.vocab_size, (1, max_seq_len + 10))

        # Should not raise, should crop internally
        generated = small_model.generate(x, max_new_tokens=5)
        assert generated.shape[1] == max_seq_len + 10 + 5


class TestGPTModelDevice:
    """Tests for device handling."""

    @pytest.mark.gpu
    def test_model_to_device(self, small_model, device):
        model = small_model.to(device)
        # Check a parameter is on the right device
        assert next(model.parameters()).device.type == device.type

    @pytest.mark.gpu
    def test_forward_on_device(self, small_model, tokenizer, device):
        model = small_model.to(device)
        x = torch.randint(0, tokenizer.vocab_size, (1, 4)).to(device)

        logits = model(x)

        assert logits.device.type == device.type
