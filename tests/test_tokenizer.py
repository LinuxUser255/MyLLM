"""
Unit tests for the CharTokenizer class.
"""

import json
import tempfile
from pathlib import Path

import pytest

from tokenizer import CharTokenizer


class TestCharTokenizerInit:
    """Tests for CharTokenizer initialization."""

    def test_init_creates_empty_mappings(self):
        tok = CharTokenizer()
        assert tok.char_to_id == {}
        assert tok.id_to_char == {}
        assert tok.vocab_size == 0


class TestBuildVocab:
    """Tests for vocabulary building."""

    def test_build_vocab_sets_correct_size(self, tokenizer, sample_text):
        expected_size = len(set(sample_text))
        assert tokenizer.vocab_size == expected_size

    def test_build_vocab_creates_bidirectional_mappings(self, tokenizer):
        for char, idx in tokenizer.char_to_id.items():
            assert tokenizer.id_to_char[idx] == char

    def test_build_vocab_sorts_characters(self, tokenizer):
        chars = list(tokenizer.char_to_id.keys())
        assert chars == sorted(chars)

    def test_build_vocab_handles_empty_string(self):
        tok = CharTokenizer()
        tok.build_vocab("")
        assert tok.vocab_size == 0

    def test_build_vocab_handles_single_char(self):
        tok = CharTokenizer()
        tok.build_vocab("a")
        assert tok.vocab_size == 1
        assert tok.char_to_id["a"] == 0


class TestEncode:
    """Tests for encoding text to token IDs."""

    def test_encode_returns_list(self, tokenizer):
        result = tokenizer.encode("test")
        assert isinstance(result, list)

    def test_encode_returns_integers(self, tokenizer):
        result = tokenizer.encode("Hello")
        assert all(isinstance(x, int) for x in result)

    def test_encode_length_matches_input(self, tokenizer):
        text = "Hello"
        result = tokenizer.encode(text)
        assert len(result) == len(text)

    def test_encode_empty_string(self, tokenizer):
        result = tokenizer.encode("")
        assert result == []

    def test_encode_unknown_char_returns_zero(self, tokenizer):
        # Character not in vocabulary
        result = tokenizer.encode("@")
        assert result == [0]


class TestDecode:
    """Tests for decoding token IDs to text."""

    def test_decode_returns_string(self, tokenizer):
        ids = [0, 1, 2]
        result = tokenizer.decode(ids)
        assert isinstance(result, str)

    def test_decode_empty_list(self, tokenizer):
        result = tokenizer.decode([])
        assert result == ""

    def test_decode_unknown_id_returns_empty(self, tokenizer):
        result = tokenizer.decode([9999])
        assert result == ""


class TestEncodeDecodeRoundtrip:
    """Tests for encode/decode consistency."""

    def test_roundtrip_preserves_text(self, tokenizer, sample_text):
        encoded = tokenizer.encode(sample_text)
        decoded = tokenizer.decode(encoded)
        assert decoded == sample_text

    def test_roundtrip_single_char(self, tokenizer):
        for char in "Hello":
            encoded = tokenizer.encode(char)
            decoded = tokenizer.decode(encoded)
            assert decoded == char


class TestSaveLoad:
    """Tests for saving and loading tokenizer."""

    def test_save_creates_file(self, tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"
            tokenizer.save(str(path))
            assert path.exists()

    def test_save_creates_valid_json(self, tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"
            tokenizer.save(str(path))
            with open(path) as f:
                data = json.load(f)
            assert "char_to_id" in data
            assert "id_to_char" in data
            assert "vocab_size" in data

    def test_load_restores_state(self, tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"
            tokenizer.save(str(path))

            new_tok = CharTokenizer()
            new_tok.load(str(path))

            assert new_tok.vocab_size == tokenizer.vocab_size
            assert new_tok.char_to_id == tokenizer.char_to_id

    def test_loaded_tokenizer_encodes_same(self, tokenizer, sample_text):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"
            tokenizer.save(str(path))

            new_tok = CharTokenizer()
            new_tok.load(str(path))

            assert new_tok.encode(sample_text) == tokenizer.encode(sample_text)
