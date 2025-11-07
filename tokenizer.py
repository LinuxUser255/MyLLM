"""
Simple character-level tokenizer for the LLM.
Converts text to sequences of integers and back.
"""

import json
from typing import List, Dict


class CharTokenizer:
    """Character-level tokenizer that maps characters to unique IDs."""
    
    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab_size = 0
    
    def build_vocab(self, text: str) -> None:
        """Build vocabulary from text."""
        # Get unique characters
        chars = sorted(list(set(text)))
        
        # Create mappings
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Sample vocab: {list(self.char_to_id.items())[:10]}...")
    
    def encode(self, text: str) -> List[int]:
        """Convert text to list of token IDs."""
        return [self.char_to_id.get(ch, 0) for ch in text]
    
    def decode(self, ids: List[int]) -> str:
        """Convert list of token IDs back to text."""
        return ''.join([self.id_to_char.get(i, '') for i in ids])
    
    def save(self, path: str) -> None:
        """Save tokenizer vocabulary to file."""
        with open(path, 'w') as f:
            json.dump({
                'char_to_id': self.char_to_id,
                'id_to_char': {str(k): v for k, v in self.id_to_char.items()},
                'vocab_size': self.vocab_size
            }, f, indent=2)
    
    def load(self, path: str) -> None:
        """Load tokenizer vocabulary from file."""
        with open(path, 'r') as f:
            data = json.load(f)
            self.char_to_id = data['char_to_id']
            self.id_to_char = {int(k): v for k, v in data['id_to_char'].items()}
            self.vocab_size = data['vocab_size']