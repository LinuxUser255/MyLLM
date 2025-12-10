# MyLLM Usage Guide

A GPT-style language model optimized for Apple Silicon (M3 Max).

## Quick Start

### Prerequisites

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training a Model

**Basic training:**
```bash
python train.py --data data/sample.txt --epochs 10
```

**Full training with custom parameters:**
```bash
python train.py \
    --data data/combined_literature.txt \
    --epochs 50 \
    --batch_size 64 \
    --lr 3e-4 \
    --block_size 128 \
    --d_model 384 \
    --n_heads 6 \
    --n_layers 6 \
    --val_split 0.1
```

### Generating Text

**Single prompt:**
```bash
python generate.py --prompt "To be or not to be"
```

**Interactive mode:**
```bash
python generate.py --interactive
```

**With custom settings:**
```bash
python generate.py \
    --prompt "Once upon a time" \
    --max_tokens 500 \
    --temperature 0.8 \
    --top_k 50
```

---

## Command-Line Reference

### train.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | `data/input.txt` | Path to training text file |
| `--epochs` | int | 50 | Number of training epochs |
| `--batch_size` | int | 64 | Training batch size |
| `--lr` | float | 3e-4 | Learning rate |
| `--block_size` | int | 128 | Context window size |
| `--d_model` | int | 384 | Model embedding dimension |
| `--n_heads` | int | 6 | Number of attention heads |
| `--n_layers` | int | 6 | Number of transformer layers |
| `--val_split` | float | 0.1 | Fraction of data for validation |

### generate.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | `checkpoints/best_model.pt` | Path to model checkpoint |
| `--tokenizer` | str | `checkpoints/tokenizer.json` | Path to tokenizer |
| `--prompt` | str | None | Text prompt for generation |
| `--max_tokens` | int | 200 | Max tokens to generate |
| `--temperature` | float | 0.8 | Sampling temperature (0.1-2.0) |
| `--top_k` | int | 50 | Top-k sampling parameter |
| `--seed` | int | None | Random seed for reproducibility |
| `--interactive` | flag | - | Run in interactive mode |

---

## Python API

### CharTokenizer

```python
from tokenizer import CharTokenizer

# Build vocabulary from text
tokenizer = CharTokenizer()
tokenizer.build_vocab("Hello world!")

# Encode text to token IDs
ids = tokenizer.encode("Hello")  # [7, 4, 9, 9, 12]

# Decode token IDs back to text
text = tokenizer.decode(ids)  # "Hello"

# Save/load tokenizer
tokenizer.save("tokenizer.json")
tokenizer.load("tokenizer.json")
```

### GPTModel

```python
from model import GPTModel, get_device

# Create model
model = GPTModel(
    vocab_size=100,
    d_model=384,
    n_heads=6,
    n_layers=6,
    max_seq_len=128,
    d_ff=1536,
    dropout=0.1
)

# Move to device
device = get_device()  # Auto-selects MPS/CUDA/CPU
model = model.to(device)

# Forward pass
import torch
input_ids = torch.randint(0, 100, (1, 32)).to(device)
logits = model(input_ids)  # Shape: [1, 32, 100]

# Generate text
generated = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50
)
```

### Training

```python
from train import TextDataset, Trainer
from model import GPTModel
from tokenizer import CharTokenizer

# Prepare data
with open("data/sample.txt") as f:
    text = f.read()

tokenizer = CharTokenizer()
tokenizer.build_vocab(text)

# Create dataset
dataset = TextDataset(text, tokenizer, block_size=128)

# Create model
model = GPTModel(
    vocab_size=tokenizer.vocab_size,
    d_model=384,
    n_heads=6,
    n_layers=6,
    max_seq_len=128
)

# Train
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    learning_rate=3e-4,
    batch_size=64,
    num_epochs=10
)
trainer.train()
```

---

## Generation Parameters Explained

### Temperature
Controls randomness in sampling:
- **Low (0.1-0.5):** More focused, deterministic output
- **Medium (0.7-1.0):** Balanced creativity and coherence
- **High (1.2-2.0):** More creative, potentially nonsensical

### Top-k Sampling
Limits sampling to the k most likely tokens:
- **Low k (5-20):** Very conservative, repetitive
- **Medium k (40-100):** Balanced variety
- **High k (200+):** More variety, potentially incoherent

### Recommended Settings

| Use Case | Temperature | Top-k |
|----------|-------------|-------|
| Factual/consistent | 0.3 | 20 |
| Creative writing | 0.8 | 50 |
| Experimental | 1.2 | 100 |

---

## Training Data

### Available Datasets

| File | Size | Description |
|------|------|-------------|
| `sample.txt` | 3 KB | Quick testing |
| `alice_in_wonderland.txt` | 148 KB | Lewis Carroll |
| `pride_and_prejudice.txt` | 735 KB | Jane Austen |
| `shakespeare_complete.txt` | 5.3 MB | Complete works |
| `sherlock_holmes.txt` | 593 KB | Arthur Conan Doyle |
| `combined_literature.txt` | 6.8 MB | All combined |

### Adding Custom Data

1. Place text file in `data/` directory
2. Ensure UTF-8 encoding
3. Run training with `--data data/your_file.txt`

---

## Output Files

Training produces:
- `checkpoints/tokenizer.json` - Vocabulary mappings
- `checkpoints/checkpoint_epoch_N.pt` - Per-epoch checkpoints
- `checkpoints/best_model.pt` - Best validation loss checkpoint

### Checkpoint Contents

```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'loss': float,
    'model_config': {
        'vocab_size': int,
        'd_model': int,
        'n_heads': int,
        'n_layers': int,
        'max_seq_len': int,
        'd_ff': int
    }
}
```

---

## Performance Tips

### Apple Silicon (M3 Max)

- Uses MPS (Metal Performance Shaders) automatically
- Optimal batch size: 32-128
- Pin memory enabled for faster data transfer

### Memory Usage

Approximate VRAM requirements:
- Default config (384d, 6L): ~500 MB
- Medium (512d, 8L): ~1 GB
- Large (768d, 12L): ~2.5 GB

### Reducing Memory

```bash
# Smaller model
python train.py --d_model 256 --n_layers 4 --batch_size 32

# Shorter context
python train.py --block_size 64
```
