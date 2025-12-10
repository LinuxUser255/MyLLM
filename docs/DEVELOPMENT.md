# MyLLM Development Guide

Guide for developers contributing to or extending MyLLM.

---

## Project Structure

```
MyLLM/
├── model.py          # GPTModel, TransformerBlock, Attention, FeedForward
├── tokenizer.py      # CharTokenizer
├── train.py          # TextDataset, Trainer, CLI entry point
├── generate.py       # TextGenerator, interactive mode
├── requirements.txt  # Dependencies
├── checkpoints/      # Model checkpoints (gitignored)
├── data/             # Training data
├── docs/             # Documentation
└── tests/            # Unit tests
    ├── conftest.py   # Shared fixtures
    ├── test_tokenizer.py
    ├── test_model.py
    └── test_training.py
```

---

## Development Setup

### 1. Clone and Create Environment

```bash
git clone <repo-url>
cd MyLLM
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install pytest pytest-cov black isort flake8 mypy
```

### 3. Verify Setup

```bash
# Run tests
pytest tests/ -v

# Check formatting
black --check .
isort --check-only .
```

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific file
pytest tests/test_model.py -v

# Specific test class
pytest tests/test_model.py::TestGPTModel -v

# Specific test
pytest tests/test_model.py::TestGPTModel::test_output_shape -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing

# Skip slow tests
pytest tests/ -m "not slow"

# Run GPU tests (if available)
pytest tests/ -m gpu
```

### Writing Tests

**Test file naming:** `test_<module>.py`

**Test structure:**
```python
class TestFeatureName:
    """Tests for FeatureName."""

    @pytest.fixture
    def setup_data(self):
        """Fixture for test data."""
        return {"key": "value"}

    def test_behavior_description(self, setup_data):
        """Single behavior being tested."""
        result = function_under_test(setup_data)
        assert result == expected

    def test_edge_case(self):
        """Edge case or error handling."""
        with pytest.raises(ValueError):
            function_under_test(invalid_input)
```

**Test markers:**
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.gpu` - Tests requiring GPU

### Fixtures

Common fixtures in `conftest.py`:
- `sample_text` - Short text for tokenizer tests
- `longer_text` - Longer text for training tests
- `tokenizer` - Initialized CharTokenizer
- `small_model` - Small GPTModel for fast tests
- `model_config` - Dict of small model hyperparameters
- `device` - Best available torch device

---

## Code Style

### Formatting

```bash
# Format code
black .
isort .

# Check without changing
black --check .
isort --check-only .
```

### Linting

```bash
# Run flake8
flake8 . --max-line-length=99 --ignore=E203,W503

# Type checking
mypy . --ignore-missing-imports
```

### Style Guidelines

1. **Line length:** 99 characters max
2. **Imports:** Sorted with isort (black profile)
3. **Docstrings:** Google style for public functions/classes
4. **Type hints:** Required for public API

**Example:**
```python
def encode(self, text: str) -> List[int]:
    """Convert text to list of token IDs.

    Args:
        text: Input string to encode.

    Returns:
        List of integer token IDs.

    Raises:
        ValueError: If text contains null characters.
    """
    return [self.char_to_id.get(ch, 0) for ch in text]
```

---

## Architecture Overview

### Model Components

```
GPTModel
├── token_embedding (nn.Embedding)
├── position_embedding (nn.Embedding)
├── dropout (nn.Dropout)
├── blocks (nn.ModuleList)
│   └── TransformerBlock × n_layers
│       ├── ln1 (LayerNorm)
│       ├── attention (MultiHeadAttention)
│       │   ├── W_q, W_k, W_v, W_o (Linear)
│       │   └── dropout
│       ├── ln2 (LayerNorm)
│       ├── feed_forward (FeedForward)
│       │   ├── linear1, linear2 (Linear)
│       │   └── gelu, dropout
│       └── dropout
├── ln_final (LayerNorm)
└── lm_head (Linear, weight-tied)
```

### Data Flow

```
Input tokens [B, T]
    ↓
Token embedding + Position embedding [B, T, D]
    ↓
Dropout
    ↓
┌─────────────────────────────────────┐
│ TransformerBlock × N                │
│   ├── LayerNorm → Attention → Add   │
│   └── LayerNorm → FFN → Add         │
└─────────────────────────────────────┘
    ↓
Final LayerNorm [B, T, D]
    ↓
Linear projection [B, T, V]
    ↓
Output logits
```

---

## Extending MyLLM

### Adding a New Tokenizer

```python
# tokenizer.py
class BPETokenizer:
    """Byte-Pair Encoding tokenizer."""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}

    def train(self, text: str) -> None:
        """Train BPE on corpus."""
        # Implementation...

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        # Implementation...

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        # Implementation...
```

### Adding Attention Variants

```python
# model.py
class FlashAttention(nn.Module):
    """Memory-efficient attention using Flash Attention."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # Use torch.nn.functional.scaled_dot_product_attention
        # with memory_efficient_attention backend

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # Implementation with flash attention
        pass
```

### Adding Learning Rate Schedulers

```python
# train.py
def get_scheduler(optimizer, config):
    """Factory for learning rate schedulers."""
    if config.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(...)
    elif config.scheduler == "linear_warmup":
        return LinearWarmupScheduler(...)
    elif config.scheduler == "one_cycle":
        return optim.lr_scheduler.OneCycleLR(...)
```

---

## Debugging

### Common Issues

**1. CUDA/MPS out of memory:**
```python
# Reduce batch size
trainer = Trainer(..., batch_size=16)

# Or use gradient accumulation
for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**2. NaN loss:**
```python
# Check for gradient explosion
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm()}")

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)
```

**3. Slow training on MPS:**
```python
# Force synchronization for accurate timing
torch.mps.synchronize()
```

### Profiling

```python
# PyTorch profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    model(input_ids)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## Git Workflow

### Branch Naming

- `feature/add-bpe-tokenizer`
- `fix/nan-loss-gradient-clip`
- `refactor/attention-module`
- `docs/update-readme`

### Commit Messages

```
<type>: <short description>

<longer description if needed>

Types:
- feat: New feature
- fix: Bug fix
- refactor: Code restructuring
- docs: Documentation
- test: Adding/updating tests
- perf: Performance improvement
```

### Pre-commit Checks

Before committing:
```bash
# Format
black .
isort .

# Lint
flake8 .

# Test
pytest tests/ -x

# Type check (optional)
mypy . --ignore-missing-imports
```

---

## Performance Benchmarks

Run benchmarks to track performance:

```python
# benchmark.py
import time
import torch
from model import GPTModel

def benchmark_forward(model, batch_size, seq_len, n_iterations=100):
    device = next(model.parameters()).device
    x = torch.randint(0, 100, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(10):
        model(x)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iterations):
        model(x)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed / n_iterations
```
