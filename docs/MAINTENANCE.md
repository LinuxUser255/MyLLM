# MyLLM Maintenance Guide

Guide for maintaining, troubleshooting, and upgrading MyLLM.

---

## Checkpoint Management

### Checkpoint Structure

Each checkpoint contains:
```python
{
    'epoch': 10,
    'model_state_dict': {...},      # Model weights
    'optimizer_state_dict': {...},  # Optimizer state (momentum, etc.)
    'scheduler_state_dict': {...},  # LR scheduler state
    'loss': 1.234,                  # Validation loss at save time
    'model_config': {               # For model reconstruction
        'vocab_size': 95,
        'd_model': 384,
        'n_heads': 6,
        'n_layers': 6,
        'max_seq_len': 128,
        'd_ff': 1536
    }
}
```

### Loading Checkpoints

```python
import torch
from model import GPTModel

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')

# Reconstruct model
config = checkpoint['model_config']
model = GPTModel(
    vocab_size=config['vocab_size'],
    d_model=config['d_model'],
    n_heads=config['n_heads'],
    n_layers=config['n_layers'],
    max_seq_len=config['max_seq_len'],
    d_ff=config['d_ff']
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
```

### Resuming Training

```python
from train import Trainer

# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint_epoch_10.pt')

# Create trainer
trainer = Trainer(model=model, train_dataset=dataset, ...)

# Restore optimizer and scheduler state
trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
trainer.best_val_loss = checkpoint['loss']

# Continue training from epoch 11
for epoch in range(checkpoint['epoch'], total_epochs):
    trainer.train_epoch(epoch)
```

### Checkpoint Cleanup

Remove old checkpoints to save disk space:

```bash
# Keep only best model and last 3 epochs
cd checkpoints
ls -t checkpoint_epoch_*.pt | tail -n +4 | xargs rm -f
```

Automated cleanup script:
```bash
#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${1:-checkpoints}"
KEEP_LAST="${2:-3}"

# Always keep best_model.pt
# Remove epoch checkpoints older than KEEP_LAST
find "$CHECKPOINT_DIR" -name 'checkpoint_epoch_*.pt' -type f | \
    sort -t_ -k3 -n | \
    head -n -"$KEEP_LAST" | \
    xargs -r rm -v
```

---

## Dependency Management

### Current Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0 | Deep learning framework |
| numpy | >=1.20 | Numerical operations |
| tqdm | >=4.60 | Progress bars |

### Updating Dependencies

```bash
# Check for outdated packages
pip list --outdated

# Update specific package
pip install --upgrade torch

# Update all
pip install --upgrade -r requirements.txt

# Freeze current versions
pip freeze > requirements-lock.txt
```

### PyTorch Version Compatibility

| PyTorch | Python | MPS Support | Notes |
|---------|--------|-------------|-------|
| 2.0+ | 3.8-3.11 | Full | Recommended |
| 1.13 | 3.7-3.10 | Limited | Older MPS |
| 1.12 | 3.7-3.10 | None | CPU/CUDA only |

### Known Compatibility Issues

**torch 2.1 + Python 3.12:**
- Some operations may fail on MPS
- Workaround: Use Python 3.11

**NumPy 2.0:**
- Breaking changes in array API
- Pin to `numpy<2.0` if issues occur

---

## Troubleshooting

### Training Issues

#### Loss is NaN or Inf

**Causes:**
1. Learning rate too high
2. Gradient explosion
3. Division by zero in attention

**Solutions:**
```python
# Lower learning rate
trainer = Trainer(..., learning_rate=1e-4)

# Increase gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

# Check for NaN in inputs
assert not torch.isnan(input_ids.float()).any()
```

#### Loss Not Decreasing

**Causes:**
1. Learning rate too low
2. Model too small for data
3. Data quality issues

**Solutions:**
```bash
# Increase learning rate
python train.py --lr 1e-3

# Increase model capacity
python train.py --d_model 512 --n_layers 8

# Check data
head -100 data/input.txt  # Verify content
```

#### Out of Memory

**Solutions:**
```bash
# Reduce batch size
python train.py --batch_size 16

# Reduce model size
python train.py --d_model 256 --n_layers 4

# Reduce context window
python train.py --block_size 64
```

### Generation Issues

#### Repetitive Output

**Causes:**
1. Temperature too low
2. Model undertrained

**Solutions:**
```bash
# Increase temperature
python generate.py --prompt "Hello" --temperature 1.2

# Increase top-k
python generate.py --prompt "Hello" --top_k 100
```

#### Gibberish Output

**Causes:**
1. Temperature too high
2. Model undertrained
3. Tokenizer mismatch

**Solutions:**
```bash
# Lower temperature
python generate.py --prompt "Hello" --temperature 0.5

# Verify tokenizer matches checkpoint
python -c "
import torch
ckpt = torch.load('checkpoints/best_model.pt')
print('Checkpoint vocab_size:', ckpt['model_config']['vocab_size'])
"
```

### Device Issues

#### MPS Not Available

**Check availability:**
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

**Requirements:**
- macOS 12.3+
- Apple Silicon (M1/M2/M3)
- PyTorch 1.12+

#### MPS Memory Issues

```python
# Set memory limits
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Or force CPU
import torch
device = torch.device('cpu')
```

---

## Monitoring

### Training Metrics

Key metrics to track:
1. **Train loss** - Should decrease over epochs
2. **Validation loss** - Watch for overfitting
3. **Learning rate** - Verify scheduler behavior
4. **GPU memory** - Ensure not exceeding limits

### Adding TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

# In training loop
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

# Launch TensorBoard
# tensorboard --logdir runs/
```

### Memory Monitoring

```python
import torch

def log_memory():
    if torch.backends.mps.is_available():
        # MPS memory tracking (limited)
        print(f"MPS allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
    elif torch.cuda.is_available():
        print(f"CUDA allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"CUDA cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

---

## Backup and Recovery

### Backup Strategy

```bash
# Backup checkpoints
tar -czvf checkpoints_backup_$(date +%Y%m%d).tar.gz checkpoints/

# Backup to cloud (example: AWS S3)
aws s3 sync checkpoints/ s3://bucket/myllm/checkpoints/
```

### Disaster Recovery

If checkpoints are corrupted:

1. **Try loading with `weights_only=False`:**
   ```python
   checkpoint = torch.load('checkpoint.pt', weights_only=False)
   ```

2. **Extract just model weights:**
   ```python
   # If optimizer state is corrupted
   checkpoint = torch.load('checkpoint.pt', weights_only=False)
   model.load_state_dict(checkpoint['model_state_dict'])
   # Reinitialize optimizer from scratch
   ```

3. **Restore from backup:**
   ```bash
   tar -xzvf checkpoints_backup_20240101.tar.gz
   ```

---

## Upgrade Procedures

### Upgrading PyTorch

1. **Backup checkpoints**
2. **Update PyTorch:**
   ```bash
   pip install --upgrade torch
   ```
3. **Test loading checkpoints:**
   ```python
   checkpoint = torch.load('checkpoints/best_model.pt')
   ```
4. **Run test suite:**
   ```bash
   pytest tests/ -v
   ```

### Migrating Checkpoints

If model architecture changes:

```python
def migrate_checkpoint(old_path, new_path):
    """Migrate checkpoint to new architecture."""
    old = torch.load(old_path, weights_only=False)

    # Create new model
    new_model = NewGPTModel(...)

    # Map old weights to new
    new_state = {}
    for old_key, value in old['model_state_dict'].items():
        new_key = old_key.replace('old_name', 'new_name')
        new_state[new_key] = value

    # Handle missing keys
    for key in new_model.state_dict():
        if key not in new_state:
            print(f"Initializing missing key: {key}")
            new_state[key] = new_model.state_dict()[key]

    old['model_state_dict'] = new_state
    torch.save(old, new_path)
```

---

## Health Checks

### Quick Validation Script

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== MyLLM Health Check ==="

# Check Python
python --version

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check device
python -c "
import torch
if torch.backends.mps.is_available():
    print('Device: MPS (Apple Silicon)')
elif torch.cuda.is_available():
    print(f'Device: CUDA ({torch.cuda.get_device_name()})')
else:
    print('Device: CPU')
"

# Check imports
python -c "from model import GPTModel; from tokenizer import CharTokenizer; print('Imports: OK')"

# Check tests
pytest tests/ -q --tb=no && echo "Tests: PASSED" || echo "Tests: FAILED"

# Check checkpoints
if [ -f "checkpoints/best_model.pt" ]; then
    python -c "
import torch
ckpt = torch.load('checkpoints/best_model.pt', weights_only=False)
print(f\"Checkpoint: epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}\")
"
else
    echo "Checkpoint: None found"
fi

echo "=== Health Check Complete ==="
```
