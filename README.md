# MyLLM - A Basic Transformer Language Model

A beginner-friendly implementation of a GPT-style language model built from scratch in PyTorch, optimized for Apple Silicon (M3 Max).

## 🎯 Overview

This project implements a small transformer-based language model capable of learning from text data and generating new text. It's designed as an educational project to understand how LLMs work at a fundamental level.

### Features

- **Character-level tokenization** - Simple and intuitive
- **Transformer architecture** - Multi-head attention, feed-forward networks, positional encoding
- **Optimized for Apple M3 Max** - Uses Metal Performance Shaders (MPS) for GPU acceleration
- **Interactive text generation** - Chat-like interface for generating text
- **Efficient training** - Mixed precision, gradient accumulation, and learning rate scheduling

## 📋 Requirements

- Python 3.8 or higher
- Apple Silicon Mac (M1/M2/M3) or NVIDIA GPU (optional)
- 8GB+ RAM recommended

## 🚀 Quick Start

### 1. Set up the environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Train your first model

Train on the included Shakespeare sample:

```bash
# Quick training (10 epochs, small model)
python train.py --data data/sample.txt --epochs 10 --batch_size 32 --d_model 256 --n_layers 4

# Recommended training (50 epochs, better quality)
python train.py --data data/sample.txt --epochs 50 --batch_size 64 --d_model 384 --n_layers 6
```

**Training on Apple M3 Max:** The model automatically detects and uses the MPS backend for GPU acceleration.

### 3. Generate text

After training, generate text using:

```bash
# Interactive mode (recommended)
python generate.py --interactive

# Single generation
python generate.py --prompt "To be or not to be" --max_tokens 200 --temperature 0.8
```

## 📁 Project Structure

```
MyLLM/
├── model.py          # Transformer model architecture
├── tokenizer.py      # Character-level tokenizer
├── train.py          # Training script
├── generate.py       # Text generation script
├── data/
│   └── sample.txt    # Sample Shakespeare text
├── checkpoints/      # Saved models (created during training)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## 🎓 Understanding the Model

### Architecture

- **Embedding Layer**: Converts character IDs to vectors
- **Positional Encoding**: Adds position information to embeddings
- **Transformer Blocks** (stacked N times):
  - Multi-Head Self-Attention
  - Feed-Forward Network
  - Layer Normalization
  - Residual Connections
- **Output Layer**: Projects back to vocabulary size

### Default Model Configuration

- Vocabulary: ~65 characters (all unique characters in training data)
- Model dimension: 384
- Attention heads: 6
- Transformer layers: 6
- Context window: 128 characters
- Parameters: ~10M (varies with vocab size)

## 🛠️ Advanced Usage

### Training Options

```bash
python train.py --help

Key parameters:
  --data          Path to training text file
  --epochs        Number of training epochs (default: 50)
  --batch_size    Batch size (default: 64)
  --lr            Learning rate (default: 3e-4)
  --block_size    Context window size (default: 128)
  --d_model       Model dimension (default: 384)
  --n_heads       Number of attention heads (default: 6)
  --n_layers      Number of transformer layers (default: 6)
  --val_split     Validation split ratio (default: 0.1)
```

### Generation Options

```bash
python generate.py --help

Key parameters:
  --checkpoint    Path to model checkpoint (default: checkpoints/best_model.pt)
  --prompt        Text prompt for generation
  --max_tokens    Maximum tokens to generate (default: 200)
  --temperature   Sampling temperature 0.1-2.0 (default: 0.8)
  --top_k         Top-k sampling parameter (default: 50)
  --seed          Random seed for reproducibility
  --interactive   Run in interactive mode
```

### Using Your Own Data

1. Prepare a plain text file with your training data
2. Place it in the `data/` directory (or anywhere)
3. Train the model:

```bash
python train.py --data data/your_text.txt --epochs 100
```

**Tips for custom data:**
- Larger datasets (>1MB) generally produce better results
- More diverse text leads to more interesting generation
- Consider using books, articles, or code as training data

## 🎯 Performance Tips for M3 Max

1. **Batch Size**: Use 64-128 for optimal GPU utilization
2. **Model Size**: d_model=512, n_layers=8 works well for M3 Max
3. **Data Loading**: The script uses 4 workers by default
4. **Memory**: Monitor Activity Monitor; reduce batch_size if memory pressure occurs

## 📊 Training Progress

During training, you'll see:
- Loss values (lower is better)
- Learning rate schedule
- Sample generations every 5 epochs
- Best model saved automatically

Expected training times on M3 Max:
- Small model (256d, 4 layers): ~1 min/epoch
- Medium model (384d, 6 layers): ~2 min/epoch
- Large model (512d, 8 layers): ~3-4 min/epoch

## 🔬 Experiments to Try

1. **Different Text Styles**: Train on poetry, code, or scientific papers
2. **Temperature Settings**: Try 0.5 (focused) vs 1.5 (creative)
3. **Model Sizes**: Compare small vs large models
4. **Context Length**: Increase block_size for longer-range dependencies
5. **Custom Prompts**: Experiment with different starting texts

## 🐛 Troubleshooting

**"No module named torch"**
- Make sure you've activated the virtual environment and installed requirements

**"MPS not available"**
- Ensure you're on macOS 12.3+ with an Apple Silicon Mac
- Update PyTorch: `pip install --upgrade torch`

**Out of memory errors**
- Reduce batch_size or model size
- Close other applications to free memory

**Poor generation quality**
- Train for more epochs (try 100+)
- Use larger training dataset
- Adjust temperature (0.7-0.9 usually works best)

## 📚 Learning Resources

To understand the concepts:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide
- [GPT from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Andrej Karpathy's tutorial

## 🎉 Next Steps

Once comfortable with this basic model:
1. Implement byte-pair encoding (BPE) tokenization
2. Add attention visualization
3. Experiment with different architectures (LSTM, GRU)
4. Scale up with larger datasets
5. Implement beam search for better generation
6. Add fine-tuning capabilities

## 📄 License

This project is for educational purposes. Feel free to use and modify as needed.

---

Happy learning! 🚀 Your journey into understanding LLMs starts here.