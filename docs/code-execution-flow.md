# MyLLM Code Execution Flow Chart

## Overview
This document traces the complete execution flow of the MyLLM codebase, a GPT-style transformer model optimized for Apple Silicon (M3 Max).

---

## Entry Point: `train.py::main()`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROGRAM ENTRY                                      │
│                     if __name__ == "__main__": main()                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. ARGUMENT PARSING (argparse)                                              │
│  ─────────────────────────────────                                           │
│  --data        : Training data path (default: data/input.txt)                │
│  --epochs      : Number of training epochs (default: 50)                     │
│  --batch_size  : Batch size (default: 64)                                    │
│  --lr          : Learning rate (default: 3e-4)                               │
│  --block_size  : Context window size (default: 128)                          │
│  --d_model     : Model embedding dimension (default: 384)                    │
│  --n_heads     : Attention heads (default: 6)                                │
│  --n_layers    : Transformer layers (default: 6)                             │
│  --val_split   : Validation split ratio (default: 0.1)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. DATA LOADING                                                             │
│  ─────────────────                                                           │
│  • Read text file from args.data path                                        │
│  • Encoding: UTF-8                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. TOKENIZER INITIALIZATION (tokenizer.py::CharTokenizer)                   │
│  ───────────────────────────────────────────────────────────                 │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  CharTokenizer.__init__()                              │                  │
│  │  • Initialize empty char_to_id: Dict[str, int]         │                  │
│  │  • Initialize empty id_to_char: Dict[int, str]         │                  │
│  │  • Set vocab_size = 0                                  │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  CharTokenizer.build_vocab(text)                       │                  │
│  │  • Extract unique characters: sorted(list(set(text)))  │                  │
│  │  • Build char_to_id mapping: {char: index}             │                  │
│  │  • Build id_to_char mapping: {index: char}             │                  │
│  │  • Set vocab_size = len(unique_chars)                  │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  CharTokenizer.save('checkpoints/tokenizer.json')      │                  │
│  │  • Persist vocabulary to JSON file                     │                  │
│  └────────────────────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. DATA SPLITTING                                                           │
│  ──────────────────                                                          │
│  • split_idx = int(len(text) * (1 - val_split))                              │
│  • train_text = text[:split_idx]                                             │
│  • val_text = text[split_idx:]                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. DATASET CREATION (train.py::TextDataset)                                 │
│  ─────────────────────────────────────────────                               │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  TextDataset.__init__(text, tokenizer, block_size)     │                  │
│  │  • Store tokenizer reference                           │                  │
│  │  • Store block_size (context window)                   │                  │
│  │  • Encode entire text: tokenizer.encode(text)          │                  │
│  │  • Convert to torch.tensor (dtype=torch.long)          │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                                                                              │
│  Data Access Flow:                                                           │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  TextDataset.__getitem__(idx)                          │                  │
│  │  • chunk = data[idx : idx + block_size + 1]            │                  │
│  │  • x = chunk[:-1]  (input sequence)                    │                  │
│  │  • y = chunk[1:]   (target sequence, shifted by 1)     │                  │
│  │  • return (x, y)                                       │                  │
│  └────────────────────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. MODEL CREATION (model.py::GPTModel)                                      │
│  ────────────────────────────────────────                                    │
│                                                                              │
│  GPTModel.__init__(vocab_size, d_model, n_heads, n_layers, max_seq_len,     │
│                    d_ff, dropout)                                            │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  EMBEDDING LAYERS                                      │                  │
│  │  • token_embedding: nn.Embedding(vocab_size, d_model)  │                  │
│  │  • position_embedding: nn.Embedding(max_seq_len,       │                  │
│  │                                     d_model)           │                  │
│  │  • dropout: nn.Dropout(dropout)                        │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  TRANSFORMER BLOCKS (n_layers × TransformerBlock)      │                  │
│  │                                                        │                  │
│  │  TransformerBlock(d_model, n_heads, d_ff, dropout)     │                  │
│  │  ├── MultiHeadAttention(d_model, n_heads, dropout)     │                  │
│  │  │   ├── W_q: nn.Linear(d_model, d_model, bias=False)  │                  │
│  │  │   ├── W_k: nn.Linear(d_model, d_model, bias=False)  │                  │
│  │  │   ├── W_v: nn.Linear(d_model, d_model, bias=False)  │                  │
│  │  │   ├── W_o: nn.Linear(d_model, d_model)              │                  │
│  │  │   └── scale = sqrt(d_model / n_heads)               │                  │
│  │  │                                                     │                  │
│  │  ├── FeedForward(d_model, d_ff, dropout)               │                  │
│  │  │   ├── linear1: nn.Linear(d_model, d_ff)             │                  │
│  │  │   ├── gelu: nn.GELU()                               │                  │
│  │  │   └── linear2: nn.Linear(d_ff, d_model)             │                  │
│  │  │                                                     │                  │
│  │  ├── ln1: nn.LayerNorm(d_model)                        │                  │
│  │  └── ln2: nn.LayerNorm(d_model)                        │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  OUTPUT LAYERS                                         │                  │
│  │  • ln_final: nn.LayerNorm(d_model)                     │                  │
│  │  • lm_head: nn.Linear(d_model, vocab_size, bias=False) │                  │
│  │  • Weight tying: lm_head.weight = token_embedding.weight│                 │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  WEIGHT INITIALIZATION                                 │                  │
│  │  model.apply(_init_weights)                            │                  │
│  │  • Linear: normal_(mean=0, std=0.02)                   │                  │
│  │  • Embedding: normal_(mean=0, std=0.02)                │                  │
│  │  • LayerNorm: ones_(weight), zeros_(bias)              │                  │
│  └────────────────────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  7. TRAINER INITIALIZATION (train.py::Trainer)                               │
│  ───────────────────────────────────────────────                             │
│                                                                              │
│  Trainer.__init__(model, train_dataset, val_dataset, learning_rate,         │
│                   batch_size, num_epochs, device, checkpoint_dir)            │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  DEVICE SELECTION (model.py::get_device)               │                  │
│  │  Priority: MPS (Apple Silicon) > CUDA > CPU            │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  MODEL TO DEVICE                                       │                  │
│  │  model = model.to(device)                              │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  DATA LOADERS (torch.utils.data.DataLoader)            │                  │
│  │  • train_loader: shuffle=True, num_workers=4           │                  │
│  │  • val_loader: shuffle=False, num_workers=4            │                  │
│  │  • pin_memory=True for MPS device                      │                  │
│  │  • persistent_workers=True                             │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  OPTIMIZER: AdamW                                      │                  │
│  │  • lr = learning_rate                                  │                  │
│  │  • betas = (0.9, 0.95)                                 │                  │
│  │  • weight_decay = 0.1                                  │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  SCHEDULER: CosineAnnealingLR                          │                  │
│  │  • T_max = num_epochs * len(train_loader)              │                  │
│  │  • eta_min = learning_rate * 0.1                       │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  LOSS FUNCTION: CrossEntropyLoss                       │                  │
│  └────────────────────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  8. TRAINING LOOP (Trainer.train)                                            │
│  ──────────────────────────────────                                          │
│                                                                              │
│  for epoch in range(num_epochs):                                             │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  TRAIN EPOCH (Trainer.train_epoch)                     │                  │
│  │                                                        │                  │
│  │  model.train()  # Enable training mode                 │                  │
│  │                                                        │                  │
│  │  for (x, y) in train_loader:                           │                  │
│  │      │                                                 │                  │
│  │      ├─► Move to device: x, y = x.to(device),          │                  │
│  │      │                         y.to(device)            │                  │
│  │      │                                                 │                  │
│  │      ├─► Forward pass: logits = model(x)               │                  │
│  │      │   (See MODEL FORWARD PASS section below)        │                  │
│  │      │                                                 │                  │
│  │      ├─► Loss: CrossEntropyLoss(                       │                  │
│  │      │         logits.view(-1, vocab_size),            │                  │
│  │      │         y.view(-1))                             │                  │
│  │      │                                                 │                  │
│  │      ├─► optimizer.zero_grad()                         │                  │
│  │      │                                                 │                  │
│  │      ├─► loss.backward()                               │                  │
│  │      │                                                 │                  │
│  │      ├─► clip_grad_norm_(model.parameters(), 1.0)      │                  │
│  │      │                                                 │                  │
│  │      ├─► optimizer.step()                              │                  │
│  │      │                                                 │                  │
│  │      └─► scheduler.step()                              │                  │
│  │                                                        │                  │
│  │  return avg_loss                                       │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  VALIDATION (Trainer.validate) - if val_dataset        │                  │
│  │                                                        │                  │
│  │  model.eval()  # Disable dropout, etc.                 │                  │
│  │  with torch.no_grad():                                 │                  │
│  │      for (x, y) in val_loader:                         │                  │
│  │          logits = model(x.to(device))                  │                  │
│  │          loss = CrossEntropyLoss(logits, y)            │                  │
│  │                                                        │                  │
│  │  return avg_val_loss                                   │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  CHECKPOINT SAVING (Trainer.save_checkpoint)           │                  │
│  │                                                        │                  │
│  │  Saves to checkpoints/checkpoint_epoch_{n}.pt:         │                  │
│  │  • epoch                                               │                  │
│  │  • model_state_dict                                    │                  │
│  │  • optimizer_state_dict                                │                  │
│  │  • scheduler_state_dict                                │                  │
│  │  • loss                                                │                  │
│  │  • model_config (for reconstruction)                   │                  │
│  │                                                        │                  │
│  │  If best loss: also save to checkpoints/best_model.pt  │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  SAMPLE GENERATION (every 5 epochs)                    │                  │
│  │  Trainer.generate_sample()                             │                  │
│  │  (See GENERATION FLOW section below)                   │                  │
│  └────────────────────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## MODEL FORWARD PASS (GPTModel.forward)

```
Input: input_ids [batch_size, seq_len]
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. TOKEN EMBEDDING                                                          │
│  tok_emb = token_embedding(input_ids)                                        │
│  Shape: [batch_size, seq_len, d_model]                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. POSITION EMBEDDING                                                       │
│  positions = torch.arange(0, seq_len) → [batch_size, seq_len]                │
│  pos_emb = position_embedding(positions)                                     │
│  Shape: [batch_size, seq_len, d_model]                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. COMBINE & DROPOUT                                                        │
│  x = dropout(tok_emb + pos_emb)                                              │
│  Shape: [batch_size, seq_len, d_model]                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. CAUSAL MASK CREATION                                                     │
│  mask = create_causal_mask(seq_len)                                          │
│  Shape: [1, 1, seq_len, seq_len] (lower triangular)                          │
│                                                                              │
│  Example (seq_len=4):    [[1, 0, 0, 0],                                      │
│                           [1, 1, 0, 0],                                      │
│                           [1, 1, 1, 0],                                      │
│                           [1, 1, 1, 1]]                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. TRANSFORMER BLOCKS (× n_layers)                                          │
│                                                                              │
│  for block in blocks:                                                        │
│      x = block(x, mask)                                                      │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────┐                 │
│  │  TransformerBlock.forward(x, mask)                      │                 │
│  │                                                         │                 │
│  │  ┌───────────────────────────────────────────────────┐  │                 │
│  │  │  PRE-NORM ATTENTION                               │  │                 │
│  │  │  attn_out = attention(ln1(x), mask)               │  │                 │
│  │  │  x = x + dropout(attn_out)                        │  │                 │
│  │  └───────────────────────────────────────────────────┘  │                 │
│  │                          │                              │                 │
│  │                          ▼                              │                 │
│  │  ┌───────────────────────────────────────────────────┐  │                 │
│  │  │  PRE-NORM FEED-FORWARD                            │  │                 │
│  │  │  ff_out = feed_forward(ln2(x))                    │  │                 │
│  │  │  x = x + dropout(ff_out)                          │  │                 │
│  │  └───────────────────────────────────────────────────┘  │                 │
│  └─────────────────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. MULTI-HEAD ATTENTION DETAIL                                              │
│                                                                              │
│  MultiHeadAttention.forward(x, mask)                                         │
│                                                                              │
│  Q = W_q(x).view(B, T, n_heads, d_k).transpose(1,2)  → [B, H, T, d_k]        │
│  K = W_k(x).view(B, T, n_heads, d_k).transpose(1,2)  → [B, H, T, d_k]        │
│  V = W_v(x).view(B, T, n_heads, d_k).transpose(1,2)  → [B, H, T, d_k]        │
│                                                                              │
│  scores = (Q @ K.T) / sqrt(d_k)                      → [B, H, T, T]          │
│  scores = scores.masked_fill(mask == 0, -inf)                                │
│  attn_weights = softmax(scores, dim=-1)                                      │
│  attn_weights = dropout(attn_weights)                                        │
│                                                                              │
│  context = attn_weights @ V                          → [B, H, T, d_k]        │
│  context = context.transpose(1,2).view(B, T, d_model)                        │
│  output = W_o(context)                               → [B, T, d_model]       │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  7. FEED-FORWARD NETWORK DETAIL                                              │
│                                                                              │
│  FeedForward.forward(x)                                                      │
│                                                                              │
│  x = linear1(x)         → [B, T, d_ff]    (expansion: d_model → d_ff)        │
│  x = gelu(x)                              (non-linearity)                    │
│  x = dropout(x)                                                              │
│  x = linear2(x)         → [B, T, d_model] (projection: d_ff → d_model)       │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  8. FINAL PROCESSING                                                         │
│  x = ln_final(x)                         (final layer normalization)         │
│  logits = lm_head(x)                     → [B, T, vocab_size]                │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
Output: logits [batch_size, seq_len, vocab_size]
```

---

## GENERATION FLOW (GPTModel.generate)

```
Input: input_ids [batch_size, current_seq_len], max_new_tokens
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  model.eval()  # Disable dropout                                             │
│  with torch.no_grad():  # Disable gradient computation                       │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  for _ in range(max_new_tokens):                                             │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  1. CONTEXT CROPPING                                   │                  │
│  │  if seq_len > max_seq_len:                             │                  │
│  │      input_crop = input_ids[:, -max_seq_len:]          │                  │
│  │  else:                                                 │                  │
│  │      input_crop = input_ids                            │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  2. FORWARD PASS                                       │                  │
│  │  logits = model(input_crop)  → [B, T, vocab_size]      │                  │
│  │  logits = logits[:, -1, :]   → [B, vocab_size] (last)  │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  3. TEMPERATURE SCALING                                │                  │
│  │  logits = logits / temperature                         │                  │
│  │  (higher temp = more random, lower = more focused)     │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  4. TOP-K FILTERING                                    │                  │
│  │  values, indices = topk(logits, min(top_k, vocab))     │                  │
│  │  logits = full(-inf)                                   │                  │
│  │  logits.scatter_(indices, values)                      │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  5. SAMPLING                                           │                  │
│  │  probs = softmax(logits, dim=-1)                       │                  │
│  │  next_token = multinomial(probs, num_samples=1)        │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  6. APPEND TOKEN                                       │                  │
│  │  input_ids = cat([input_ids, next_token], dim=1)       │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          └──────────► repeat until max_new_tokens            │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
Output: input_ids [batch_size, original_seq_len + max_new_tokens]
```

---

## INFERENCE FLOW (generate.py)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  generate.py::main()                                                         │
│                                                                              │
│  1. Parse arguments (checkpoint, tokenizer, prompt, settings)                │
│  2. Validate checkpoint and tokenizer paths exist                            │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  TextGenerator.__init__(checkpoint_path, tokenizer_path)│                 │
│  │                                                        │                  │
│  │  • get_device() → MPS/CUDA/CPU                         │                  │
│  │  • Load tokenizer from JSON                            │                  │
│  │  • Load checkpoint (torch.load)                        │                  │
│  │  • Reconstruct model from saved config                 │                  │
│  │  • Load model weights (load_state_dict)                │                  │
│  │  • Move model to device                                │                  │
│  │  • model.eval()                                        │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  MODE SELECTION                                        │                  │
│  │                                                        │                  │
│  │  if --interactive:                                     │                  │
│  │      generator.interactive_mode()                      │                  │
│  │      (REPL loop with settings management)              │                  │
│  │                                                        │                  │
│  │  elif --prompt:                                        │                  │
│  │      generator.generate(prompt, settings...)           │                  │
│  │      (Single generation)                               │                  │
│  └────────────────────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## FILE DEPENDENCY GRAPH

```
train.py (Entry Point)
    │
    ├──► model.py
    │    ├── GPTModel (main model class)
    │    ├── TransformerBlock
    │    ├── MultiHeadAttention
    │    ├── FeedForward
    │    └── get_device()
    │
    └──► tokenizer.py
         └── CharTokenizer (encode/decode)

generate.py (Inference Entry Point)
    │
    ├──► model.py (GPTModel, get_device)
    │
    └──► tokenizer.py (CharTokenizer)

Data Files:
    data/
    ├── input.txt (default training data)
    ├── alice_in_wonderland.txt
    ├── pride_and_prejudice.txt
    ├── shakespeare_complete.txt
    ├── sherlock_holmes.txt
    └── combined_literature.txt

Output Files:
    checkpoints/
    ├── tokenizer.json
    ├── checkpoint_epoch_{n}.pt
    └── best_model.pt
```

---

## KEY HYPERPARAMETERS SUMMARY

| Parameter | Default | Description |
|-----------|---------|-------------|
| d_model | 384 | Embedding/hidden dimension |
| n_heads | 6 | Number of attention heads |
| n_layers | 6 | Number of transformer blocks |
| d_ff | d_model × 4 | Feed-forward hidden dimension |
| block_size | 128 | Context window (max_seq_len) |
| batch_size | 64 | Training batch size |
| learning_rate | 3e-4 | Initial learning rate |
| dropout | 0.1 | Dropout probability |
| weight_decay | 0.1 | AdamW weight decay |
| epochs | 50 | Number of training epochs |

---

## MEMORY LAYOUT

```
Model Parameters (approximate for defaults):
├── token_embedding:    vocab_size × 384
├── position_embedding: 128 × 384
├── 6 × TransformerBlock:
│   ├── attention:
│   │   ├── W_q: 384 × 384
│   │   ├── W_k: 384 × 384
│   │   ├── W_v: 384 × 384
│   │   └── W_o: 384 × 384
│   ├── feed_forward:
│   │   ├── linear1: 384 × 1536
│   │   └── linear2: 1536 × 384
│   ├── ln1: 384 (weights + biases)
│   └── ln2: 384 (weights + biases)
├── ln_final: 384
└── lm_head: 384 × vocab_size (tied with token_embedding)
```
