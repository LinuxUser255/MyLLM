#!/usr/bin/env python3
"""
Training script for the GPT model, optimized for Apple M3 Max.
Includes mixed precision training and efficient data loading.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
import argparse
from pathlib import Path

from model import GPTModel, get_device
from tokenizer import CharTokenizer


class TextDataset(Dataset):
    """Dataset for character-level language modeling."""
    
    def __init__(self, text: str, tokenizer: CharTokenizer, block_size: int = 128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Encode entire text
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
        print(f"Dataset size: {len(self.data):,} tokens")
        print(f"Number of sequences: {len(self):,}")
    
    def __len__(self):
        return max(1, len(self.data) - self.block_size)
    
    def __getitem__(self, idx):
        # Get sequence and target (shifted by 1)
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class Trainer:
    """Trainer class optimized for M3 Max with MPS acceleration."""
    
    def __init__(
        self,
        model: GPTModel,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        num_epochs: int = 10,
        device: torch.device = None,
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device or get_device()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Data loaders with M3 Max optimizations
        # Pin memory for faster transfer to GPU
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # M3 Max can handle multiple workers efficiently
            pin_memory=(self.device.type == 'mps'),
            persistent_workers=True
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=(self.device.type == 'mps'),
                persistent_workers=True
            )
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs * len(self.train_loader),
            eta_min=learning_rate * 0.1
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        self.num_epochs = num_epochs
        self.best_val_loss = float('inf')
        
        # Enable mixed precision for M3 Max (faster training)
        self.use_amp = (self.device.type == 'mps')
        if self.use_amp:
            print("Using mixed precision training for M3 Max acceleration")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            # Move to device
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            logits = self.model(x)
            
            # Calculate loss (flatten for cross-entropy)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        if not self.val_dataset:
            return None
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for x, y in tqdm(self.val_loader, desc="Validation"):
            x = x.to(self.device)
            y = y.to(self.device)
            
            logits = self.model(x)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'model_config': {
                'vocab_size': self.model.token_embedding.num_embeddings,
                'd_model': self.model.d_model,
                'n_heads': self.model.blocks[0].attention.n_heads,
                'n_layers': len(self.model.blocks),
                'max_seq_len': self.model.max_seq_len,
                'd_ff': self.model.blocks[0].feed_forward.linear1.out_features,
            }
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as best if it's the best
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with loss {loss:.4f}")
        
        return checkpoint_path
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training on {self.device}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Number of batches: {len(self.train_loader)}")
        print(f"Total epochs: {self.num_epochs}\n")
        
        for epoch in range(self.num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch+1} - Train Loss: {train_loss:.4f}")
            
            # Validation
            if self.val_dataset:
                val_loss = self.validate()
                print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
            else:
                val_loss = train_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, val_loss)
            
            # Generate sample text every few epochs
            if (epoch + 1) % 5 == 0:
                self.generate_sample()
        
        print("\nTraining complete!")
        print(f"Best model saved in {self.checkpoint_dir / 'best_model.pt'}")
    
    @torch.no_grad()
    def generate_sample(self, prompt: str = " ", max_length: int = 200):
        """Generate a sample text to monitor progress."""
        self.model.eval()
        
        # Encode prompt
        tokenizer = self.train_dataset.tokenizer
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=self.device)
        
        # Generate
        generated = self.model.generate(
            input_ids,
            max_new_tokens=max_length,
            temperature=0.8,
            top_k=50
        )
        
        # Decode and print
        generated_text = tokenizer.decode(generated[0].cpu().numpy().tolist())
        print(f"\n--- Generated Sample ---")
        print(generated_text)
        print(f"--- End Sample ---\n")
        
        self.model.train()


def load_training_data(file_path: str) -> str:
    """
    Load training data from a text file.
    
    Args:
        file_path: Path to the training data file
        
    Returns:
        The text content of the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error reading the file
    """
    print(f"Loading data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Data loaded successfully. ({len(text):,} characters)")
        return text
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        raise
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        raise


def create_datasets_and_model(args, tokenizer, text):
    """Create datasets and model based on arguments."""
    # Split data
    n = len(text)
    split_idx = int(n * (1 - args.val_split))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Create datasets
    train_dataset = TextDataset(train_text, tokenizer, args.block_size)
    val_dataset = TextDataset(val_text, tokenizer, args.block_size) if val_text else None
    
    # Create model Object
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.block_size,
        d_ff=args.d_model * 4,
        dropout=0.1
    )
    
    return train_dataset, val_dataset, model


def main():
    parser = argparse.ArgumentParser(description="Train a GPT model on text data")
    parser.add_argument('--data', type=str, default='data/sample.txt')
    
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--block_size', type=int, default=128, help='Context window size')
    parser.add_argument('--d_model', type=int, default=384, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Load data
    text = load_training_data(args.data)
    
    # Create tokenizer
    tokenizer = CharTokenizer() # jumps to CharTokenizer class in tokenize.py
    tokenizer.build_vocab(text)
    tokenizer.save('checkpoints/tokenizer.json')
    
    # Create datasets and model
    train_dataset, val_dataset, model = create_datasets_and_model(args, tokenizer, text)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()