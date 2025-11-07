"""
Text generation script for the trained GPT model.
Optimized for interactive use on Apple M3 Max.
"""

import torch
import argparse
from pathlib import Path
import json

from model import GPTModel, get_device
from tokenizer import CharTokenizer


class TextGenerator:
    """Interactive text generator using the trained model."""
    
    def __init__(self, checkpoint_path: str, tokenizer_path: str):
        self.device = get_device()
        
        # Load tokenizer
        self.tokenizer = CharTokenizer()
        self.tokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer with vocab size: {self.tokenizer.vocab_size}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['model_config']
        
        # Create model
        self.model = GPTModel(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            max_seq_len=config['max_seq_len'],
            d_ff=config['d_ff'],
            dropout=0.0  # No dropout during inference
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
        print(f"Model has {self.model.count_parameters():,} parameters")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        seed: int = None
    ) -> str:
        """Generate text from a prompt."""
        if seed is not None:
            torch.manual_seed(seed)
        
        # Encode prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        
        # Generate
        print(f"\nGenerating with temperature={temperature}, top_k={top_k}...")
        generated = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
        
        # Decode
        generated_text = self.tokenizer.decode(generated[0].cpu().numpy().tolist())
        
        return generated_text
    
    def interactive_mode(self):
        """Run interactive generation mode."""
        print("\n" + "="*60)
        print("Interactive Text Generation Mode")
        print("="*60)
        print("Enter your prompt and press Enter to generate text.")
        print("Type 'quit' to exit, 'help' for options.")
        print("="*60 + "\n")
        
        # Default settings
        settings = {
            'max_tokens': 200,
            'temperature': 0.8,
            'top_k': 50
        }
        
        while True:
            prompt = input("\n> ").strip()
            
            if prompt.lower() == 'quit':
                print("Goodbye!")
                break
            
            elif prompt.lower() == 'help':
                print("\nCommands:")
                print("  quit - Exit the program")
                print("  help - Show this help message")
                print("  settings - Show current generation settings")
                print("  set <param> <value> - Change a setting")
                print("    Parameters: max_tokens, temperature, top_k")
                print("\nExamples:")
                print("  set max_tokens 500")
                print("  set temperature 0.5")
                print("  set top_k 30")
                continue
            
            elif prompt.lower() == 'settings':
                print("\nCurrent settings:")
                for key, value in settings.items():
                    print(f"  {key}: {value}")
                continue
            
            elif prompt.lower().startswith('set '):
                parts = prompt.split()
                if len(parts) == 3:
                    param, value = parts[1], parts[2]
                    if param == 'max_tokens':
                        settings['max_tokens'] = int(value)
                        print(f"Set max_tokens to {value}")
                    elif param == 'temperature':
                        settings['temperature'] = float(value)
                        print(f"Set temperature to {value}")
                    elif param == 'top_k':
                        settings['top_k'] = int(value)
                        print(f"Set top_k to {value}")
                    else:
                        print(f"Unknown parameter: {param}")
                else:
                    print("Usage: set <param> <value>")
                continue
            
            elif prompt == '':
                continue
            
            # Generate text
            try:
                generated = self.generate(
                    prompt,
                    max_new_tokens=settings['max_tokens'],
                    temperature=settings['temperature'],
                    top_k=settings['top_k']
                )
                
                print("\n" + "="*60)
                print("Generated text:")
                print("="*60)
                print(generated)
                print("="*60)
                
            except Exception as e:
                print(f"Error generating text: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained GPT model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='checkpoints/tokenizer.json',
        help='Path to tokenizer'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Text prompt for generation'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=200,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (0.1 to 2.0)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Top-k sampling parameter'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible generation'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train the model first using: python train.py")
        return
    
    if not Path(args.tokenizer).exists():
        print(f"Error: Tokenizer not found at {args.tokenizer}")
        print("Please train the model first using: python train.py")
        return
    
    # Create generator
    generator = TextGenerator(args.checkpoint, args.tokenizer)
    
    if args.interactive:
        # Interactive mode
        generator.interactive_mode()
    elif args.prompt:
        # Single generation
        generated = generator.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed
        )
        print("\nGenerated text:")
        print("="*60)
        print(generated)
        print("="*60)
    else:
        print("Please provide a prompt with --prompt or use --interactive mode")


if __name__ == "__main__":
    main()