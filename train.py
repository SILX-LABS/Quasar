import os
import math
import json
import logging
import datetime
from pathlib import Path
from typing import Optional, List, Dict
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from datasets import Dataset, load_dataset  # Add explicit import
from torch.cuda.amp import autocast, GradScaler
import sentencepiece as spm
import os
import json
import logging
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
from typing import Optional, List
import datetime
import wandb  # Import wandb at the top of your file
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, GPT2Config, AutoTokenizer, GPT2Tokenizer  # Add this at the top
from contextlib import nullcontext
import torch.utils.checkpoint
from types import SimpleNamespace
from transformers import AutoTokenizer
import traceback  # Add this import at the top
import time
import torch.utils.checkpoint
from huggingface_hub import HfApi, upload_folder
import shutil
import re
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import spacy
import langdetect
import hashlib
import numpy as np
from datasets import Dataset, load_dataset

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear layers for Q, K, V projections
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.token_analyzer = nn.Linear(d_model, 1)  # Added back token analyzer

def forward(self, input_ids, attention_mask=None):
    # Get embeddings
    x = self.embedding(input_ids)
    
    # Apply token temperature mechanism
    x, temps, temp_stats = self.checkpoint_forward(self.token_temperature, x)
    
    # Process through decoder layers
    for block in self.blocks:
        x = self.checkpoint_forward(block, x)
    
    # Get logits
    logits = self.lm_head(x)
    
    # Adjust temperature dimensions
    # temps shape: [batch_size, num_heads, seq_len, seq_len]
    # Need to reshape to match logits: [batch_size, seq_len, vocab_size]
    temps = temps.mean(dim=1)  # Average across heads: [batch_size, seq_len, seq_len]
    temps = temps.mean(dim=-1, keepdim=True)  # Average across sequence: [batch_size, seq_len, 1]
    
    # Apply temperature scaling
    logits = logits * temps  # Broadcasting will work now
    
    return {
        'logits': logits,
        'temperature_stats': temp_stats
    }

class TokenTemperatureMechanism(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.context_processor = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Token analyzer with matching dimensions
        self.token_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, query, key=None, value=None, mask=None):
        # Default key and value to query if not provided
        key = query if key is None else key
        value = query if value is None else value
        
        # Process context - now unpacking all three values
        context_features, temperatures, raw_token_imp = self.context_processor(
            query=query,
            key=key,
            value=value,
            mask=mask
        )
        
        # Token-level importance
        token_importance = self.token_analyzer(context_features)
        context_importance = self.token_analyzer(context_features)
        
        return query, temperatures, {
            'mean_temp': temperatures.mean().item(),
            'min_temp': temperatures.min().item(),
            'max_temp': temperatures.max().item(),
            'token_importance': token_importance.mean().item(),
            'context_importance': context_importance.mean().item(),
            'raw_token_importance': raw_token_imp
        }

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_length=1024):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Create position encodings
        position = torch.arange(0, max_seq_length).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)  # Added closing parenthesis here
        # Initialize positional encoding matrix
        )
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(position.unsqueeze(1) * div_term)
        
        # Register buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model] or [batch_size, num_heads, seq_len, head_dim]
        """
        if len(x.shape) == 4:
            batch_size, num_heads, seq_len, head_dim = x.shape
            # Reshape PE for attention heads
            pe = self.pe[:seq_len].view(1, 1, seq_len, -1)
            pe = pe.expand(batch_size, num_heads, -1, head_dim)
        else:
            batch_size, seq_len, _ = x.shape
            pe = self.pe[:seq_len].unsqueeze(0)
            pe = pe.expand(batch_size, -1, -1)
        
        return x + pe.to(x.device)

class EfficientAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_length=1024)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.shape
        
        # Project and reshape
        q = self.q_proj(q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        q = self.rope(q)
        k = self.rope(k)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Compute output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 1024):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.max_seq_length = max_seq_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = min(x.size(1), self.max_seq_length)
        return x + self.pe[:, :seq_len]

class ContextProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Get dimensions from config
        self.d_model = config.hidden_size
        self.num_heads = config.num_attention_heads
        
        # Multi-scale feature extraction
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU()
            ) for _ in range(3)
        ])
        
        # Semantic fusion
        self.semantic_fusion = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU()
        )
        
        # Context enhancement
        self.context_enhancement = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, config.num_attention_heads)
        )
    
    def forward(self, x, mask=None):
        # Extract features at multiple scales
        features = []
        for extractor in self.feature_extractors:
            features.append(extractor(x))
        
        # Concatenate features along feature dimension
        combined = torch.cat(features, dim=-1)
        
        # Apply semantic fusion
        enhanced = self.semantic_fusion(combined)
        
        # Enhance with original input
        context = torch.cat([enhanced, x], dim=-1)
        context_features = self.context_enhancement(context)
        
        # Generate temperatures
        temperatures = torch.sigmoid(context_features)
        
        return context_features, temperatures, context_features

class TokenTemperatureMechanism(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_processor = ContextProcessor(config)
    
    def forward(self, x, mask=None):
        context_features, temperatures, raw_token_imp = self.context_processor(x, mask)
        return x, temperatures, {'raw_token_importance': raw_token_imp}

class ModelConfig:
    def __init__(self, d_model, num_attention_heads):
        self.hidden_size = d_model
        self.num_attention_heads = num_attention_heads

class TokenTemperatureMechanism(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_processor = ContextProcessor(config)
    
    def forward(self, x, mask=None):
        # Process input directly
        context_features, temperatures, raw_token_imp = self.context_processor(x, mask)
        return x, temperatures, {'raw_token_importance': raw_token_imp}

class EnhancedDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        # Create config object for context processor
        layer_config = ModelConfig(d_model=d_model, 
                                num_attention_heads=num_heads)
        
        # Add contextual processor
        self.context_processor = ContextProcessor(config=layer_config)
        self.self_attn = EfficientAttention(d_model, num_heads, dropout)
        
        # Enhanced feed-forward with context awareness
        self.ffn = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x)
        x = self.dropout(x)
        x = residual + x
        
        # Feed forward with context integration
        residual = x
        x = self.norm2(x)
        x = self.ffn(torch.cat([x, x], dim=-1))
        x = self.dropout(x)
        x = residual + x
        
        return self.norm3(x)

class QuasarTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_length, config, tokenizer=None, d_model=768, num_heads=16, num_layers=14, dropout=0.1):
        super().__init__()
        
        # Store model parameters as instance variables
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.gradient_checkpointing = False
        self.tokenizer = tokenizer  # Store tokenizer
        
        # Embedding with scale
        self.embedding_scale = math.sqrt(d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Create config object for TokenTemperatureMechanism
        temp_config = ModelConfig(d_model=d_model, 
                                num_attention_heads=num_heads)
        
        # Initialize TokenTemperatureMechanism with config
        self.token_temperature = TokenTemperatureMechanism(config=temp_config)
        
        # Normalization and dropout
        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            EnhancedDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Initialize weights properly
        self._init_weights()
        
        # Define output projection as a method instead of lambda
        def output_projection(x):
            return F.linear(x, self.token_embedding.weight)
        self.output_proj = output_projection
    
    def _init_weights(self):
        """Initialize weights with better defaults."""
        def _init_layer(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        self.apply(_init_layer)
    
    @torch.jit.ignore
    def checkpoint_forward(self, module, x, attention_mask=None):
        def custom_forward(*inputs):
            x, mask = inputs if len(inputs) > 1 else (inputs[0], None)
            
            # Handle different module types
            if isinstance(module, TokenTemperatureMechanism):
                # TokenTemperatureMechanism expects (x, mask)
                return module(x, mask)
            elif isinstance(module, EnhancedDecoderLayer):
                # EnhancedDecoderLayer expects (x)
                return module(x)
            else:
                # For any attention-based modules that need query/key/value
                return module(query=x, key=x, value=x, mask=mask)
        
        return torch.utils.checkpoint.checkpoint(
            custom_forward,
            x, attention_mask
        )
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get embeddings and scale
        x = self.token_embedding(input_ids) * self.embedding_scale
        
        # Add position embeddings
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        x = x + self.position_embedding(position_ids)
        
        # Apply normalization and dropout
        x = self.embed_norm(x)
        x = self.embed_dropout(x)
        
        # Apply temperature mechanism
        if self.gradient_checkpointing and self.training:
            x, temps, temp_stats = self.checkpoint_forward(self.token_temperature, x)
        else:
            with torch.cuda.amp.autocast():
                x, temps, temp_stats = self.token_temperature(x)
        
        # Process through transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = self.checkpoint_forward(block, x)
            else:
                with torch.cuda.amp.autocast():
                    x = block(x)
        
        # Output projection with temperature
        with torch.cuda.amp.autocast():
            x = self.output_norm(x)
            logits = self.output_proj(x)
            logits = logits * temps.mean(dim=-1, keepdim=True)
        
        outputs = {
            'logits': logits,
            'temperatures': temps,
            'temp_stats': temp_stats
        }
        
        if labels is not None:
            # Calculate loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )
            outputs['loss'] = loss  # Fix indentation here
        
        # Debug prints
        if self.training and torch.rand(1).item() < 0.01:  # Print 1% of batches
            print("\nDebug Token Shifts:")
            print("Input shape:", input_ids.shape)
            print("First 10 input tokens:", input_ids[0][:10])
            print("First 10 predicted tokens:", shift_logits[0][:10].argmax(-1))
            print("First 10 target tokens:", shift_labels[0][:10])
        
        return outputs

    def visualize_attention(self, input_ids, attention_mask=None):
        """Generate and visualize attention patterns"""
        self.eval()
        with torch.no_grad():
            outputs = self(input_ids, attention_mask)
            # Get attention weights from the last layer
            attn_weights = outputs['logits'][:, :, :].softmax(dim=-1)
            
            # Get tokens
            input_tokens = [self.tokenizer.decode([t]) for t in input_ids[0]]
            output_tokens = input_tokens[1:] + ['<eos>']
            
            # Create visualizer
            visualizer = AttentionVisualizer()
            
            # Plot and log to wandb
            visualizer.plot_attention(
                input_tokens,
                output_tokens,
                attn_weights[0].numpy(),
                save_path='attention_viz.png'
            )



class RegularizedTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.2):
        super().__init__()
        
        # Temperature-aware attention
        self.attention = SimplifiedAttention(d_model, num_heads, dropout)
        
        # Temperature-sensitive feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Temperature-aware processing
        attended = self.attention(self.norm1(x))
        x = x + self.dropout1(attended)
        
        ff = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff)
        
        return x

class SimplifiedAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.2):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Single projection for Q,K,V
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        B, L, _ = x.shape
        
        # Single QKV projection
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, L, self.d_model)
        x = self.out_proj(x)
        
        return x

class QuasarTokenizer:
    def __init__(self):
        self.tokenizer = None
        self.special_tokens = None
        self.vocab_size = None

    def load(self, config_dir="tokenizer"):
        """Load GPT2 tokenizer"""
        try:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Loaded GPT2 tokenizer with vocabulary size: {len(tokenizer)}")
            return tokenizer
        except Exception as e:
            print(f"Error loading GPT2 tokenizer: {str(e)}")
            raise e

    def encode(self, text, add_special_tokens=True):
        """Encode text to token IDs"""
        if not isinstance(text, str):
            return None
            
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token)


        words = text.split()
        for word in words:
            if word in self.vocab:
                tokens.append(word)
            else:
                tokens.append(self.unk_token)
                
        if add_special_tokens:
            tokens.append(self.eos_token)
            
        # Convert tokens to IDs
        ids = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        """Decode token IDs back to text"""
        tokens = []
        inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        for id in ids:
            if id in inverse_vocab:
                token = inverse_vocab[id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
                
        return " ".join(tokens)

    def get_vocab_size(self):
        """Get the vocabulary size"""
        return self.vocab_size

class HuggingFaceDatasetAdapter(Dataset):
    def __init__(self, tokenizer, max_length, data_generator):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # Convert generator to list of samples
        for text in data_generator:
            if isinstance(text, str):
                tokens = self.tokenizer.encode(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding='max_length',
                    return_tensors='pt'
                ).squeeze(0)
                
                if len(tokens) > 2:  # Ensure minimum sequence length
                    self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        # Create input and target sequences
        src = tokens[:-1]  # All tokens except last
        tgt = tokens[1:]   # All tokens except first
        return {
            'input_ids': src,
            'labels': tgt
        }

class TrainingConfig:
    def __init__(self, **kwargs):
        # Basic training params from previous version
        self.learning_rate = self._validate_positive_float(kwargs.get('learning_rate', 1e-4))
        self.weight_decay = self._validate_float_range(kwargs.get('weight_decay', 0.01), "weight_decay", 0, 1)
        
        # Model architecture params
        self.vocab_size = self._validate_positive_int(kwargs.get('vocab_size', 50257))
        self.max_seq_length = self._validate_positive_int(kwargs.get('max_seq_length', 1024))  # Reduced for memory
        self.d_model = self._validate_positive_int(kwargs.get('d_model', 1024))  # Reduced for memory
        self.num_heads = self._validate_positive_int(kwargs.get('num_heads', 16))  # Reduced for memory
        self.num_layers = self._validate_positive_int(kwargs.get('num_layers', 16))  # Reduced for memory
        self.dropout = self._validate_float_range(kwargs.get('dropout', 0.1), "dropout", 0, 1)
        
        # Curriculum learning params
        self.max_seq_length_start = self._validate_positive_int(kwargs.get('max_seq_length_start', 256))  # Reduced
        self.max_seq_length_end = self._validate_positive_int(kwargs.get('max_seq_length_end', 1024))  # Match new max_seq_length
        
        # Training stability params
        self.gradient_clip = self._validate_positive_float(kwargs.get('gradient_clip', 1.0))
        self.loss_scale = self._validate_positive_float(kwargs.get('loss_scale', 1.0))
        self.min_freq = self._validate_positive_int(kwargs.get('min_freq', 100))
        
        # Memory optimization parameters
        self.batch_size = self._validate_positive_int(kwargs.get('batch_size', 8))  # Reduced for memory
        self.gradient_accumulation_steps = self._validate_positive_int(kwargs.get('gradient_accumulation_steps', 16))  # Reduced for memory
        self.gradient_checkpointing = True
        self.mixed_precision = True
        self.dataset_fraction = self._validate_float_range(kwargs.get('dataset_fraction', 1.0), "dataset_fraction", 0, 1)
        
        # Steps and batches
        self.warmup_steps = self._validate_positive_int(kwargs.get('warmup_steps', 500))  # Adjusted for shorter training
        self.max_steps = self._validate_positive_int(kwargs.get('max_steps', 10000))  # Adjusted for shorter training
        self.total_batches = self._validate_positive_int(kwargs.get('total_batches', 1280000))  # Adjusted for new batch size
        
        # Paths and dataset params
        self.tokenizer_path = kwargs.get('tokenizer_path')
        self.save_dir = kwargs.get('save_dir', 'checkpoints')
        self.log_dir = kwargs.get('log_dir', 'logs')
        
        # Calculate total tokens and validate
        self.batch_size = self._validate_positive_int(kwargs.get('batch_size', 8))  # Reduced from 32
        self.total_tokens = self.total_batches * self.batch_size * self.max_seq_length
        if self.total_tokens < 10e9:  # Ensure we reach 10B tokens
            self.num_epochs = math.ceil((10e9) / self.total_tokens)
        else:
            self.num_epochs = kwargs.get('num_epochs', 15)
        
        # Add pad_token_id with default value 0
        self.pad_token_id = kwargs.get('pad_token_id', 0)
        
        # Validation steps
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps cannot be greater than max_steps")

    # Validation methods
    @staticmethod
    def _validate_positive_int(value, name="parameter"):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value
        
    @staticmethod
    def _validate_positive_float(value, name="parameter"):
        if not isinstance(value, float) or value <= 0:
            raise ValueError(f"{name} must be a positive float")
        return value
    
    @staticmethod
    def _validate_float_range(value, name="parameter", min_val=0, max_val=1):
        if not isinstance(value, float) or value < min_val or value > max_val:
            raise ValueError(f"{name} must be a float between {min_val} and {max_val}")
        return value

    @classmethod
    def from_json(cls, json_path):
        with open(json_path, 'r') as f:
            config = json.load(f)
        return cls(**config)

class TrainingStats:
    def __init__(self):
        self.train_losses = []
        self.val_perplexities = []
        self.learning_rates = []
        self.epochs = []

class QuasarTrainer:
    def __init__(self, model, config, device, train_dataset, val_dataset, tokenizer, start_epoch=0, best_loss=float('inf')):
        self.model = model
        self.config = config
        self.device = device
        self.tokenizer = tokenizer  # Add tokenizer here
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.start_epoch = start_epoch
        self.best_loss = best_loss
        self.global_step = 0  # Add global step counter
        self.hf_token = "your_access_token"
        self.repo_id = "eyad-silx/quasar-llm"
        self.api = HfApi()
        
        # Initialize optimizer with proper learning rate
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,  # Use config learning rate
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Calculate proper training steps
        samples_per_step = config.batch_size * config.gradient_accumulation_steps
        total_steps = config.total_batches // samples_per_step
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        
        print(f"\nTraining Configuration:")
        print(f"Total samples: {config.total_batches * config.batch_size * config.max_seq_length}")
        print(f"Batch size: {config.batch_size}")
        print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
        print(f"Samples per step: {samples_per_step}")
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        
        # Initialize scheduler with warmup
        from transformers import get_linear_schedule_with_warmup
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"Training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Initial learning rate: {config.learning_rate}")
        
        # Initialize loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            reduction='mean',
            label_smoothing=0.1
        )
        
        # Add gradient clipping
        self.max_grad_norm = 1.0
        
        # Move model to device
        self.model.to(device)
        
        # Calculate total steps
        total_steps = config.num_epochs * (len(self.train_dataset) // (config.batch_size * config.gradient_accumulation_steps))
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        
        # Custom learning rate schedule
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        # Initialize scheduler with cosine decay and linear warmup
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda
        )

        self.patience = 3  # Number of epochs to wait for improvement
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Gradient clipping
        self.grad_clip = 1.0

        # Add memory optimization settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Set environment variable for memory allocation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint and restore training state."""
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Filter out mismatched token analyzer weights
        filtered_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if 'token_temperature.token_analyzer' in key:
                # Skip token analyzer weights from checkpoint
                continue
            filtered_state_dict[key] = value
            
        # Load filtered state dict
        self.model.load_state_dict(filtered_state_dict, strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {self.start_epoch}")

    def train(self, train_dataset, val_dataset):
        try:
            # Create checkpoint directory
            os.makedirs("checkpoints", exist_ok=True)
            os.makedirs("temp_checkpoint", exist_ok=True)
            
            # Initialize wandb
            wandb.init(
                project="quasar-llm",
                config=self.config.__dict__,
                resume="allow"
            )
            
            # Log initial learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            wandb.log({"learning_rate": current_lr, "step": 0})
            
            # Set memory optimization flags
            torch.cuda.empty_cache()
            
            # Initialize wandb properly at start
            wandb.init(
                project="quasar-llm",
                name=f"quasar-{time.strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model_size": f"{sum(p.numel() for p in self.model.parameters())/1e6:.1f}M",
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "max_seq_length": self.config.max_seq_length,
                    "num_layers": self.config.num_layers,
                    "num_heads": self.config.num_heads
                }
            )
            
            # Setup visualization
            vis_dir = os.path.join(os.getcwd(), 'visualizations')
            visualizer = NetworkVisualizer(self.model, save_dir=vis_dir)
            visualizer.visualize_network()
            
            # Print model configuration
            print("\nModel Configuration:")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
            print(f"Hidden size: {self.model.d_model}")
            print(f"Number of layers: {len(self.model.blocks)}")
            print(f"Number of heads: {self.model.num_heads}")
            print(f"Vocab size: {self.model.vocab_size}")
            print(f"Max sequence length: {self.model.max_seq_length}")
            
            print("\nTraining Configuration:")
            print(f"Training on device: {self.device}")
            print(f"Batch size: {self.config.batch_size}")
            print(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
            print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
            print(f"Learning rate: {self.config.learning_rate}")
            print(f"Mixed precision: {True}")
            print(f"Gradient checkpointing: {self.model.gradient_checkpointing}")
            
            # Memory optimization settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable gradient checkpointing if configured
            if hasattr(self.config, 'gradient_checkpointing') and self.config.gradient_checkpointing:
                print("Enabling gradient checkpointing...")
                self.model.gradient_checkpointing_enable()
            
            # Calculate training steps
            total_samples = len(train_dataset)
            steps_per_epoch = total_samples // (self.config.batch_size * self.config.gradient_accumulation_steps)
            print(f"Total samples: {total_samples:,}")
            print(f"Steps per epoch: {steps_per_epoch:,}")
            
            # Create data loaders with single worker for Windows
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=self.collate_batch,
                num_workers=0,  # Set to 0 to avoid multiprocessing
                pin_memory=False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=self.collate_batch,
                num_workers=0,  # Set to 0 to avoid multiprocessing
                pin_memory=False
            )

            best_val_loss = float('inf')
            self.global_step = 0

            for epoch in range(self.start_epoch, self.config.num_epochs):
                self.model.train()
                total_loss = 0
                pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}')
                batch_start_time = time.time()  # Add timer

                # Initialize log_dict at the start of epoch
                log_dict = {
                    'epoch': epoch,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'batch_losses': [],
                    'temperature_stats': {},
                }

                for i, batch in enumerate(train_loader):
                    try:
                        # Ensure batch is properly tokenized and moved to GPU
                        if isinstance(batch, str):
                            inputs = self.tokenizer(
                                batch,
                                padding=True,
                                truncation=True,
                                max_length=self.config.max_seq_length,
                                return_tensors="pt"
                            )
                        else:
                            inputs = batch
                        
                        # Move input tensors to GPU
                        input_ids = inputs['input_ids'].cuda(non_blocking=True)
                        attention_mask = inputs['attention_mask'].cuda(non_blocking=True)
                        
                        # Create labels for language modeling (don't shift padding tokens)
                        labels = input_ids.clone()
                        
                        # Create attention mask for padding
                        padding_mask = attention_mask.bool()
                        
                        # Shift labels and inputs, preserving padding
                        labels = torch.where(padding_mask, labels, labels.roll(-1))
                        labels[:, -1] = self.config.pad_token_id  # Set last position to padding
                        
                        # Create input by using original sequence except last token
                        model_input = input_ids
                        
                        # For debugging
                        if i % 100 == 0:
                            print("\nInput-Label Alignment Check:")
                            for j in range(min(1, input_ids.size(0))):  # Check first sequence
                                input_text = self.tokenizer.decode(input_ids[j])
                                target_text = self.tokenizer.decode(labels[j])
                                print(f"\nOriginal text: {input_text[:100]}")
                                print(f"Target text:   {target_text[:100]}")
                        
                        # Forward pass with automatic mixed precision
                        with torch.cuda.amp.autocast():
                            outputs = self.model(
                                input_ids=model_input,
                                attention_mask=attention_mask,
                                labels=labels
                            )
                            loss = outputs['loss']
                            current_loss = loss.item()
                            
                            # Add the safety check here
                            if current_loss < 0.05:  # Dangerously low loss
                                print("\nERROR: Loss has decreased to suspicious levels")
                                print("Training stopped to prevent potential issues")
                                print(f"Current loss: {current_loss}")
                                print("Please check model and loss calculation")
                                raise ValueError("Loss decreased to suspicious levels")

                            # Loss debugging prints
                            print(f"\nLoss debugging:")
                            print(f"Raw loss value: {loss.item()}")
                            print(f"Loss shape: {loss.shape}")
                            if hasattr(outputs, 'logits'):
                                print(f"Logits shape: {outputs['logits'].shape}")
                                # Print sample probabilities
                                probs = torch.softmax(outputs['logits'][0, 0], dim=-1)
                                top_p, top_idx = torch.topk(probs, 5)
                                print(f"Top 5 probabilities for first prediction: {top_p}")
                                print(f"Top 5 tokens: {[self.tokenizer.decode([idx.item()]) for idx in top_idx]}")

                        if i % 500 == 0:
                            self.visualize_predictions(batch, outputs)

                        if i % 100 == 0:
                            print("\nSample Text Verification:")
                            # Decode input
                            input_text = self.tokenizer.decode(input_ids[0][:50])
                            print(f"Input text (first 50 tokens): {input_text}")
                            
                            # Decode labels
                            label_text = self.tokenizer.decode(labels[0][:50])
                            print(f"Target text (first 50 tokens): {label_text}")
                            
                            # Print token statistics
                            unique_tokens = len(torch.unique(input_ids))
                            print(f"Unique tokens in batch: {unique_tokens}")
                            print(f"Sequence length: {input_ids.size(1)}")
                            
                            # Monitor attention patterns
                            if hasattr(outputs, 'attentions'):
                                att = outputs.attentions[0]  # First layer attention
                                print(f"Attention max: {att.max().item():.3f}")
                                print(f"Attention mean: {att.mean().item():.3f}")
                        
                        # Backward pass with gradient scaling
                        self.scaler.scale(loss).backward()
                        
                        try:
                            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                                # Unscale gradients before clipping
                                self.scaler.unscale_(self.optimizer)
                                
                                # Gradient clipping
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                
                                # Optimizer step with scaler
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                                self.optimizer.zero_grad(set_to_none=True)
                                self.scheduler.step()
                        except RuntimeError as e:
                            print(f"Error in optimization step: {str(e)}")
                            self.optimizer.zero_grad(set_to_none=True)
                            continue
                        
                        # Calculate batch time and log metrics
                        batch_time = time.time() - batch_start_time
                        current_lr = self.optimizer.param_groups[0]['lr']
                        
                        # Update wandb logging
                        log_dict.update({
                            'learning_rate': current_lr,
                            'batch_speed': batch_time,
                            'gpu_memory': torch.cuda.max_memory_allocated() / 1e9
                        })
                        
                        wandb.log(log_dict, step=self.global_step)
                        batch_start_time = time.time()  # Reset timer

                        # Calculate perplexity and log metrics
                        perplexity = torch.exp(loss)
                        
                        # Wandb logging
                        log_dict = {
                            "train/loss": loss.item(),
                            "train/perplexity": perplexity.item(),
                            "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                            "train/epoch": epoch,
                            "train/step": self.global_step,
                        }
                        
                        if 'temperatures' in outputs:
                            log_dict.update({
                                "temperature/mean": outputs['temperatures'].mean().item(),
                                "temperature/min": outputs['temperatures'].min().item(),
                                "temperature/max": outputs['temperatures'].max().item(),
                                "temperature/std": outputs['temperatures'].std().item(),
                            })
                        
                        # Safely add temperature stats if they exist
                        temp_stats = outputs.get('temp_stats', {})
                        if temp_stats:
                            log_dict.update({
                                "model/coord_influence": temp_stats.get('coord_influence', 0),
                                "model/long_range_influence": temp_stats.get('long_range_influence', 0),
                                "temp_stats/entropy": temp_stats.get('entropy', 0),
                                "temp_stats/diversity": temp_stats.get('diversity', 0),
                            })

                        wandb.log(log_dict, step=self.global_step)
                        self.global_step += 1

                        # Update progress bar with more detailed info
                        pbar.update(1)
                        pbar.set_postfix({
                            'raw_loss': f"{loss.item():.4f}",
                            'ppl': f"{perplexity.item():.2f}",
                            'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        })

                    except Exception as e:
                        print(f"Error in batch {i}: {str(e)}")
                        traceback.print_exc()
                        continue

                pbar.close()


                # Validation phase
                val_loss = self.validate(val_loader)
                print(f"\nValidation loss: {val_loss:.4f}")
                
                # Log epoch metrics
                wandb.log({
                    'epoch': epoch,
                    'train/loss_epoch': total_loss / len(train_loader),
                    'val/loss': val_loss
                })
                
                # Save checkpoint with full state
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                    'config': self.config,
                    'scaler_state_dict': self.scaler.state_dict()
                }
                torch.save(checkpoint, f"checkpoints/checkpoint_epoch_{epoch+1}.pt")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"New best validation loss: {val_loss:.4f}")
                    self.save_checkpoint(epoch, val_loss, is_best=True)
                    
                # Save to HuggingFace Hub
                self.save_to_hub(epoch, val_loss)

        except Exception as e:
            print(f"Error in training: {str(e)}")
            traceback.print_exc()
            raise e

    def train_step(self, batch, batch_idx):
        """Single training step with improved error handling"""
        try:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            current_loss = loss.item()
            
            # Scale loss by gradient accumulation steps
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                
                # Clear gradients
                self.optimizer.zero_grad()
            
            return loss.item()
            
        except Exception as e:
            print(f"Error in training step: {str(e)}")
            return None

    def collate_batch(self, batch):
        """Collate batch of samples into model input format"""
        if isinstance(batch[0], str):
            texts = batch
        elif isinstance(batch[0], dict):
            texts = [item['text'] for item in batch]
        else:
            raise ValueError(f"Unexpected batch format: {type(batch[0])}")
        
        # Tokenize all texts in the batch
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )
        
        return inputs

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save checkpoint with proper directory creation"""
        try:
            # Ensure checkpoint directory exists
            os.makedirs("checkpoints", exist_ok=True)
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'config': self.config,
                'scaler_state_dict': self.scaler.state_dict()
            }
            
            # Save regular checkpoint
            checkpoint_path = os.path.join("checkpoints", f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # Save best model if needed
            if is_best:
                best_path = os.path.join("checkpoints", "best_model.pt")
                torch.save(checkpoint, best_path)
                print(f"Saved best model to {best_path}")
                
        except Exception as e:
            print(f"Error in save_checkpoint: {str(e)}")
            traceback.print_exc()
        
    def save_to_hub(self, epoch, val_loss, is_best=False):
        """Save checkpoint to HuggingFace Hub with custom model handling"""
        try:
            # Create temp directory
            temp_dir = Path("temp_checkpoint")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model state dict
            model_path = temp_dir / "pytorch_model.bin"
            torch.save(self.model.state_dict(), model_path)
            
            # Save config
            config_dict = {
                "model_type": "quasar",
                "d_model": self.config.d_model,
                "num_heads": self.config.num_heads,
                "num_layers": self.config.num_layers,
                "vocab_size": self.config.vocab_size,
                "max_seq_length": self.config.max_seq_length,
                "dropout": self.config.dropout,
                "architecture": "QuasarTransformer"
            }
            
            config_path = temp_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            # Save training state
            training_state = {
                'epoch': epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'global_step': self.global_step,
                'config': self.config.__dict__,
                'scaler_state_dict': self.scaler.state_dict()
            }
            torch.save(training_state, temp_dir / "training_state.pt")
            
            # Upload to hub
            commit_message = f"Checkpoint epoch {epoch}"
            if is_best:
                commit_message += " (Best)"
            
            self.api.upload_folder(
                folder_path=str(temp_dir),
                repo_id=self.repo_id,
                repo_type="model",
                token=self.hf_token,
                commit_message=commit_message
            )
            print(f"Successfully uploaded checkpoint for epoch {epoch}")
            
        except Exception as e:
            print(f"Error saving to HuggingFace Hub: {e}")
            traceback.print_exc()
        
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def validate(self, val_loader):
        """Evaluate the model on the validation set"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    # Handle string or dict input
                    if isinstance(batch, str):
                        inputs = self.tokenizer(
                            batch,
                            padding=True,
                            truncation=True,
                            max_length=self.config.max_seq_length,
                            return_tensors="pt"
                        )
                    else:
                        inputs = batch
                    
                    # Move to GPU
                    input_ids = inputs['input_ids'].cuda(non_blocking=True)
                    attention_mask = inputs['attention_mask'].cuda(non_blocking=True)
                    
                    # Forward pass
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                        
                        # Proper loss calculation for validation
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = input_ids[..., 1:].contiguous()
                        
                        loss = self.criterion(
                            shift_logits.view(-1, self.model.vocab_size),
                            shift_labels.view(-1)
                        )
                        
                        # Count tokens for proper averaging
                        num_tokens = shift_labels.ne(self.config.pad_token_id).sum().item()
                        if num_tokens > 0:  # Only add if we have valid tokens
                            total_loss += loss.item() * num_tokens
                            total_tokens += num_tokens
                    
                except Exception as e:
                    print(f"Error in validation batch: {str(e)}")
                    continue

            # Prevent division by zero
            if total_tokens == 0:
                print("Warning: No valid tokens found in validation set")
                return float('inf')  # Return infinity to indicate invalid loss
                
            return total_loss / total_tokens  # Proper per-token loss
        
    def visualize_predictions(self, batch, outputs, num_samples=3):
        """Visualize model predictions compared to targets."""
        try:
            # Get logits from outputs
            if isinstance(outputs, dict):
                logits = outputs.get('logits', None)
            else:
                logits = outputs.logits if hasattr(outputs, 'logits') else None
            
            if logits is None:
                print("No logits found in model outputs")
                return
            
            # Get input_ids and labels, handling different input types
            if isinstance(batch, dict):
                input_ids = batch['input_ids']
                labels = batch.get('labels', None)
            elif isinstance(batch, (list, str)):
                # Handle raw text input
                if isinstance(batch, str):
                    batch = [batch]
                encoded = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                input_ids = encoded['input_ids']
                labels = None
            elif hasattr(batch, 'input_ids'):  # Handle BatchEncoding
                input_ids = batch.input_ids
                labels = getattr(batch, 'labels', None)
            else:
                input_ids = batch
                labels = None
            
            # Ensure input_ids is on the correct device
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.to(logits.device)
            
            # Get batch size safely
            batch_size = input_ids.shape[0] if len(input_ids.shape) > 1 else 1
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
            
            print(f"\nVisualization of {num_samples} samples:")
            for i in range(min(num_samples, batch_size)):
                try:
                    # Get predictions for the current sequence
                    pred_tokens = logits[i].argmax(dim=-1)
                    
                    # Decode texts
                    input_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    prediction_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                    
                    print(f"\nSample {i+1}:")
                    print("Input:", input_text[:100], "..." if len(input_text) > 100 else "")
                    print("Prediction:", prediction_text[:100], "..." if len(prediction_text) > 100 else "")
                    
                    if labels is not None:
                        if isinstance(labels, torch.Tensor):
                            target_text = self.tokenizer.decode(labels[i], skip_special_tokens=True)
                            print("Target:", target_text[:100], "..." if len(target_text) > 100 else "")
                    
                    print("-" * 50)
                except Exception as e:
                    print(f"Error processing sample {i}: {str(e)}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def visualize_predictions(self, batch, outputs, num_samples=3):
        """Visualize model predictions compared to targets."""
        try:
            # Get logits from outputs
            if isinstance(outputs, dict):
                logits = outputs.get('logits', None)
            else:
                logits = outputs.logits if hasattr(outputs, 'logits') else None
            
            if logits is None:
                print("No logits found in model outputs")
                return
            
            # Get input_ids and labels, handling different input types
            if isinstance(batch, dict):
                input_ids = batch['input_ids']
                labels = batch.get('labels', None)
            elif isinstance(batch, (list, str)):
                # Handle raw text input
                if isinstance(batch, str):
                    batch = [batch]
                encoded = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                input_ids = encoded['input_ids']
                labels = None
            elif hasattr(batch, 'input_ids'):  # Handle BatchEncoding
                input_ids = batch.input_ids
                labels = getattr(batch, 'labels', None)
            else:
                input_ids = batch
                labels = None
            
            # Ensure input_ids is on the correct device
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.to(logits.device)
            
            # Get batch size safely
            batch_size = input_ids.shape[0] if len(input_ids.shape) > 1 else 1
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
            
            print(f"\nVisualization of {num_samples} samples:")
            for i in range(min(num_samples, batch_size)):
                try:
                    # Get predictions for the current sequence
                    pred_tokens = logits[i].argmax(dim=-1)
                    
                    # Decode texts
                    input_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    prediction_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                    
                    print(f"\nSample {i+1}:")
                    print("Input:", input_text[:100], "..." if len(input_text) > 100 else "")
                    print("Prediction:", prediction_text[:100], "..." if len(prediction_text) > 100 else "")
                    
                    if labels is not None:
                        if isinstance(labels, torch.Tensor):
                            target_text = self.tokenizer.decode(labels[i], skip_special_tokens=True)
                            print("Target:", target_text[:100], "..." if len(target_text) > 100 else "")
                    
                    print("-" * 50)
                except Exception as e:
                    print(f"Error processing sample {i}: {str(e)}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            import traceback
            traceback.print_exc()
    
def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    init_process_group(
        "nccl", 
        rank=rank,  
        world_size=world_size,
        timeout=datetime.timedelta(minutes=30)
    )
    
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True

def cleanup_distributed():
    if torch.distributed.is_initialized():
        destroy_process_group()

def train_distributed(rank, world_size, config_path):
    setup_distributed(rank, world_size)
    
    config = TrainingConfig.from_json(config_path)
    trainer = QuasarTrainer(config, local_rank=rank)
    
    train_dataset = HuggingFaceDatasetAdapter(
        dataset_name=config.dataset_name,
        tokenizer=trainer.tokenizer,
        max_length=config.max_seq_length,
        split="train"
    )
    
    val_dataset = HuggingFaceDatasetAdapter(
        dataset_name=config.dataset_name,
        tokenizer=trainer.tokenizer,        
        max_length=config.max_seq_length,
        split="validation"
    )
    
    trainer.train(train_dataset, val_dataset)
    cleanup_distributed()

def evaluate_model(model, val_loader):
    """Evaluate model and return metrics"""
    model.eval()
    total_loss = 0
    total_perplexity = 0
    steps = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']
            perplexity = torch.exp(loss)
            
            total_loss += loss.item()
            total_perplexity += perplexity.item()
            steps += 1
    
    return {
        'loss': total_loss / steps,
        'perplexity': total_perplexity / steps
    }

def automated_temperature_sweep(model, data_loader, temperature_range=(0.35, 1.5), num_tests=10):
    temperatures = torch.linspace(temperature_range[0], temperature_range[1], num_tests)
    results = {}

    for temp in temperatures:
        model.temp_scale.data.fill_(temp)  # Set the temperature
        performance_metrics = evaluate_model(model, data_loader)  # Now defined
        results[temp.item()] = performance_metrics

    return results

class ReasoningModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Add temperature processing layer
        self.temp_processor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.parse_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Modified to accept temperature info
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.hypothesis_generator = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # Modified for temp-aware hypotheses
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.evaluation_layer = nn.Sequential(
            nn.Linear(d_model * 4, d_model),  # Modified to include temperature
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x, mission_tokens, hidden_potential, temp_info=None):
        # Process temperature information if available
        temp_features = self.temp_processor(
            torch.cat([x, temp_info['temperatures']], dim=-1)
        ) if temp_info is not None else torch.zeros_like(x)
        
        # Stage 1: Parse input with context and temperature
        parsed = self.parse_layer(torch.cat([x, temp_features], dim=-1))
        
        # Stage 2: Generate temperature-aware hypotheses
        combined = torch.cat([parsed, mission_tokens, temp_features], dim=-1)
        hypotheses = self.hypothesis_generator(combined)
        
        # Stage 3: Evaluate with all context including temperature
        evaluation_input = torch.cat([
            hypotheses,
            mission_tokens,
            hidden_potential,
            temp_features
        ], dim=-1)
        confidence = self.evaluation_layer(evaluation_input)
        
        return hypotheses, confidence

class ContextualUnderstanding(nn.Module):
    def __init__(self, d_model, num_heads=16, hidden_dim=512):
        super().__init__()
        self.hierarchical_attention = nn.ModuleList([
            EfficientAttention(d_model, num_heads) 
            for _ in range(3)
        ])
        
        self.semantic_processor = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )
        
        self.pattern_memory = nn.Parameter(torch.randn(512, d_model))
        self.pattern_attention = EfficientAttention(d_model, num_heads)

    def forward(self, x, mask=None):
        multi_scale_features = []
        current_x = x
        
        for attention in self.hierarchical_attention:
            current_x = attention(current_x, current_x, current_x, mask)
            multi_scale_features.append(current_x)
        
        combined = torch.cat(multi_scale_features, dim=-1)
        semantic_features = self.semantic_processor(combined)
        
        # Pattern attention
        batch_size = x.size(0)
        pattern_memory = self.pattern_memory.unsqueeze(0).expand(batch_size, -1, -1)
        pattern_context = self.pattern_attention(
            semantic_features,
            pattern_memory,
            pattern_memory,
            mask
        )
        
        return pattern_context, semantic_features

class TokenPredictionModule(nn.Module):
    def __init__(self, d_model, num_heads=16):      
        super().__init__()
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2)  # mean and variance
        )
        
        self.hypothesis_generator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(3)  # Generate multiple hypotheses
        ])
        
        self.confidence_scorer = EfficientAttention(d_model, num_heads)
        
    def forward(self, x, context_features, hidden_tokens=None):
        # Get uncertainty estimates
        uncertainty = self.uncertainty_estimator(x)
        mean, log_var = uncertainty.chunk(2, dim=-1)
        
        # Include hidden token information
        if hidden_tokens is not None:
            x = torch.cat([x, hidden_tokens], dim=-1)
            
        hypotheses = []
        for generator in self.hypothesis_generator:
            combined = torch.cat([x, context_features], dim=-1)
            hypothesis = generator(combined)
            hypotheses.append(hypothesis)
        
        stacked_hypotheses = torch.stack(hypotheses, dim=1)
        confidence_scores = self.confidence_scorer(
            stacked_hypotheses, 
            stacked_hypotheses, 
            stacked_hypotheses
        )
        
        return {
            'predictions': stacked_hypotheses,
            'confidence': confidence_scores,
            'uncertainty': {
                'mean': mean,
                'log_var': log_var
            }
        }

class DatasetPreprocessor:
    def __init__(self, tokenizer, min_tokens=5, max_tokens=2048, lang='en', 
                min_entropy=2.0, repetition_thresholds=(0.3, 0.4)):
        self.tokenizer = tokenizer
        # Make spaCy optional
        try:
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'lemmatizer'])
        except:
            print("SpaCy model not found, continuing without language model support...")
            self.nlp = None
            
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.target_lang = lang
        
        # Configurable quality thresholds
        self.min_entropy = min_entropy
        self.rep_threshold_long = repetition_thresholds[0]  # For longer texts
        self.rep_threshold_short = repetition_thresholds[1]  # For shorter texts
        
        # Enhanced noise patterns
        self.noise_patterns = {
            'html_tags': r'<[^>]+>',
            'urls': r'http[s]?://\S+',
            'emails': r'\S+@\S+\.\S+',
            'repeated_chars': r'(.)\1{3,}',
            'special_chars': r'[^\w\s.,!?\'"(){}[\]:;/-]',
            'invisible_chars': r'[\x00-\x1F\x7F]+',
            'multiple_spaces': r'\s+'
        }
        
        self.token_cache = {}
        
    def get_token_count(self, text: str) -> int:
        """More robust token counting with preprocessing"""
        # Normalize whitespace before hashing
        normalized_text = ' '.join(text.split())
        text_hash = hashlib.md5(normalized_text.encode()).hexdigest()
        
        if text_hash not in self.token_cache:
            # Quick preliminary check using tokenizer
            try:
                token_count = len(self.tokenizer.encode(
                    normalized_text, 
                    add_special_tokens=False,
                    truncation=False
                ))
                self.token_cache[text_hash] = token_count
            except Exception:
                # Fallback to word count if tokenization fails
                token_count = len(normalized_text.split())
                
        return self.token_cache[text_hash]
    
    def check_repetition(self, words: list) -> bool:
        """Enhanced repetition detection with configurable thresholds"""
        if len(words) < 5:
            return True
            
        word_freq = Counter(words)
        total_words = len(words)
        
        # Dynamic threshold based on text length
        threshold = (
            self.rep_threshold_long if total_words > 20 
            else self.rep_threshold_short
        )
        
        # Frequency check
        max_freq = max(word_freq.values())
        if max_freq > total_words * threshold:
            return False
            
        # Entropy check with configurable threshold
        entropy = sum(-freq/total_words * math.log2(freq/total_words) 
                    for freq in word_freq.values())
        
        return entropy >= self.min_entropy
    
    def normalize_text(self, text: str) -> str:
        """Optimized text normalization"""
        if not isinstance(text, str):
            return ""
            
        # Apply cleaning patterns
        for pattern_name, pattern in self.noise_patterns.items():
            text = re.sub(pattern, ' ' if pattern_name != 'multiple_spaces' else ' ', text)
        
        # Basic normalization
        text = text.lower().strip()
        
        # Efficient tokenization without full pipeline
        if self.nlp is not None:
            doc = self.nlp.make_doc(text)
            tokens = [token.text for token in doc 
                    if not token.is_space and not token.is_punct]
        else:
            tokens = text.split()
        
        return ' '.join(tokens)
    
    def filter_entry(self, text: str) -> bool:
        """Enhanced filtering with better edge case handling"""
        if not text or not isinstance(text, str):
            return False
            
        # Quick token count check
        token_count = self.get_token_count(text)
        if token_count < self.min_tokens or token_count > self.max_tokens:
            return False
            
        # Language check with error handling
        try:
            if self.nlp is not None:
                detected_lang = langdetect.detect(text)
                if detected_lang != self.target_lang:
                    return False
        except Exception:
            return False
            
        # Enhanced repetition check
        words = text.split()
        if not self.check_repetition(words):
            return False
                
        return True

class DatasetBalancer:
    def __init__(self, target_proportions: Dict[str, float]):
        self.target_proportions = target_proportions
        
    def balance_datasets(self, datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """Balance datasets according to target proportions"""
        total_size = sum(len(ds) for ds in datasets.values())
        balanced_datasets = {}
        
        for source, proportion in self.target_proportions.items():
            if source in datasets:
                target_size = int(total_size * proportion)
                current_size = len(datasets[source])
                
                if current_size > target_size:
                    # Subsample if too large
                    indices = np.random.choice(current_size, target_size, replace=False)
                    balanced_datasets[source] = datasets[source].select(indices)
                else:
                    balanced_datasets[source] = datasets[source]
                    
        return balanced_datasets
    
def split_text_into_windows(text, tokenizer, max_length, stride=128):
    """Split long text into overlapping windows."""
    # Tokenize the full text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # If text is short enough, return as single window
    if len(tokens) <= max_length:
        return [text]
    
    # Calculate number of windows needed
    window_size = max_length - 2  # Account for special tokens
    windows = []
    
    # Create overlapping windows
    for i in range(0, len(tokens), window_size - stride):
        window_tokens = tokens[i:i + window_size]
        window_text = tokenizer.decode(window_tokens, skip_special_tokens=True)
        windows.append(window_text)
        
        # Break if we have enough context
        if len(windows) >= 3:  # Limit to 3 windows per text
            break
    
    return windows

def load_and_preprocess_data(tokenizer, max_length, dataset_fraction=1.0):
    try:
        preprocessor = DatasetPreprocessor(tokenizer)
        
        MAX_SEQ_LENGTH = 2048        # Increased to match model config
        SAFETY_MARGIN = 50
        STRIDE = 256                 # Increased for longer sequences
        TOTAL_SAMPLES = 1000000      # Increased to match config
        target_proportions = {
            'books': 0.5,            # Increased books proportion
            'wikipedia': 0.35,       # Slightly increased Wikipedia
            'stackexchange': 0.15    # Reduced StackExchange
        }
        
        all_processed_texts = []
        sources = ['books', 'wikipedia', 'stackexchange']
        overall_progress = tqdm(total=len(sources), desc="Overall Dataset Progress")
        
        for source in sources:
            try:
                print(f"\nLoading {source} dataset...")
                source_samples = int(TOTAL_SAMPLES * target_proportions[source])
                print(f"Target samples for {source}: {source_samples}")
                
                raw_dataset = load_dataset(
                    "togethercomputer/RedPajama-Data-1T-Sample",
                    split='train',
                    streaming=True,
                    trust_remote_code=True
                )
                
                filtered_texts = []
                source_progress = tqdm(
                    total=source_samples,
                    desc=f"Processing {source}",
                    position=1,
                    leave=False
                )
                
                for item in raw_dataset.take(source_samples):
                    try:
                        text_windows = split_text_into_windows(
                            item['text'],
                            tokenizer,
                            MAX_SEQ_LENGTH - SAFETY_MARGIN,
                            STRIDE
                        )
                        
                        for window_text in text_windows:
                            if preprocessor.filter_entry(window_text):
                                filtered_texts.append({
                                    'text': preprocessor.normalize_text(window_text),
                                    'source': source
                                })
                                
                        source_progress.update(1)
                        
                    except Exception as e:
                        print(f"Error processing text: {str(e)}")
                        continue
                
                source_progress.close()
                
                if filtered_texts:
                    print(f"\nLoaded {len(filtered_texts):,} examples from {source}")
                    all_processed_texts.extend(filtered_texts)
                    
                    print(f"\nSource: {source}")
                    print(f"Number of examples: {len(filtered_texts)}")
                    
                    # Sample and analyze tokens
                    sample_size = min(1000, len(filtered_texts))
                    sample_texts = filtered_texts[:sample_size]
                    
                    total_tokens = 0
                    token_freq = Counter()
                    for item in sample_texts:
                        tokens = tokenizer.encode(item['text'])
                        total_tokens += len(tokens)
                        token_freq.update(tokens)
                    
                    avg_tokens = total_tokens / sample_size
                    print(f"Average tokens per example: {avg_tokens:.2f}")
                
                overall_progress.update(1)
                
            except Exception as e:
                print(f"Error processing source {source}: {str(e)}")
                overall_progress.update(1)
                continue
        
        overall_progress.close()
        
        # Create the final dataset
        print("\nCreating final datasets...")
        if not all_processed_texts:
            print("Warning: No texts were collected!")
            empty_dataset = Dataset.from_dict({'text': [], 'source': []})
            return (empty_dataset, empty_dataset)
            
        combined_dataset = Dataset.from_dict({
            'text': [item['text'] for item in all_processed_texts],
            'source': [item['source'] for item in all_processed_texts]
        })
        
        # Shuffle the dataset
        combined_dataset = combined_dataset.shuffle(seed=42)
        
        # Split into train and validation sets (90/10)
        total_size = len(combined_dataset)
        train_size = int(0.9 * total_size)
        
        train_dataset = combined_dataset.select(range(train_size))
        val_dataset = combined_dataset.select(range(train_size, total_size))
        
        print(f"\nFinal dataset sizes:")
        print(f"Training set: {len(train_dataset):,} examples")
        print(f"Validation set: {len(val_dataset):,} examples")
        
        # Return as explicit tuple
        return (train_dataset, val_dataset)
        
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        traceback.print_exc()
        empty_dataset = Dataset.from_dict({'text': [], 'source': []})
        return (empty_dataset, empty_dataset)

class AttentionVisualizer:
    def __init__(self):
        self.fig_size = (10, 8)
        
    def plot_attention(self, input_tokens, output_tokens, attention_weights, save_path):
        plt.figure(figsize=self.fig_size)
        sns.heatmap(
            attention_weights,
            xticklabels=input_tokens,
            yticklabels=output_tokens,
            cmap='viridis'
        )
        plt.title('Attention Weights Visualization')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def init_config():
    """Initialize training configuration with extreme memory optimization"""
    config = TrainingConfig(
        # Model architecture params - REDUCED for memory constraints
        vocab_size=50257,
        max_seq_length=1024,     # Reduced from 2048
        d_model=1024,            # Reduced from 2048
        num_heads=16,            # Reduced from 32
        num_layers=16,           # Reduced from 24
        dropout=0.1,
        
        # Training optimization for A100 - CRITICAL MEMORY SETTINGS
        batch_size=8,            # Reduced from 32
        gradient_accumulation_steps=16,  # Reduced from 32
        gradient_checkpointing=True,
        mixed_precision=True,
        
        # Memory optimization flags
        use_flash_attention=True,  # Enable flash attention
        empty_cache_between_batches=True,  # Aggressive memory clearing
        
        # Learning parameters
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=500,
        max_steps=10000,
        total_samples=1280000,   # Adjusted for new batch size
        total_batches=1280000,   # Same as total_samples
        
        # Dataset params
        dataset_fraction=1.0,
        
        # Memory optimization
        gradient_clip=1.0,
        loss_scale=1.0,
        
        # Paths
        save_dir='checkpoints',
        log_dir='logs'
    )
    
    # Set PyTorch memory optimization flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
    
    return config

def load_tokenizer():
    """Load GPT2 tokenizer with padding token."""
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        raise e

def init_wandb(config, tokenizer):
    """Initialize wandb with config and tokenizer details."""
    return wandb.init(
                project="quasar-llm",
                config={
            "model_size": f"{config.vocab_size * config.d_model}",
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "tokenizer": tokenizer.__class__.__name__,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "max_seq_length": config.max_seq_length
        }
    )

def main():
    try:
        # Step 1: Initialize configuration
        config = init_config()
        print("Configuration initialized")

        # Step 2: Initialize tokenizer
        tokenizer = load_tokenizer()
        print("Tokenizer initialized")

        # Step 3: Initialize wandb with both config and tokenizer
        run = init_wandb(config, tokenizer)
        print("Wandb initialized")
        
        # Initialize model
        model = QuasarTransformer(
            vocab_size=len(tokenizer),
            max_seq_length=config.max_seq_length,
            config=config,
            tokenizer=tokenizer,  # Pass tokenizer here
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
        
        # Initialize device before loading checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_loss = float('inf')  # Initialize best_loss here

        # Load the latest checkpoint
        checkpoint_path = "checkpoints/checkpoint_epoch_4.pt"
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            if 'model_state_dict' in checkpoint:
                # Filter out mismatched token analyzer weights
                state_dict = checkpoint['model_state_dict']
                model_dict = model.state_dict()
                
                # Remove token_temperature keys from loaded state dict
                filtered_state_dict = {k: v for k, v in state_dict.items() 
                                        if not k.startswith('token_temperature')}
                

                
                # Load the filtered state dict
                model.load_state_dict(filtered_state_dict, strict=False)
                print("Loaded checkpoint and initialized token_temperature parameters")
                
                # Start training from the saved epoch
                start_epoch = checkpoint.get('epoch', 0)
                print(f"Resuming from epoch {start_epoch}")
                
                # Initialize optimizer before loading its state
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay
                )
                
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print("Loaded optimizer state")
                    except ValueError:
                        print("Optimizer state loading failed - initializing fresh optimizer")
                        optimizer = torch.optim.AdamW(
                            model.parameters(),
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay
                        )
            else:
                model.load_state_dict(checkpoint)
                start_epoch = 0
                best_loss = float('inf')
                print("Loaded model weights only")
        else:
            print("No checkpoint found, starting from scratch")
            start_epoch = 0
            best_loss = float('inf')
        
        model = model.to(device)
        
        # Load datasets - keep the unpacking since we're returning a tuple
        print("Loading datasets...")
        train_dataset, val_dataset = load_and_preprocess_data(
            tokenizer, 
            max_length=config.max_seq_length,
            dataset_fraction=config.dataset_fraction
        )
        
        print(f"Loaded datasets successfully:")
        print(f"Training set size: {len(train_dataset):,}")
        print(f"Validation set size: {len(val_dataset):,}")
        
        # Initialize trainer with the loaded state
        trainer = QuasarTrainer(
            model=model,
            config=config,  
            device=device,
            tokenizer=tokenizer,  # Add tokenizer here
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            start_epoch=start_epoch,
            best_loss=best_loss
        )

        # Continue training from the last checkpoint
        trainer.train(train_dataset, val_dataset)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        traceback.print_exc()
        if wandb.run is not None:
            wandb.finish(exit_code=1)
        raise e 
    

    import networkx as nx
import matplotlib.pyplot as plt

class NetworkVisualizer:
    def __init__(self, model, save_dir='visualizations'):
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def visualize_network(self, temperature=None, loss=None):
        plt.figure(figsize=(12, 8))
        
        # Get network dimensions
        input_nodes = self.model.d_model
        num_layers = len(self.model.blocks)
        
        # Create positions for nodes
        pos = {}
        layer_spacing = 1.0 / (num_layers + 1)
        
        # Default colors when no metrics available
        temp_color = plt.cm.RdYlBu(temperature) if temperature is not None else 'lightgreen'
        edge_alpha = min(1.0, max(0.2, 1.0 - (loss or 0)) if loss is not None else 0.2)
        
        # Add layers with dynamic properties
        for i in range(input_nodes):
            # Input layer
            pos[f'input_{i}'] = np.array([0, i / input_nodes])
            
            # Hidden layers
            for l in range(num_layers):
                pos[f'hidden_{l}_{i}'] = np.array([(l + 1) * layer_spacing, i / input_nodes])
            
            # Output layer
            pos[f'output_{i}'] = np.array([1.0, i / input_nodes])
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for i in range(input_nodes):
            # Add input nodes
            G.add_node(f'input_{i}', color='lightblue', size=100)
            
            # Add hidden layer nodes and connections
            prev_layer = f'input_{i}'
            for l in range(num_layers):
                current = f'hidden_{l}_{i}'
                G.add_node(current, color=temp_color, size=100)
                G.add_edge(prev_layer, current, alpha=edge_alpha)
                prev_layer = current
            
            # Add output nodes and final connections
            G.add_node(f'output_{i}', color='lightgreen', size=100)
            G.add_edge(prev_layer, f'output_{i}', alpha=edge_alpha)
        
        # Draw network
        plt.clf()
        nx.draw(G, pos=pos, 
                node_color=[G.nodes[node]['color'] for node in G.nodes],
                node_size=[G.nodes[node]['size'] for node in G.nodes],
                edge_color='gray',
                width=[G[u][v]['alpha'] for u, v in G.edges],
                with_labels=False)
        
        # Add title with metrics
        title = f"Network Architecture"
        if temperature is not None:
            title += f"\nTemperature: {temperature:.2f}"
        if loss is not None:
            title += f"\nLoss: {loss:.4f}"
        plt.title(title)
        
        # Save visualization
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        plt.savefig(os.path.join(self.save_dir, f'network_{timestamp}.png'))
        plt.close()

    def update_visualization(self, temperature=None, loss=None):
        """Update visualization with current metrics"""
        self.visualize_network(temperature, loss)

def convert_to_adapter(balanced_datasets, tokenizer, max_length):
    """Convert balanced datasets to HuggingFaceDatasetAdapter format."""
    adapted = {}
    for source, dataset in balanced_datasets.items():
        adapted[source] = HuggingFaceDatasetAdapter(
            tokenizer=tokenizer,
            max_length=max_length,
            data_generator=dataset['text']
        )   
    return adapted

def load_checkpoint(checkpoint_path, device):
    if not checkpoint_path.exists():
        print("No checkpoint found, starting from scratch")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting from scratch")
        return None

def validate_checkpoint(model, checkpoint):
    if checkpoint is None:
        return
        
    # Check if model architecture matches
    model_state = model.state_dict()
    checkpoint_state = checkpoint['model_state_dict']
    
    if model_state.keys() != checkpoint_state.keys():
        raise ValueError("Checkpoint architecture doesn't match current model")

if __name__ == "__main__":
    main()