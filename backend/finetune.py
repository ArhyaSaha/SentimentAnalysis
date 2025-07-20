#!/usr/bin/env python3
import os
import json
import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)
from torch.optim import AdamW
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType,
    prepare_model_for_kbit_training
)
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_path):
    """Load data from JSONL file"""
    texts = []
    labels = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            texts.append(data['text'])
            # Convert label to numeric
            label = 1 if data['label'].lower() == 'positive' else 0
            labels.append(label)
    
    return texts, labels

def train_model(model, train_loader, val_loader, device, epochs, learning_rate):
    """Fine-tune the model"""
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    model.train()
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Average training loss: {avg_loss:.4f}")
        
        # Validation
        if val_loader:
            val_accuracy = evaluate_model(model, val_loader, device)
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")

def evaluate_model(model, data_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            predicted_labels = torch.argmax(logits, dim=-1)
            predictions.extend(predicted_labels.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(actual_labels, predictions)
    logger.info(f"\nClassification Report:\n{classification_report(actual_labels, predictions)}")
    
    model.train()
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Fine-tune sentiment analysis model')
    parser.add_argument('command', choices=['data'], help='Command to run')
    parser.add_argument('data_path', help='Path to training data (JSONL format)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--model_name', default='cardiffnlp/twitter-roberta-base-sentiment-latest', help='Base model name')
    parser.add_argument('--output_dir', default='./model', help='Output directory for fine-tuned model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA for parameter-efficient fine-tuning')
    parser.add_argument('--use_quantization', action='store_true', help='Use 4-bit quantization')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    
    args = parser.parse_args()

    # Adjust batch size and other parameters for CPU if needed
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        # Reduce batch size for CPU to avoid memory issues
        if args.batch_size > 8:
            logger.info(f"Reducing batch size from {args.batch_size} to 8 for CPU training")
            args.batch_size = 8
        # Reduce max length for CPU
        if args.max_length > 256:
            logger.info(f"Reducing max_length from {args.max_length} to 256 for CPU training")
            args.max_length = 256

    set_seed(args.seed)
    logger.info(f"Using device: {device}")
    
    logger.info(f"Loading data from {args.data_path}")
    texts, labels = load_data(args.data_path)
    logger.info(f"Loaded {len(texts)} examples")
    
    split_idx = int(len(texts) * (1 - args.val_split))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    # Quantization Config
    bnb_config = None
    if args.use_quantization:
        logger.info("Setting up 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with proper configuration
    model_kwargs = {
        'num_labels': 2,
        'ignore_mismatched_sizes': True,
    }
    
    # Only add quantization config if using GPU
    if args.use_quantization and torch.cuda.is_available():
        model_kwargs.update({
            'quantization_config': bnb_config,
            'device_map': "auto",
            'torch_dtype': torch.float16,
        })
        logger.info("Using GPU with quantization")
    elif args.use_quantization and not torch.cuda.is_available():
        logger.warning("Quantization requested but no GPU available. Falling back to regular loading.")
        args.use_quantization = False  # Disable quantization for CPU
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        **model_kwargs
    )
    
    # Prepare model for quantization if using GPU
    if args.use_quantization and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)
    
    # LoRA Config
    if args.use_lora:
        logger.info("Setting up LoRA configuration...")
        
        # Updated target modules for RoBERTa
        target_modules = ["query", "value", "key", "dense"]
        
        # If using quantization, add modules_to_save to handle classifier
        modules_to_save = None
        if args.use_quantization:
            modules_to_save = ["classifier"]
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        
        try:
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except Exception as e:
            logger.error(f"Error setting up LoRA: {e}")
            logger.info("Falling back to regular fine-tuning...")
            args.use_lora = False
    
    # Move model to device if not using quantization
    if not (args.use_quantization and torch.cuda.is_available()):
        model.to(device)
    
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_length=args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Train model
    logger.info("Starting training...")
    train_model(model, train_loader, val_loader, device, args.epochs, args.lr)
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.use_lora:
        # Save LoRA adapters
        model.save_pretrained(args.output_dir)
        # Save base model config and tokenizer
        model.base_model.model.config.save_pretrained(args.output_dir)
    else:
        model.save_pretrained(args.output_dir)
    
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Fine-tuning completed!")

if __name__ == "__main__":
    main()