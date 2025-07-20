import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from typing import Dict, Any
import asyncio
import logging
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelReloadHandler(FileSystemEventHandler):
    """Handles file system events for model directory"""
    
    def __init__(self, model_handler):
        self.model_handler = model_handler
        self.reload_lock = threading.Lock()
        self.last_reload_time = 0
        self.reload_delay = 2  # Wait 2 seconds before reloading to avoid multiple triggers
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.bin', '.json', '.txt')):
            self._schedule_reload()
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.bin', '.json', '.txt')):
            self._schedule_reload()
    
    def _schedule_reload(self):
        """Schedule a reload with debouncing to avoid multiple rapid reloads"""
        current_time = time.time()
        if current_time - self.last_reload_time > self.reload_delay:
            self.last_reload_time = current_time
            threading.Thread(target=self._delayed_reload, daemon=True).start()
    
    def _delayed_reload(self):
        """Perform the actual reload after a delay"""
        time.sleep(self.reload_delay)
        with self.reload_lock:
            try:
                logger.info("Detected model file changes, reloading model...")
                self.model_handler.reload_model()
                logger.info("Model reloaded successfully")
            except Exception as e:
                logger.error(f"Error reloading model: {e}")

class ModelHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.custom_model_path = "./model"
        self.reload_handler = None
        self.observer = None
        self.is_reloading = False
        self.model_lock = threading.Lock()
        
        self.load_model()
        self.setup_file_watcher()
    
    def load_model(self):
        """Load model from custom path or default HuggingFace model"""
        try:
            if os.path.exists(self.custom_model_path) and os.listdir(self.custom_model_path):
                # Check if it's a LoRA adapter by looking for adapter_config.json
                adapter_config_path = os.path.join(self.custom_model_path, "adapter_config.json")
                
                if os.path.exists(adapter_config_path):
                    logger.info(f"Loading LoRA adapter from {self.custom_model_path}")
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name,
                        num_labels=2,
                        ignore_mismatched_sizes=True
                    )
                    
                    # Load LoRA adapter
                    self.model = PeftModel.from_pretrained(base_model, self.custom_model_path)
                    self.tokenizer = AutoTokenizer.from_pretrained(self.custom_model_path)
                else:
                    logger.info(f"Loading fine-tuned model from {self.custom_model_path}")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.custom_model_path)
                    
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.custom_model_path,
                        num_labels=2,  
                        ignore_mismatched_sizes=True  
                    )
            else:
                logger.info(f"Loading default model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=2,
                    ignore_mismatched_sizes=True
                )
            
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def setup_file_watcher(self):
        """Set up file system watcher for the model directory"""
        try:
            if os.path.exists(self.custom_model_path):
                self.reload_handler = ModelReloadHandler(self)
                self.observer = Observer()
                self.observer.schedule(self.reload_handler, self.custom_model_path, recursive=True)
                self.observer.start()
                logger.info(f"File watcher started for {self.custom_model_path}")
            else:
                logger.info(f"Model directory {self.custom_model_path} does not exist, file watcher not started")
        except Exception as e:
            logger.error(f"Error setting up file watcher: {e}")
    
    def reload_model(self):
        """Reload the model (called by file watcher)"""
        with self.model_lock:
            self.is_reloading = True
            try:
                old_model = self.model
                old_tokenizer = self.tokenizer
                
                # Load new model
                self.load_model()
                
                # Clean up old model
                if old_model:
                    del old_model
                if old_tokenizer:
                    del old_tokenizer
                
                # Force garbage collection
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            finally:
                self.is_reloading = False
    
    def stop_file_watcher(self):
        """Stop the file watcher"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("File watcher stopped")
    
    def is_reload_in_progress(self) -> bool:
        """Check if a reload is currently in progress"""
        return self.is_reloading
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None
    
    async def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for given text"""
        if not self.is_loaded():
            raise ValueError("Model not loaded")
        
        # Wait if reload is in progress
        while self.is_reloading:
            await asyncio.sleep(0.1)
        
        with self.model_lock:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._predict_sync, text)
            return result
    
    def _predict_sync(self, text: str) -> Dict[str, Any]:
        """Synchronous prediction method"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Map to positive/negative
            # For cardiffnlp/twitter-roberta-base-sentiment-latest:
            # LABEL_0: NEGATIVE, LABEL_1: NEUTRAL, LABEL_2: POSITIVE
            if self.model.config.num_labels == 3:
                if predicted_class == 0:
                    label = "negative"
                    score = confidence
                elif predicted_class == 2:
                    label = "positive"
                    score = confidence
                else:  
                    label = "positive"
                    score = 0.5
            else:
                label = "positive" if predicted_class == 1 else "negative"
                score = confidence
            
            return {
                "label": label,
                "score": float(score)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def __del__(self):
        """Cleanup when the object is destroyed"""
        self.stop_file_watcher()