import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from typing import Dict, Any
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.custom_model_path = "./model"
        self.load_model()
    
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
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None
    
    async def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for given text"""
        if not self.is_loaded():
            raise ValueError("Model not loaded")
        
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