"""
BART Summarizer implementation for MoodMap.

This module provides text summarization capabilities using BART (Bidirectional and 
Auto-Regressive Transformers) from the Hugging Face transformers library.
"""

import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("bart_summarizer")

class BartSummarizer:
    """
    A class implementing text summarization using BART from Hugging Face.
    
    This implementation loads the BART model for abstractive summarization
    of text and provides methods to generate concise, fluent summaries.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the BART summarizer with the specified model.
        
        Args:
            model_name: The name of the Hugging Face BART model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the BART model and tokenizer from Hugging Face"""
        try:
            from transformers import BartForConditionalGeneration, BartTokenizer
            
            logger.info(f"Loading BART model: {self.model_name}")
            self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
            logger.info("BART model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BART model: {str(e)}")
            raise
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50) -> Dict[str, Any]:
        """
        Generate a summary for the provided text.
        
        Args:
            text: The text to summarize
            max_length: Maximum length of the generated summary
            min_length: Minimum length of the generated summary
            
        Returns:
            A dictionary containing the summary and metadata
        """
        if not text or len(text) < 100:
            # Text is too short, return original
            return {
                "summary": text,
                "original_length": len(text),
                "summary_length": len(text),
                "method": "original_text_returned"
            }
        
        try:
            # Prepare input for the model
            inputs = self.tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
            
            # Generate summary
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode the generated summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            return {
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "method": "bart",
                "model_used": self.model_name
            }
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            
            # Fallback to a simple extractive summary
            from textwrap import shorten
            simple_summary = shorten(text, width=max_length*5, placeholder="...")
            
            return {
                "summary": simple_summary,
                "original_length": len(text),
                "summary_length": len(simple_summary),
                "method": "fallback_truncation",
                "error": str(e)
            }
    
    def __repr__(self):
        """String representation of the summarizer"""
        return f"BartSummarizer(model_name='{self.model_name}')"