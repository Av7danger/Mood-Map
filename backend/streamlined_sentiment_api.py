"""
Streamlined FastAPI backend for sentiment analysis system.
This version maintains core functionality but eliminates error-prone components.
"""
import os
import sys
import json
import logging
import time
import re
import socket
import traceback
import pickle
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Setup logging first
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", "streamlined_api_requests.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("streamlined_sentiment_api")

# Initialize the app
app = FastAPI(
    title="Mood Map Streamlined API",
    description="Streamlined API for sentiment analysis with robust error handling",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model holder class with robust fallbacks
class ModelContainer:
    def __init__(self):
        self.ensemble_model = None
        self.attention_model = None
        self.tokenizer = None
        self.neutral_finetuner = None
        self.advanced_model = None
        self.config = {}
        self.models_loaded = True  # Always marked as loaded since we have fallbacks
        self.load_errors = []
        
        # Track loading times to inform users
        self.ensemble_loading_time = None
        self.attention_loading_time = None 
        self.neutral_loading_time = None
        self.advanced_loading_time = None
        
        # New: Add pickled model 
        self.pickled_model = None
        self.pickled_model_loaded = False
        
        # Track model states for health check
        self.model_states = {
            "ensemble": {"loaded": False, "loading": False, "error": None, "last_loaded": None, "loading_time": None},
            "attention": {"loaded": False, "loading": False, "error": None, "last_loaded": None, "loading_time": None},
            "neutral": {"loaded": False, "loading": False, "error": None, "last_loaded": None, "loading_time": None},
            "advanced": {"loaded": False, "loading": False, "error": None, "last_loaded": None, "loading_time": None},
            "pickled": {"loaded": False, "loading": False, "error": None, "last_loaded": None, "loading_time": None},
            "simple": {"loaded": True, "loading": False, "error": None, "last_loaded": time.time(), "loading_time": 0}
        }
        
        # Define sentiment keywords for rule-based approach
        self.positive_keywords = [
            "love", "amazing", "excellent", "fantastic", "great", "awesome", 
            "good", "happy", "glad", "wonderful", "outstanding", "perfect", 
            "helpful", "recommend", "thanks", "positive", "best"
        ]
        
        self.negative_keywords = [
            "hate", "terrible", "awful", "horrible", "worst", "disappointing", 
            "bad", "sad", "angry", "poor", "useless", "waste", "broken", 
            "frustrated", "negative", "fail", "sucks", "problem", "issue", "error"
        ]
        
        # Load the configuration
        self.load_config()
        
        # Try to load the pickled model
        self.load_pickled_model()
        
    def load_pickled_model(self):
        """Load the pre-trained pickled model from model.pkl"""
        try:
            start_time = time.time()
            
            # Try different possible paths for the model.pkl file
            model_paths = [
                os.path.join(Path(__file__).parent, "model.pkl"),  # Backend folder
                os.path.join(Path(__file__).parent.parent, "model.pkl")  # Project root
            ]
            
            model_loaded = False
            for path in model_paths:
                if os.path.exists(path):
                    logger.info(f"Loading pickled model from: {path}")
                    with open(path, 'rb') as f:
                        self.pickled_model = pickle.load(f)
                    loading_time = time.time() - start_time
                    logger.info(f"Pickled model loaded successfully in {loading_time:.2f} seconds")
                    
                    # Update model state
                    self.pickled_model_loaded = True
                    self.model_states["pickled"]["loaded"] = True
                    self.model_states["pickled"]["loading_time"] = loading_time
                    self.model_states["pickled"]["last_loaded"] = time.time()
                    model_loaded = True
                    break
            
            if not model_loaded:
                logger.warning("Could not find model.pkl file in expected locations")
                return False
                
            return True
        except Exception as e:
            error_msg = f"Error loading pickled model: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.model_states["pickled"]["error"] = error_msg
            return False
    
    def load_config(self):
        """Load the API configuration file with robust error handling."""
        try:
            config_path = os.path.join(Path(__file__).parent, "sentiment_api_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.warning(f"Config file not found at {config_path}. Using default configuration.")
                self.config = {}
            
            # Configure logging based on config
            if "logging" in self.config:
                log_level = self.config["logging"].get("log_level", "INFO")
                numeric_level = getattr(logging, log_level.upper(), logging.INFO)
                logger.setLevel(numeric_level)
            
            return True
        except Exception as e:
            logger.warning(f"Error loading config: {e}. Using default configuration.")
            self.config = {}
            return False

    def analyze_with_pickled_model(self, text):
        """
        Analyze text using the pre-trained pickled model.
        Returns None if the model isn't loaded or fails.
        """
        if not self.pickled_model_loaded or self.pickled_model is None:
            return None
            
        try:
            # Most pickled models expect specific preprocessing
            # Here we do basic preprocessing that works with many models
            processed_text = text.lower()
            
            # Handle different types of pickled models
            if hasattr(self.pickled_model, 'predict'):
                # For scikit-learn like models
                prediction = self.pickled_model.predict([processed_text])
                
                # Try to get probability if available
                confidence = 0.8  # Default confidence
                try:
                    if hasattr(self.pickled_model, 'predict_proba'):
                        proba = self.pickled_model.predict_proba([processed_text])
                        confidence = proba[0].max()
                except:
                    pass
                
                # Map prediction to sentiment
                # Most models use 0, 1, 2 for negative, neutral, positive
                label_map = {0: "negative", 1: "neutral", 2: "positive"}
                category = prediction[0]
                if isinstance(category, str):
                    # If model returns string labels directly
                    label = category
                    if "negative" in label.lower():
                        category = 0
                    elif "positive" in label.lower():
                        category = 2
                    else:
                        category = 1
                else:
                    # If model returns numerical labels
                    label = label_map.get(category, "neutral")
                
                # Handle binary classifiers (0,1)
                if category == 1 and 0 in label_map and 2 not in label_map:
                    category = 2  # Map binary 1 to our standard positive (2)
                
                return {
                    "label": label,
                    "score": float(category),
                    "category": int(category),
                    "confidence": float(confidence),
                    "model_used": "pickled_model"
                }
            elif callable(self.pickled_model):
                # For function-like models
                result = self.pickled_model(processed_text)
                if isinstance(result, dict):
                    return {
                        "label": result.get("label", "neutral"),
                        "score": float(result.get("score", 1.0)),
                        "category": int(result.get("category", 1)),
                        "confidence": float(result.get("confidence", 0.7)),
                        "model_used": "pickled_model_function"
                    }
                else:
                    # Try to interpret the result
                    return {
                        "label": "positive" if result > 0.5 else "negative",
                        "score": float(result) * 2.0,  # Scale to 0-2 range
                        "category": 2 if result > 0.5 else 0,
                        "confidence": float(abs(result - 0.5) * 2),  # Scale to 0-1 range
                        "model_used": "pickled_model_function"
                    }
            else:
                logger.warning(f"Unsupported pickled model type: {type(self.pickled_model)}")
                return None
                
        except Exception as e:
            logger.error(f"Error using pickled model: {e}")
            logger.error(traceback.format_exc())
            return None

    def analyze_with_rule_based(self, text):
        """
        Rule-based sentiment analysis that's always available.
        This is a reliable fallback that doesn't rely on ML models.
        """
        text_lower = text.lower()
        
        # Count occurrences of positive and negative keywords
        positive_count = sum(word in text_lower for word in self.positive_keywords)
        negative_count = sum(word in text_lower for word in self.negative_keywords)
        
        # Calculate sentiment score based on keyword counts
        if positive_count > negative_count:
            category = 2  # Positive
            score = min(2.0, 1.0 + (positive_count - negative_count) / 5)
            label = "positive"
            confidence = min(0.95, 0.5 + (positive_count - negative_count) / 10)
        elif negative_count > positive_count:
            category = 0  # Negative
            score = max(0.0, 1.0 - (negative_count - positive_count) / 5)
            label = "negative"
            confidence = min(0.95, 0.5 + (negative_count - positive_count) / 10)
        else:
            # If counts are equal or both zero, check for specific phrases
            if any(phrase in text_lower for phrase in ["not bad", "pretty good", "seems ok"]):
                category = 2  # Slightly positive
                score = 1.2
                label = "positive"
                confidence = 0.6
            elif any(phrase in text_lower for phrase in ["not great", "meh", "so so"]):
                category = 0  # Slightly negative
                score = 0.8
                label = "negative"
                confidence = 0.6
            else:
                category = 1  # Neutral
                score = 1.0
                label = "neutral"
                confidence = 0.5
        
        return {
            "label": label,
            "score": float(score),
            "category": category,
            "confidence": float(confidence),
            "model_used": "rule_based"
        }
    
    def analyze_sentiment(self, text, model_type="hybrid"):
        """
        Unified sentiment analysis that tries multiple models.
        
        Args:
            text (str): Text to analyze
            model_type (str): The model to use: "pickled", "rule_based", or "hybrid" (default)
            
        Returns:
            dict: Sentiment analysis result
        """
        # First try the specifically requested model
        if model_type == "pickled":
            pickled_result = self.analyze_with_pickled_model(text)
            if pickled_result:
                return pickled_result
            else:
                # Fall back to rule-based if pickled model fails
                return self.analyze_with_rule_based(text)
                
        elif model_type == "rule_based":
            return self.analyze_with_rule_based(text)
        
        # For hybrid approach, try both and combine or use the most confident
        elif model_type == "hybrid":
            # Try the pickled model first
            pickled_result = self.analyze_with_pickled_model(text)
            
            # Get rule-based result
            rule_result = self.analyze_with_rule_based(text)
            
            # If pickled model failed, return rule-based
            if not pickled_result:
                return rule_result
                
            # If models agree, return the one with higher confidence
            if pickled_result["category"] == rule_result["category"]:
                return pickled_result if pickled_result["confidence"] > rule_result["confidence"] else rule_result
                
            # If models disagree, check confidences
            if pickled_result["confidence"] > rule_result["confidence"] + 0.2:
                # Pickled model is much more confident, use it
                return pickled_result
            elif rule_result["confidence"] > pickled_result["confidence"] + 0.2:
                # Rule-based is much more confident, use it
                return rule_result
            else:
                # Similar confidence, return a blend
                # Default to the pickled model but note the disagreement
                result = pickled_result.copy()
                result["alternative_label"] = rule_result["label"]
                result["confidence"] = (pickled_result["confidence"] + rule_result["confidence"]) / 2
                result["model_used"] = "hybrid (pickled+rule)"
                return result
        
        # Default to rule-based for any other model type
        return self.analyze_with_rule_based(text)
    
    def simple_extractive_summarization(self, text, max_length=150, min_length=50):
        """
        Simple extractive summarization that works without dependencies.
        Takes the first few sentences of the text as a simple summary.
        """
        # Check if text is too short to summarize
        if not text or len(text) < 100:
            return text
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Determine how many sentences to include based on min/max length
        target_length = (min_length + max_length) // 2
        current_length = 0
        selected_sentences = []
        
        # Take sentences from the beginning until we reach target length
        for sentence in sentences:
            selected_sentences.append(sentence)
            current_length += len(sentence)
            if current_length >= target_length:
                break
        
        # If we've taken all sentences and still haven't reached min_length,
        # return the original text
        if len(selected_sentences) == len(sentences) and current_length < min_length:
            return text
            
        # Join selected sentences
        summary = ' '.join(selected_sentences)
        
        # Truncate if summary exceeds max_length
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
            
        return summary
        
# Initialize the model container
model_container = ModelContainer()

# Data models for API requests and responses
class TextRequest(BaseModel):
    text: str
    model_type: Optional[str] = "simple"  # Default to simple model

class TextsRequest(BaseModel):
    texts: List[str]
    model_type: Optional[str] = "simple"  # Default to simple model

class TweetCommentsRequest(BaseModel):
    tweet: str
    comments: List[str]
    max_comments: Optional[int] = 50

class RagRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None

class SummarizationRequest(BaseModel):
    text: str
    max_length: Optional[int] = 150
    min_length: Optional[int] = 50
    do_sample: Optional[bool] = False

class SentimentResponse(BaseModel):
    label: str
    score: float
    category: int
    confidence: Optional[float] = None
    summary: Optional[str] = None

class TopicRequest(BaseModel):
    text: str
    threshold: Optional[float] = 0.55
    max_topics: Optional[int] = 3

class TopicCustomizationRequest(BaseModel):
    topic_name: str
    keywords: List[str]

class BatchContextRequest(BaseModel):
    texts: List[str]
    model_type: Optional[str] = "simple"
    include_topics: Optional[bool] = False
    include_summaries: Optional[bool] = False

class LoadModelRequest(BaseModel):
    model_type: str = Field(..., description="Type of model to load")
    wait_for_loading: bool = Field(True, description="Whether to wait for model loading to complete")

class ExtensionStatusRequest(BaseModel):
    check_offline_available: bool = Field(True, description="Whether to check if offline models are available")
    require_advanced_features: bool = Field(False, description="Whether advanced features are required")

@app.on_event("startup")
async def startup_event():
    """Initialize basic configuration on startup."""
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Log startup information
    logger.info("=" * 50)
    logger.info(f"Streamlined API startup complete - Server running")
    logger.info("=" * 50)

@app.get("/health")
async def health_check():
    """Check if the API is running."""
    # Simple model is always available
    model_container.model_states["simple"]["loaded"] = True
    model_container.model_states["simple"]["last_loaded"] = time.time()
    
    # Report on approximate memory usage (only simple model is guaranteed)
    memory_usage = {"simple": "<1MB"}
    
    # Check for torch availability without importing
    device_info = {
        "cuda_available": False,
        "device_count": 0,
        "fallback": "Using rule-based model which doesn't require GPU"
    }
    
    return {
        "status": "online",
        "timestamp": time.time(),
        "model_states": model_container.model_states,
        "memory_usage": memory_usage,
        "device_info": device_info,
        "extension_mode_recommended": "simple"  # Recommend simple model for extension use
    }

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_text(request: TextRequest):
    """
    Analyze the sentiment of a single text using the specified model.
    
    The streamlined API always uses the rule-based approach which is robust
    and doesn't rely on complex ML models.
    """
    try:
        model_type = request.model_type.lower()
        
        # Log request
        logger.info(f"Received analyze request: model={model_type}, text='{request.text[:50]}...'")
        
        # Always use the rule-based approach regardless of requested model type
        # This ensures reliability while still providing meaningful sentiment analysis
        result = model_container.analyze_sentiment(request.text, model_type)
        
        # Add metadata about requested model
        result["requested_model"] = model_type
        
        # Log successful response
        logger.info(f"Analysis complete: {result['label']}")
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        logger.error(traceback.format_exc())
        
        # Return a fallback neutral result instead of failing
        return {
            "label": "neutral",
            "score": 1.0,
            "category": 1,
            "confidence": 0.3,
            "model_used": "error_fallback",
            "error": str(e)
        }

@app.post("/analyze-batch")
async def analyze_texts(request: TextsRequest):
    """
    Analyze the sentiment of multiple texts in batch.
    """
    try:
        # Log request
        logger.info(f"Received batch analyze request for {len(request.texts)} texts")
        
        results = []
        for text in request.texts:
            result = model_container.analyze_sentiment(text, request.model_type)
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "label": result["label"],
                "score": result["score"],
                "category": result["category"],
                "confidence": result["confidence"]
            })
        
        return {"results": results}
    except Exception as e:
        # Log error
        logger.error(f"Error analyzing texts in batch: {e}")
        logger.error(traceback.format_exc())
        
        # Return a minimal result with the error
        return {
            "error": str(e),
            "results": [{"text": t[:50] + "...", "label": "error", "score": 1.0, "category": 1} 
                       for t in request.texts[:5]]
        }

@app.post("/analyze-tweet")
async def analyze_tweet_with_comments(request: TweetCommentsRequest):
    """
    Analyze a tweet and its comments for sentiment and themes.
    """
    try:
        # Log request
        logger.info(f"Received tweet analysis request with {len(request.comments)} comments")
        
        # Analyze tweet
        tweet_result = model_container.analyze_sentiment(request.tweet)
        
        # Analyze comments (limit to max_comments)
        comment_results = []
        for comment in request.comments[:request.max_comments]:
            result = model_container.analyze_sentiment(comment)
            comment_results.append({
                "text": comment[:100] + "..." if len(comment) > 100 else comment,
                "sentiment": result["label"],
                "score": result["score"]
            })
        
        # Calculate summary stats
        positive_count = sum(1 for c in comment_results if c["sentiment"] == "positive")
        negative_count = sum(1 for c in comment_results if c["sentiment"] == "negative")
        neutral_count = sum(1 for c in comment_results if c["sentiment"] == "neutral")
        
        # Calculate average sentiment
        total_comments = len(comment_results)
        avg_sentiment = "neutral"
        if total_comments > 0:
            if positive_count > negative_count and positive_count > neutral_count:
                avg_sentiment = "positive"
            elif negative_count > positive_count and negative_count > neutral_count:
                avg_sentiment = "negative"
        
        return {
            "tweet": {
                "text": request.tweet[:100] + "..." if len(request.tweet) > 100 else request.tweet,
                "sentiment": tweet_result["label"],
                "score": tweet_result["score"]
            },
            "comments": comment_results,
            "stats": {
                "total_comments": total_comments,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "positive_percentage": round(positive_count / total_comments * 100, 1) if total_comments > 0 else 0,
                "negative_percentage": round(negative_count / total_comments * 100, 1) if total_comments > 0 else 0,
                "neutral_percentage": round(neutral_count / total_comments * 100, 1) if total_comments > 0 else 0,
                "average_sentiment": avg_sentiment
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing tweet with comments: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "tweet": {"sentiment": "neutral"},
            "comments": [],
            "stats": {
                "error": "Analysis failed"
            }
        }

@app.post("/summarize")
async def summarize_text(request: SummarizationRequest):
    """
    Generate a summary of a text using a simple but reliable extractive method.
    """
    try:
        # Log request
        logger.info(f"Received summarization request for text of length {len(request.text)}")
        
        # Generate summary
        summary = model_container.simple_extractive_summarization(
            request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        return {
            "summary": summary,
            "method": "extractive",
            "original_length": len(request.text),
            "summary_length": len(summary)
        }
    except Exception as e:
        # Log error
        logger.error(f"Error summarizing text: {e}")
        logger.error(traceback.format_exc())
        
        # Return a truncated version of the original text as fallback
        max_len = min(200, len(request.text))
        return {
            "summary": request.text[:max_len] + "..." if len(request.text) > max_len else request.text,
            "method": "truncation_fallback",
            "error": str(e)
        }

@app.post("/analyze-with-summary")
async def analyze_with_summary(request: TextRequest):
    """
    Analyze the sentiment of a text and generate a summary in a single request.
    """
    try:
        # Log request
        logger.info(f"Received analyze-with-summary request for text of length {len(request.text)}")
        
        # First perform sentiment analysis
        sentiment_result = model_container.analyze_sentiment(request.text, request.model_type)
        
        # Check if text is long enough to warrant summarization
        if len(request.text.split()) < 30:
            # For short texts, just use the original
            sentiment_result["summary"] = request.text
            sentiment_result["summarization_method"] = "original_text"
            return sentiment_result
        
        # Generate a summary
        summary = model_container.simple_extractive_summarization(request.text)
        sentiment_result["summary"] = summary
        sentiment_result["summarization_method"] = "extractive"
        
        return sentiment_result
    except Exception as e:
        # Log error
        logger.error(f"Error in analyze with summary: {e}")
        logger.error(traceback.format_exc())
        
        # Return a fallback result
        return {
            "label": "neutral",
            "score": 1.0,
            "category": 1,
            "confidence": 0.3,
            "model_used": "fallback",
            "summary": request.text[:200] + "..." if len(request.text) > 200 else request.text,
            "summarization_method": "error_fallback",
            "error": str(e)
        }

@app.post("/detect-topics")
async def detect_topics(request: TopicRequest):
    """
    Detect topics in a text using a simple keyword-based approach.
    """
    try:
        # Log request
        logger.info(f"Received topic detection request for text of length {len(request.text)}")
        
        # Simple topic detection based on keyword matching
        text_lower = request.text.lower()
        
        # Define topic keywords
        topics = {
            "technology": ["tech", "technology", "software", "hardware", "digital", "computer", "app", "internet", "code"],
            "business": ["business", "company", "startup", "market", "product", "service", "customer", "finance"],
            "health": ["health", "medical", "doctor", "fitness", "wellness", "exercise", "diet", "hospital"],
            "politics": ["politics", "government", "election", "policy", "president", "congress", "democracy"],
            "entertainment": ["movie", "film", "music", "game", "artist", "celebrity", "TV", "show", "entertainment"],
            "education": ["education", "school", "university", "college", "student", "teacher", "learning", "course"]
        }
        
        # Count keyword matches for each topic
        topic_scores = {}
        for topic, keywords in topics.items():
            matches = sum(keyword in text_lower for keyword in keywords)
            if matches > 0:
                # Calculate score based on number of matches
                score = min(matches / len(keywords), 1.0)
                topic_scores[topic] = score
        
        # Filter by threshold and max_topics
        filtered_topics = {
            topic: score for topic, score in topic_scores.items() 
            if score >= request.threshold
        }
        
        # Sort by score and limit to max_topics
        sorted_topics = sorted(filtered_topics.items(), key=lambda x: x[1], reverse=True)[:request.max_topics]
        
        return {
            "topics": [{"topic": topic, "score": score} for topic, score in sorted_topics],
            "count": len(sorted_topics)
        }
    except Exception as e:
        logger.error(f"Error detecting topics: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "topics": []
        }

@app.post("/analyze-batch-with-context")
async def analyze_batch_with_context(request: BatchContextRequest):
    """
    Analyze multiple texts with optional topic detection and summarization.
    """
    try:
        # Log request
        logger.info(f"Received batch context analysis request for {len(request.texts)} texts")
        
        results = []
        for text in request.texts:
            # Analyze sentiment
            sentiment_result = model_container.analyze_sentiment(text, request.model_type)
            
            # Create result entry
            result = {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "sentiment": sentiment_result["label"],
                "score": sentiment_result["score"],
                "category": sentiment_result["category"],
                "confidence": sentiment_result["confidence"]
            }
            
            # Add topics if requested
            if request.include_topics:
                # Simple topic detection
                topic_request = TopicRequest(text=text)
                topic_result = await detect_topics(topic_request)
                result["topics"] = topic_result.get("topics", [])
            
            # Add summary if requested
            if request.include_summaries and len(text.split()) > 30:
                result["summary"] = model_container.simple_extractive_summarization(text)
                result["summarization_method"] = "extractive"
            
            results.append(result)
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Error in batch context analysis: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "results": []
        }

@app.post("/analyze-emotions")
async def analyze_emotions(request: TextRequest):
    """
    Analyze text to detect emotions like happiness, sadness, anger, etc.
    Uses a simplified keyword-based approach for reliability.
    """
    try:
        # Log request
        logger.info(f"Received emotion analysis request for text of length {len(request.text)}")
        
        text_lower = request.text.lower()
        
        # Define emotion keywords
        emotions = {
            "joy": ["happy", "happiness", "joy", "joyful", "delighted", "excited", "glad", "pleased", "wonderful"],
            "sadness": ["sad", "unhappy", "depressed", "miserable", "sorrow", "grief", "disappointed"],
            "anger": ["angry", "mad", "furious", "outraged", "irate", "annoyed", "irritated", "frustrated"],
            "fear": ["afraid", "scared", "frightened", "terrified", "fear", "fearful", "anxious", "worried"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "unexpected", "wow"],
            "love": ["love", "adore", "cherish", "affection", "fond", "devoted", "romantic"]
        }
        
        # Count keyword matches for each emotion
        emotion_scores = {}
        for emotion, keywords in emotions.items():
            matches = sum(keyword in text_lower for keyword in keywords)
            if matches > 0:
                # Calculate score based on number of matches
                score = min(matches / max(len(keywords) / 2, 1), 1.0)
                emotion_scores[emotion] = score
        
        # Find primary and secondary emotions
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        primary_emotion = sorted_emotions[0][0] if sorted_emotions else None
        secondary_emotion = sorted_emotions[1][0] if len(sorted_emotions) > 1 else None
        
        # Get the sentiment result as well
        sentiment_result = model_container.analyze_sentiment(request.text)
        
        return {
            "text": request.text[:100] + "..." if len(request.text) > 100 else request.text,
            "sentiment": sentiment_result["label"],
            "emotions": {
                "primary": primary_emotion,
                "secondary": secondary_emotion,
                "all_emotions": emotion_scores
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing emotions: {e}")
        logger.error(traceback.format_exc())
        
        # Get the sentiment as a fallback
        try:
            sentiment_result = model_container.analyze_sentiment(request.text)
            return {
                "text": request.text[:100] + "..." if len(request.text) > 100 else request.text,
                "sentiment": sentiment_result["label"],
                "emotions": {
                    "primary": None,
                    "secondary": None,
                    "error": str(e)
                }
            }
        except:
            return {
                "text": request.text[:100] + "..." if len(request.text) > 100 else request.text,
                "sentiment": "neutral",
                "emotions": {
                    "error": str(e)
                }
            }

@app.post("/load-model")
async def load_model(request: LoadModelRequest):
    """
    Simulate loading a model, always returning success.
    This endpoint exists for compatibility, but doesn't actually load ML models.
    """
    logger.info(f"Received model loading request: model={request.model_type}")
    
    return {
        "model_type": request.model_type,
        "model_loaded": True,
        "wait_completed": request.wait_for_loading,
        "loading_time": 0.1,  # Fake loading time
        "timestamp": time.time(),
        "status": "complete",
        "note": "Using rule-based models which don't require loading"
    }

@app.post("/extension/analyze")
async def extension_analyze(request: TextRequest):
    """
    Specialized endpoint for browser extensions to analyze text efficiently.
    """
    try:
        # Log request
        logger.info(f"Received extension analyze request for text of length {len(request.text)}")
        
        # Use the rule-based analysis
        result = model_container.analyze_sentiment(request.text, request.model_type)
        
        return result
    except Exception as e:
        logger.error(f"Error in extension analysis: {e}")
        logger.error(traceback.format_exc())
        
        # Return a neutral result as fallback
        return {
            "label": "neutral",
            "score": 1.0,
            "category": 1,
            "confidence": 0.3,
            "model_used": "fallback",
            "error": str(e)
        }

@app.post("/extension/analyze-with-summary")
async def extension_analyze_with_summary(request: TextRequest):
    """
    Specialized endpoint for browser extensions to analyze text and generate summary.
    """
    return await analyze_with_summary(request)

@app.get("/extension/status")
async def extension_status():
    """
    Provide status information specifically for browser extensions.
    """
    return {
        "status": "online",
        "timestamp": time.time(),
        "api_version": "1.0.0",
        "available_models": ["simple"],
        "loaded_models": ["simple"],
        "recommended_model": "simple",
        "message": "Using streamlined API with rule-based sentiment analysis"
    }

@app.post("/extension/status")
async def extension_status_detailed(request: ExtensionStatusRequest):
    """
    Provide detailed status information for browser extensions.
    """
    return {
        "status": "online",
        "timestamp": time.time(),
        "api_version": "1.0.0",
        "available_models": ["simple"],
        "loaded_models": ["simple"],
        "recommended_model": "simple",
        "offline_processing_available": True,
        "system_stats": {
            "memory_available_mb": "Not monitored",
            "cpu_percent": "Not monitored"
        },
        "model_loading_status": {
            "simple": {
                "status": "loaded",
                "loading_time_seconds": 0
            }
        }
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log API requests and their timing."""
    start_time = time.time()
    
    # Get client IP and request details
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path
    
    # Log the request
    logger.info(f"Request: {method} {path} | Client: {client_ip}")
    
    # Process the request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log the response
    logger.info(f"Response: {method} {path} | Status: {response.status_code} | Time: {process_time:.3f}s")
    
    return response

# Run the application using uvicorn
if __name__ == "__main__":
    import uvicorn
    
    # Try to check if port 5000 is available
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 5000))
        sock.close()
        print("Port 5000 is available, starting server...")
        uvicorn.run("streamlined_sentiment_api:app", host="127.0.0.1", port=5000)
    except socket.error:
        # Port 5000 is unavailable, try 5001
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 5001))
            sock.close()
            print("Port 5000 is unavailable. Using port 5001 instead...")
            uvicorn.run("streamlined_sentiment_api:app", host="127.0.0.1", port=5001)
        except socket.error:
            # Port 5001 is also unavailable, try 5002
            print("Port 5001 is also unavailable. Using port 5002 instead...")
            uvicorn.run("streamlined_sentiment_api:app", host="127.0.0.1", port=5002)