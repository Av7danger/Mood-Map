"""
FastAPI backend for the enhanced sentiment analysis system 
with integrated ensemble, attention, and neutral fine-tuning models.
With lazy loading for models and improved error handling.
"""
import os
import sys
import json
import time
import logging
import traceback
import re
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

# Try importing torch - handle gracefully if not available
try:
    import torch
except ImportError:
    pass

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Setup logging first
from src.utils.logging_utils import setup_logging
# Use os.path.join for cross-platform path handling
log_path = os.path.join("logs", "api_requests.log")
logger = setup_logging(log_path)

# Initialize the app
app = FastAPI(
    title="Mood Map Enhanced API",
    description="API for enhanced sentiment analysis with ensemble, attention, and neutral fine-tuning models",
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

@app.get("/")
async def root():
    """Root endpoint that provides information about the API and available endpoints."""
    available_endpoints = []
    
    # Gather all API endpoints
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            methods = ", ".join(route.methods)
            path = route.path
            
            # Get endpoint description from route handler if available
            description = None
            if hasattr(route, "endpoint") and hasattr(route.endpoint, "__doc__") and route.endpoint.__doc__:
                description = route.endpoint.__doc__.strip().split("\n")[0]
            
            available_endpoints.append({
                "path": path,
                "methods": methods,
                "description": description
            })
    
    return {
        "name": "Mood Map Sentiment Analysis API",
        "description": "API for enhanced sentiment analysis with ensemble, attention, and neutral fine-tuning models",
        "version": "1.0.0",
        "status": "online",
        "timestamp": time.time(),
        "endpoints": available_endpoints,
        "documentation_url": "/docs",  # Link to the automatic FastAPI docs
        "models": {
            "loaded": [model for model in model_container.model_states if model_container.model_states[model]["loaded"]],
            "available": list(model_container.model_states.keys())
        }
    }

# Define model holder class to handle lazy loading
class ModelContainer:
    def __init__(self):
        self.ensemble_model = None
        self.attention_model = None
        self.tokenizer = None
        self.neutral_finetuner = None
        self.advanced_model = None
        self.config = {}
        self.models_loaded = False
        self.load_errors = []
        
        # Track loading times to inform users
        self.ensemble_loading_time = None
        self.attention_loading_time = None 
        self.neutral_loading_time = None
        self.advanced_loading_time = None
        
        # New: Track model states for health check
        self.model_states = {
            "ensemble": {"loaded": False, "loading": False, "error": None, "last_loaded": None, "loading_time": None},
            "attention": {"loaded": False, "loading": False, "error": None, "last_loaded": None, "loading_time": None},
            "neutral": {"loaded": False, "loading": False, "error": None, "last_loaded": None, "loading_time": None},
            "advanced": {"loaded": False, "loading": False, "error": None, "last_loaded": None, "loading_time": None},
            "simple": {"loaded": True, "loading": False, "error": None, "last_loaded": time.time(), "loading_time": 0}
        }
        
        # Simple model is always available
        self.models_loaded = True
        
    def load_config(self):
        """Load the API configuration file."""
        try:
            config_path = os.path.join(Path(__file__).parent, "sentiment_api_config.json")
            with open(config_path, "r") as f:
                self.config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            
            # Configure logging based on config
            if "logging" in self.config:
                log_level = self.config["logging"].get("log_level", "INFO")
                numeric_level = getattr(logging, log_level.upper(), logging.INFO)
                logger.setLevel(numeric_level)
                
                if self.config["logging"].get("log_model_loading", True):
                    logger.info("Model loading logs enabled")
                    
            return True
        except Exception as e:
            logger.warning(f"Error loading config: {e}. Using default configuration.")
            self.config = {}
            return False
        
    def load_ensemble_model(self):
        """Load the ensemble model."""
        try:
            from src.models.ensemble_sentiment_model import create_ensemble_model
            
            # Start timing model loading
            start_time = time.time()
            self.model_states["ensemble"]["loading"] = True
            
            # Get ensemble config path from the configuration
            ensemble_config_path = self.config.get("ensemble_config_path", None)
            
            # Resolve relative path if needed
            if ensemble_config_path and ensemble_config_path.startswith(".."):
                base_dir = Path(__file__).parent.parent
                ensemble_config_path = str(base_dir / ensemble_config_path[3:])
                
            logger.info(f"Initializing ensemble sentiment model from config: {ensemble_config_path}")
            self.ensemble_model = create_ensemble_model(ensemble_config_path)
            
            # Apply threshold from config if available
            if "model_thresholds" in self.config and hasattr(self.ensemble_model, "neutral_threshold"):
                self.ensemble_model.neutral_threshold = self.config["model_thresholds"].get("neutral_threshold", 0.6)
                logger.info(f"Set ensemble neutral threshold to {self.ensemble_model.neutral_threshold}")
            
            # Record the loading time
            self.ensemble_loading_time = time.time() - start_time
            logger.info(f"Ensemble model initialized successfully in {self.ensemble_loading_time:.2f} seconds")
                
            # Update model state
            self.model_states["ensemble"]["loaded"] = True
            self.model_states["ensemble"]["loading"] = False
            self.model_states["ensemble"]["loading_time"] = self.ensemble_loading_time
            self.model_states["ensemble"]["last_loaded"] = time.time()
            
            return self.ensemble_model
        except Exception as e:
            error_msg = f"Error initializing ensemble model: {str(e)}"
            logger.error(error_msg)
            self.load_errors.append(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            
            # Update error state
            self.model_states["ensemble"]["loading"] = False
            self.model_states["ensemble"]["error"] = str(e)
            
            return None
    
    def load_attention_model(self):
        """Load the attention model."""
        try:
            from src.models.attention_sentiment_model import create_attention_enhanced_model
            
            # Start timing model loading
            start_time = time.time()
            self.model_states["attention"]["loading"] = True
            
            # Get attention model path from the configuration
            attention_model_path = self.config.get("attention_model_path", None)
            
            # Resolve relative path if needed
            if attention_model_path and attention_model_path.startswith(".."):
                base_dir = Path(__file__).parent.parent
                attention_model_path = str(base_dir / attention_model_path[3:])
                
            logger.info(f"Initializing attention sentiment model from: {attention_model_path}")
            
            # If custom path is provided, use it, otherwise use default creation
            if attention_model_path and os.path.exists(attention_model_path):
                from src.models.attention_sentiment_model import SentimentAttentionModel
                from transformers import RobertaTokenizer
                
                self.attention_model = SentimentAttentionModel.from_pretrained(attention_model_path)
                self.tokenizer = RobertaTokenizer.from_pretrained(attention_model_path)
            else:
                self.attention_model, self.tokenizer = create_attention_enhanced_model()
            
            # Record the loading time
            self.attention_loading_time = time.time() - start_time
            logger.info(f"Attention model initialized successfully in {self.attention_loading_time:.2f} seconds")
                
            # Update model state
            self.model_states["attention"]["loaded"] = True
            self.model_states["attention"]["loading"] = False
            self.model_states["attention"]["loading_time"] = self.attention_loading_time
            self.model_states["attention"]["last_loaded"] = time.time()
            
            return self.attention_model, self.tokenizer
        except Exception as e:
            error_msg = f"Error initializing attention model: {str(e)}"
            logger.error(error_msg)
            self.load_errors.append(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            
            # Update error state
            self.model_states["attention"]["loading"] = False
            self.model_states["attention"]["error"] = str(e)
            
            return None, None
    
    def load_neutral_finetuner(self):
        """Load the neutral finetuner."""
        try:
            from src.models.neutral_fine_tuner import NeutralSentimentFinetuner
            
            # Start timing model loading
            start_time = time.time()
            self.model_states["neutral"]["loading"] = True
            
            # Get neutral model path from the configuration
            neutral_model_path = self.config.get("neutral_model_path", None)
            
            # Resolve relative path if needed
            if neutral_model_path and neutral_model_path.startswith(".."):
                base_dir = Path(__file__).parent.parent
                neutral_model_path = str(base_dir / neutral_model_path[3:])
                
            logger.info(f"Initializing neutral sentiment fine-tuner from: {neutral_model_path}")
            
            # Create the NeutralSentimentFinetuner without the model_path parameter
            # Check the signature of the constructor to see what parameters it accepts
            self.neutral_finetuner = NeutralSentimentFinetuner()
            
            # If we have a model path, try to load the model afterwards
            if neutral_model_path and os.path.exists(neutral_model_path):
                logger.info(f"Loading pre-trained model from {neutral_model_path}")
                # Try different loading methods based on what's available in the class
                if hasattr(self.neutral_finetuner, 'load_model'):
                    self.neutral_finetuner.load_model(neutral_model_path)
                elif hasattr(self.neutral_finetuner, 'load'):
                    self.neutral_finetuner.load(neutral_model_path)
                elif hasattr(self.neutral_finetuner, 'from_pretrained'):
                    self.neutral_finetuner = self.neutral_finetuner.from_pretrained(neutral_model_path)
                    
            # Prepare the model
            if hasattr(self.neutral_finetuner, 'load_and_prepare_base_model'):
                self.neutral_finetuner.load_and_prepare_base_model()
            
            # Record the loading time
            self.neutral_loading_time = time.time() - start_time
            logger.info(f"Neutral fine-tuner initialized successfully in {self.neutral_loading_time:.2f} seconds")
                
            # Update model state
            self.model_states["neutral"]["loaded"] = True
            self.model_states["neutral"]["loading"] = False
            self.model_states["neutral"]["loading_time"] = self.neutral_loading_time
            self.model_states["neutral"]["last_loaded"] = time.time()
            
            return self.neutral_finetuner
        except Exception as e:
            error_msg = f"Error initializing neutral finetuner: {str(e)}"
            logger.error(error_msg)
            self.load_errors.append(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            
            # Update error state
            self.model_states["neutral"]["loading"] = False
            self.model_states["neutral"]["error"] = str(e)
            
            return None
    
    def load_advanced_model(self):
        """Load the advanced model with improved error handling and CPU fallback."""
        try:
            from src.models.advanced import EnhancedSentimentModel
            
            # Get advanced model config from the configuration
            advanced_config = self.config.get("advanced_model_config", {})
            
            # Start timing model loading
            start_time = time.time()
            self.model_states["advanced"]["loading"] = True
            
            # Try to load with GPU first if available
            use_gpu = advanced_config.get("use_gpu", True)
            
            # Check if torch is available and CUDA is available
            try:
                import torch
                cuda_available = use_gpu and torch.cuda.is_available()
                if cuda_available:
                    logger.info(f"Using CUDA for advanced model. Available device count: {torch.cuda.device_count()}")
                else:
                    logger.info("Using CPU for advanced model (CUDA not available or disabled)")
            except ImportError:
                cuda_available = False
                logger.warning("PyTorch not available, falling back to CPU for advanced model")
            
            logger.info(f"Initializing advanced sentiment model with config: {advanced_config}")
            
            try:
                # First attempt - try with config settings (without device parameter)
                self.advanced_model = EnhancedSentimentModel(
                    use_roberta=advanced_config.get("use_roberta", True),
                    use_bart=advanced_config.get("use_bart", True),
                    use_rag=advanced_config.get("use_rag", True),
                    model_type=advanced_config.get("base_model_type", "distilbert")
                    # Remove device parameter - it's not supported
                )
                
                # Record the loading time
                self.advanced_loading_time = time.time() - start_time
                logger.info(f"Advanced model loaded successfully in {self.advanced_loading_time:.2f} seconds")
                
                self.model_states["advanced"]["loaded"] = True
                self.model_states["advanced"]["loading"] = False
                self.model_states["advanced"]["loading_time"] = self.advanced_loading_time
                self.model_states["advanced"]["last_loaded"] = time.time()
                
            except Exception as load_error:
                # If first attempt fails, try with reduced capabilities
                logger.warning(f"Failed to load advanced model with error: {load_error}")
                logger.info("Attempting to load advanced model with reduced capabilities...")
                
                try:
                    # Second attempt - Reduced capabilities
                    self.advanced_model = EnhancedSentimentModel(
                        use_roberta=True,      # Keep RoBERTa as it's the core model
                        use_bart=False,        # Disable BART to reduce memory usage
                        use_rag=False,         # Disable RAG as it's resource intensive
                        model_type="distilbert" # Use smaller model
                    )
                    
                    # Record the loading time for reduced capabilities
                    self.advanced_loading_time = time.time() - start_time
                    logger.info(f"Advanced model loaded with reduced capabilities in {self.advanced_loading_time:.2f} seconds")
                    
                    self.model_states["advanced"]["loaded"] = True
                    self.model_states["advanced"]["loading"] = False
                    self.model_states["advanced"]["loading_time"] = self.advanced_loading_time
                    self.model_states["advanced"]["last_loaded"] = time.time()
                    self.model_states["advanced"]["note"] = "Running in reduced capability mode"
                    
                except Exception as minimal_error:
                    # Third attempt - Minimal configuration
                    logger.warning(f"Failed to load reduced advanced model with error: {minimal_error}")
                    logger.info("Attempting to load minimal advanced model configuration...")
                    
                    try:
                        # Try with absolute minimal configuration
                        self.advanced_model = EnhancedSentimentModel(
                            use_roberta=False,
                            use_bart=False,
                            use_rag=False,
                            model_type="distilbert",
                            safe_mode=True  # Enable safe mode if it's supported
                        )
                        
                        # Record the loading time for minimal fallback
                        self.advanced_loading_time = time.time() - start_time
                        logger.info(f"Minimal advanced model loaded in {self.advanced_loading_time:.2f} seconds")
                        
                        self.model_states["advanced"]["loaded"] = True
                        self.model_states["advanced"]["loading"] = False
                        self.model_states["advanced"]["loading_time"] = self.advanced_loading_time
                        self.model_states["advanced"]["last_loaded"] = time.time()
                        self.model_states["advanced"]["note"] = "Running in minimal capability mode"
                        
                    except Exception as final_error:
                        # All attempts failed
                        error_msg = f"Failed to load advanced model in all configurations: {final_error}"
                        logger.error(error_msg)
                        self.load_errors.append(error_msg)
                        
                        self.model_states["advanced"]["loading"] = False
                        self.model_states["advanced"]["error"] = str(final_error)
                        return None
            
            return self.advanced_model
        except ImportError as e:
            # Handle missing dependencies
            error_msg = f"Error importing dependencies for advanced model: {str(e)}"
            logger.error(error_msg)
            self.load_errors.append(error_msg)
            
            # Check for specific missing dependencies and provide helpful message
            if "sentence_transformers" in str(e):
                logger.error("The sentence_transformers package is required for RAG functionality.")
                logger.error("Install it with: pip install sentence-transformers")
            elif "transformers" in str(e):
                logger.error("The transformers package is required for advanced models.")
                logger.error("Install it with: pip install transformers")
            
            self.model_states["advanced"]["loading"] = False
            self.model_states["advanced"]["error"] = str(e)
            logger.error(traceback.format_exc())
            return None
        except Exception as e:
            error_msg = f"Error initializing advanced model: {str(e)}"
            logger.error(error_msg)
            self.load_errors.append(error_msg)
            
            self.model_states["advanced"]["loading"] = False
            self.model_states["advanced"]["error"] = str(e)
            logger.error(traceback.format_exc())
            return None
            
    def load_all_models(self):
        """Load all models at once rather than lazily."""
        logger.info("Starting to load all models...")
        
        # Load config first
        self.load_config()
        
        # Get list of models to load from config or use default list
        models_to_load = self.config.get("models_to_load", ["ensemble", "attention", "neutral", "advanced"])
        
        # Load each model based on configuration
        if "ensemble" in models_to_load:
            logger.info("Loading ensemble model...")
            self.ensemble_model = self.load_ensemble_model()
            
        if "attention" in models_to_load:
            logger.info("Loading attention model...")
            self.attention_model, self.tokenizer = self.load_attention_model()
            
        if "neutral" in models_to_load:
            logger.info("Loading neutral model...")
            self.neutral_finetuner = self.load_neutral_finetuner()
            
        if "advanced" in models_to_load:
            logger.info("Loading advanced model...")
            self.advanced_model = self.load_advanced_model()
            
        # Set models_loaded flag if at least one model loaded successfully
        self.models_loaded = (
            self.ensemble_model is not None or 
            self.attention_model is not None or
            self.neutral_finetuner is not None or
            self.advanced_model is not None
        )
        
        # Log summary of loaded models
        loaded_models = []
        if self.ensemble_model is not None:
            loaded_models.append("ensemble")
        if self.attention_model is not None:
            loaded_models.append("attention")
        if self.neutral_finetuner is not None:
            loaded_models.append("neutral")
        if self.advanced_model is not None:
            loaded_models.append("advanced")
            
        if loaded_models:
            logger.info(f"Successfully loaded models: {', '.join(loaded_models)}")
        else:
            logger.warning("No models were loaded successfully")
            
        # Log any errors that occurred during loading
        if self.load_errors:
            logger.warning(f"Encountered {len(self.load_errors)} errors during model loading")
            for i, error in enumerate(self.load_errors):
                logger.warning(f"Error {i+1}: {error}")
                
        return self.models_loaded

# Initialize the model container
model_container = ModelContainer()

# Data models for API requests and responses
class TextRequest(BaseModel):
    text: str
    model_type: Optional[str] = "ensemble"  # Default to ensemble model

class TextsRequest(BaseModel):
    texts: List[str]
    model_type: Optional[str] = "ensemble"  # Default to ensemble model

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

# Add new request models for topic detection and batch context analysis
class TopicRequest(BaseModel):
    text: str
    threshold: Optional[float] = 0.55
    max_topics: Optional[int] = 3

class TopicCustomizationRequest(BaseModel):
    topic_name: str
    keywords: List[str]

class BatchContextRequest(BaseModel):
    texts: List[str]
    model_type: Optional[str] = "ensemble"
    include_topics: Optional[bool] = False
    include_summaries: Optional[bool] = False

class LoadModelRequest(BaseModel):
    """Request model for loading a model explicitly"""
    model_type: str = Field(..., description="Type of model to load: 'ensemble', 'attention', 'neutral', or 'advanced'")
    wait_for_loading: bool = Field(True, description="Whether to wait for model loading to complete before returning")

class ExtensionStatusRequest(BaseModel):
    """Request model for browser extensions checking model status"""
    check_offline_available: bool = Field(True, description="Whether to check if offline models are available")
    require_advanced_features: bool = Field(False, description="Whether advanced features are required")

@app.on_event("startup")
async def startup_event():
    """Initialize configuration on startup and load all models immediately."""
    # Load configuration
    model_container.load_config()
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Log startup information
    logger.info("=" * 50)
    logger.info(f"API startup complete - Server running on http://127.0.0.1:5000")
    logger.info(f"Starting to load all models...")
    logger.info(f"Available endpoints:")
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            methods = ", ".join(route.methods)
            logger.info(f"  {methods:<10} {route.path}")
    logger.info("=" * 50)
    
    # Get the list of models to load from config
    models_to_load = model_container.config.get("models_to_load", ["ensemble", "attention", "neutral", "advanced"])
    logger.info(f"Loading the following models: {', '.join(models_to_load)}")
    
    # Load each model synchronously to avoid memory issues
    if "ensemble" in models_to_load:
        logger.info("Loading ensemble model...")
        start_time = time.time()
        model_container.model_states["ensemble"]["loading"] = True
        ensemble_model = model_container.load_ensemble_model()
        loading_time = time.time() - start_time
        if ensemble_model:
            logger.info(f"Ensemble model loaded successfully in {loading_time:.2f} seconds")
            model_container.model_states["ensemble"]["loaded"] = True
            model_container.model_states["ensemble"]["loading"] = False
            model_container.model_states["ensemble"]["loading_time"] = loading_time
            model_container.model_states["ensemble"]["last_loaded"] = time.time()
        else:
            logger.warning("Failed to load ensemble model during startup")
            model_container.model_states["ensemble"]["loading"] = False
            model_container.model_states["ensemble"]["error"] = "Failed to load during startup"
    
    if "attention" in models_to_load:
        logger.info("Loading attention model...")
        start_time = time.time()
        model_container.model_states["attention"]["loading"] = True
        attention_model, tokenizer = model_container.load_attention_model()
        loading_time = time.time() - start_time
        if attention_model and tokenizer:
            logger.info(f"Attention model loaded successfully in {loading_time:.2f} seconds")
            model_container.model_states["attention"]["loaded"] = True
            model_container.model_states["attention"]["loading"] = False
            model_container.model_states["attention"]["loading_time"] = loading_time
            model_container.model_states["attention"]["last_loaded"] = time.time()
        else:
            logger.warning("Failed to load attention model during startup")
            model_container.model_states["attention"]["loading"] = False
            model_container.model_states["attention"]["error"] = "Failed to load during startup"
    
    if "neutral" in models_to_load:
        logger.info("Loading neutral model...")
        start_time = time.time()
        model_container.model_states["neutral"]["loading"] = True
        neutral_model = model_container.load_neutral_finetuner()
        loading_time = time.time() - start_time
        if neutral_model:
            logger.info(f"Neutral model loaded successfully in {loading_time:.2f} seconds")
            model_container.model_states["neutral"]["loaded"] = True
            model_container.model_states["neutral"]["loading"] = False
            model_container.model_states["neutral"]["loading_time"] = loading_time
            model_container.model_states["neutral"]["last_loaded"] = time.time()
        else:
            logger.warning("Failed to load neutral model during startup")
            model_container.model_states["neutral"]["loading"] = False
            model_container.model_states["neutral"]["error"] = "Failed to load during startup"
    
    if "advanced" in models_to_load:
        logger.info("Loading advanced model...")
        start_time = time.time()
        model_container.model_states["advanced"]["loading"] = True
        advanced_model = model_container.load_advanced_model()
        loading_time = time.time() - start_time
        if advanced_model:
            logger.info(f"Advanced model loaded successfully in {loading_time:.2f} seconds")
            model_container.model_states["advanced"]["loaded"] = True
            model_container.model_states["advanced"]["loading"] = False
            model_container.model_states["advanced"]["loading_time"] = loading_time
            model_container.model_states["advanced"]["last_loaded"] = time.time()
        else:
            logger.warning("Failed to load advanced model during startup")
            model_container.model_states["advanced"]["loading"] = False
            model_container.model_states["advanced"]["error"] = "Failed to load during startup"
    
    # Log summary of loaded models
    loaded_models = [model for model in model_container.model_states if model_container.model_states[model]["loaded"]]
    logger.info("=" * 50)
    logger.info("Model loading complete!")
    logger.info(f"Loaded models: {', '.join(loaded_models)}")
    logger.info("=" * 50)

@app.get("/health")
async def health_check():
    """Check if the API is running and report on model status without loading models."""
    # Update model states with current information
    model_container.model_states = {
        "ensemble": {
            "loaded": model_container.ensemble_model is not None,
            "loading": False,  # We'd need to update this from other endpoints
            "error": model_container.load_errors[-1] if model_container.load_errors else None,
            "last_loaded": time.time() if model_container.ensemble_model else None,
            "loading_time": model_container.ensemble_loading_time
        },
        "attention": {
            "loaded": model_container.attention_model is not None,
            "loading": False,
            "error": None,
            "last_loaded": time.time() if model_container.attention_model else None,
            "loading_time": model_container.attention_loading_time
        },
        "neutral": {
            "loaded": model_container.neutral_finetuner is not None,
            "loading": False,
            "error": None,
            "last_loaded": time.time() if model_container.neutral_finetuner else None,
            "loading_time": model_container.neutral_loading_time
        },
        "advanced": {
            "loaded": model_container.advanced_model is not None,
            "loading": False,
            "error": None,
            "last_loaded": time.time() if model_container.advanced_model else None,
            "loading_time": model_container.advanced_loading_time
        },
        "simple": {
            "loaded": True,  # Simple model is always available
            "loading": False,
            "error": None,
            "last_loaded": time.time(),
            "loading_time": 0
        }
    }
    
    # Report on approximate memory usage if models are loaded
    memory_usage = {}
    if model_container.ensemble_model:
        memory_usage["ensemble"] = "~50MB"  # Approximation
    if model_container.attention_model:
        memory_usage["attention"] = "~300MB"  # Approximation
    if model_container.neutral_finetuner:
        memory_usage["neutral"] = "~200MB"  # Approximation
    if model_container.advanced_model:
        memory_usage["advanced"] = "~500MB"  # Approximation
    memory_usage["simple"] = "<1MB"  # Always available rule-based model
    
    # Report device information to help with optimization
    device_info = {}
    try:
        import torch
        device_info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    except ImportError:
        device_info = {
            "cuda_available": False,
            "device_count": 0,
            "error": "PyTorch not installed"
        }
    
    # Add information about estimated loading times (based on previous loads)
    loading_times = {
        "ensemble": model_container.ensemble_loading_time,
        "attention": model_container.attention_loading_time,
        "neutral_finetuner": model_container.neutral_loading_time,
        "advanced": model_container.advanced_loading_time,
        "simple": 0  # Simple model loads instantly
    }
    
    # Add details about advanced model if it's loaded
    if model_container.advanced_model and hasattr(model_container.advanced_model, "active_models"):
        model_container.model_states["advanced_details"] = {
            "roberta": model_container.advanced_model.active_models.get("roberta", False),
            "bart": model_container.advanced_model.active_models.get("bart", False),
            "rag": model_container.advanced_model.active_models.get("rag", False)
        }
    
    return {
        "status": "online",
        "timestamp": time.time(),
        "model_states": model_container.model_states,
        "memory_usage": memory_usage,
        "device_info": device_info,
        "loading_times": loading_times,
        "load_errors": model_container.load_errors if model_container.load_errors else None,
        "extension_mode_recommended": "simple"  # Recommend simple model for extension use
    }

from src.models.ensemble_sentiment_model import analyze_sentiment

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_text(request: TextRequest):
    """
    Analyze the sentiment of a single text using the specified model.
    
    Available model types:
    - ensemble: Uses the ensemble sentiment model (default)
    - attention: Uses the advanced attention model
    - neutral: Uses the neutral-optimized model
    - advanced: Uses the enhanced model (if available)
    - simple: Uses a simple rule-based approach (always available)
    """
    try:
        model_type = request.model_type.lower()
        loading_time = None
        model_was_loaded = False
        
        # Log full request for diagnostics
        logger.info(f"Received analyze request: model={model_type}, text='{request.text[:50]}...'")
        
        # Simple rule-based approach is always available
        if model_type == "simple":
            text = request.text.lower()
            
            # Check for obviously positive sentiment
            positive_words = ["love", "amazing", "excellent", "fantastic", "great", "awesome", "exceeded"]
            if any(word in text for word in positive_words):
                result = {
                    "label": "positive",
                    "score": 2.0,
                    "category": 2,
                    "confidence": 0.95,
                    "model_used": "simple_rule_based"
                }
                logger.info(f"Simple model response: {result}")
                return result
                
            # Check for obviously negative sentiment
            negative_words = ["hate", "terrible", "awful", "horrible", "worst", "disappointing", "bad"]
            if any(word in text for word in negative_words):
                result = {
                    "label": "negative",
                    "score": 0.0,
                    "category": 0,
                    "confidence": 0.95,
                    "model_used": "simple_rule_based"
                }
                logger.info(f"Simple model response: {result}")
                return result
            
            # Default to neutral for simple model
            result = {
                "label": "neutral",
                "score": 1.0,
                "category": 1,
                "confidence": 0.5,
                "model_used": "simple_rule_based"
            }
            logger.info(f"Simple model response: {result}")
            return result
        
        elif model_type == "ensemble":
            # Check if model is already loaded
            model_was_loaded = model_container.ensemble_model is not None
            
            # Start timing for model loading
            start_time = time.time()
            
            # Load the ensemble model
            ensemble_model = model_container.load_ensemble_model()
            
            # Calculate loading time if model was just loaded
            if not model_was_loaded and ensemble_model is not None:
                loading_time = time.time() - start_time
                logger.info(f"Ensemble model loaded in {loading_time:.2f} seconds")
            
            if not ensemble_model:
                raise HTTPException(status_code=503, detail="Ensemble model not initialized")
            
            # Import function only when needed
            from src.models.ensemble_sentiment_model import analyze_sentiment
            
            # Use the ensemble model with best_confidence method
            result = analyze_sentiment(request.text, ensemble=ensemble_model, method="best_confidence")
            
            # Log results for debugging
            logger.info(f"Analyzing text with ensemble model: '{request.text}'")
            
            response = {
                "label": result["sentiment"],
                "score": float(result["sentiment_label"]),
                "category": int(result["sentiment_label"]),
                "confidence": float(result["confidence"]),
                "model_used": "ensemble",
                "model_loading_time": loading_time if loading_time else None
            }
            logger.info(f"Ensemble model response: {response}")
            return response
            
        elif model_type == "attention":
            # Check if model is already loaded
            model_was_loaded = model_container.attention_model is not None
            
            # Start timing for model loading
            start_time = time.time()
            
            # Load the attention model
            attention_model, tokenizer = model_container.load_attention_model()
            
            # Calculate loading time if model was just loaded
            if not model_was_loaded and attention_model is not None:
                loading_time = time.time() - start_time
                logger.info(f"Attention model loaded in {loading_time:.2f} seconds")
            
            if not attention_model or not tokenizer:
                raise HTTPException(status_code=503, detail="Attention model not initialized")
            
            # Import function only when needed
            from src.models.attention_sentiment_model import analyze_sentiment_with_attention
            
            # Use the attention model
            results = analyze_sentiment_with_attention(
                attention_model,
                tokenizer,
                [request.text],
                visualize=False
            )
            
            if not results:
                raise HTTPException(status_code=500, detail="Error analyzing with attention model")
                
            result = results[0]
            
            # Map prediction to our common format
            sentiment_category = {
                "Negative": 0,
                "Neutral": 1,
                "Positive": 2
            }
            
            response = {
                "label": result["sentiment"],
                "score": float(result["prediction"]),
                "category": sentiment_category.get(result["sentiment"], 1),
                "confidence": 0.8,  # Attention model doesn't provide confidence, use a default
                "model_used": "attention",
                "model_loading_time": loading_time if loading_time else None
            }
            logger.info(f"Attention model response: {response}")
            return response
            
        elif model_type == "neutral":
            # Check if model is already loaded
            model_was_loaded = model_container.neutral_finetuner is not None
            
            # Start timing for model loading
            start_time = time.time()
            
            # Load the neutral finetuner
            neutral_finetuner = model_container.load_neutral_finetuner()
            
            # Calculate loading time if model was just loaded
            if not model_was_loaded and neutral_finetuner is not None:
                loading_time = time.time() - start_time
                logger.info(f"Neutral model loaded in {loading_time:.2f} seconds")
            
            if not neutral_finetuner or not hasattr(neutral_finetuner, 'model'):
                raise HTTPException(status_code=503, detail="Neutral fine-tuner not initialized")
            
            try:
                import torch
                
                # Tokenize the text
                inputs = neutral_finetuner.tokenizer(
                    request.text,
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(neutral_finetuner.device)
                
                # Make prediction
                with torch.no_grad():
                    outputs = neutral_finetuner.model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    prediction = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][prediction].item()
                
                # Map prediction to sentiment
                sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
                
                response = {
                    "label": sentiment_map[prediction],
                    "score": float(prediction),
                    "category": int(prediction),
                    "confidence": float(confidence),
                    "model_used": "neutral_finetuner",
                    "model_loading_time": loading_time if loading_time else None
                }
                logger.info(f"Neutral model response: {response}")
                return response
            except Exception as e:
                logger.error(f"Error in neutral model prediction: {e}")
                raise HTTPException(status_code=500, detail=f"Neutral model error: {str(e)}")
            
        elif model_type == "advanced":
            # Check if model is already loaded
            model_was_loaded = model_container.advanced_model is not None
            
            # Start timing for model loading
            start_time = time.time()
            
            # Load the advanced model
            advanced_model = model_container.load_advanced_model()
            
            # Calculate loading time if model was just loaded
            if not model_was_loaded and advanced_model is not None:
                loading_time = time.time() - start_time
                logger.info(f"Advanced model loaded in {loading_time:.2f} seconds")
            
            if not advanced_model:
                raise HTTPException(status_code=503, detail="Advanced model not initialized")
            
            # Use the advanced model - Fix for both 'int' object and 'list index out of range' errors
            prediction_result = advanced_model.predict(request.text)
            
            # Check if the result is a list or a single value
            if isinstance(prediction_result, list):
                prediction = prediction_result[0] if prediction_result else 0
            else:
                # If it's already a single value, use it directly
                prediction = prediction_result
            
            # Map prediction to standard categories (0 to 2)
            # Making sure we're in the expected range for sentiment_categories
            if prediction < -0.3:
                category = 0  # negative
            elif prediction < 0.3:
                category = 1  # neutral
            else:
                category = 2  # positive
            
            # Get label safely using category (0-2) instead of prediction (which could be 4)
            try:
                label = advanced_model.get_sentiment_label(category)
            except (IndexError, ValueError):
                # Fallback labels if get_sentiment_label fails
                sentiment_labels = ["negative", "neutral", "positive"]
                label = sentiment_labels[category if 0 <= category < 3 else 1]
            
            # Get summary if BART is available and text is long enough
            summary = None
            if hasattr(advanced_model, "active_models") and advanced_model.active_models.get("bart", False) and len(request.text.split()) > 10:
                summary_result = advanced_model.summarize_text(request.text)
                summary = summary_result.get("summary")
            
            response = {
                "label": label,
                "score": float(prediction),
                "category": category,
                "summary": summary,
                "model_used": "advanced",
                "model_loading_time": loading_time if loading_time else None
            }
            logger.info(f"Advanced model response: {response}")
            return response
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-batch")
async def analyze_texts(request: TextsRequest):
    """
    Analyze the sentiment of multiple texts in batch using the specified model.
    """
    try:
        model_type = request.model_type.lower()
        
        if model_type == "ensemble":
            if not model_container.ensemble_model:
                raise HTTPException(status_code=503, detail="Ensemble model not initialized")
            
            # Use the ensemble model
            results = analyze_sentiment(request.texts, ensemble=model_container.ensemble_model)
            
            # Format the results to a common format
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append({
                    "text": request.texts[i],
                    "label": result["sentiment"],
                    "score": float(result["sentiment_label"]),
                    "category": int(result["sentiment_label"]),
                    "confidence": float(result["confidence"])
                })
            
            return {"results": formatted_results}
            
        elif model_type == "attention":
            if not model_container.attention_model:
                raise HTTPException(status_code=503, detail="Attention model not initialized")
            
            # Use the attention model
            results = analyze_sentiment_with_attention(
                model_container.attention_model,
                model_container.tokenizer,
                request.texts,
                visualize=False
            )
            
            if not results:
                raise HTTPException(status_code=500, detail="Error analyzing with attention model")
            
            # Map prediction to our common format
            sentiment_category = {
                "Negative": 0,
                "Neutral": 1,
                "Positive": 2
            }
            
            # Format the results
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append({
                    "text": request.texts[i],
                    "label": result["sentiment"],
                    "score": float(result["prediction"]),
                    "category": sentiment_category.get(result["sentiment"], 1),
                    "confidence": 0.8  # Attention model doesn't provide confidence, use a default
                })
            
            return {"results": formatted_results}
            
        elif model_type == "advanced":
            if not model_container.advanced_model:
                raise HTTPException(status_code=503, detail="Advanced model not initialized")
            
            # Use the advanced model
            predictions = model_container.advanced_model.predict(request.texts)
            
            # Format results
            formatted_results = []
            for i, prediction in enumerate(predictions):
                label = model_container.advanced_model.get_sentiment_label(prediction)
                
                # Map the prediction to our 3-category system
                if prediction < -0.3:
                    category = 0  # negative
                elif prediction < 0.3:
                    category = 1  # neutral
                else:
                    category = 2  # positive
                
                formatted_results.append({
                    "text": request.texts[i],
                    "label": label,
                    "score": float(prediction),
                    "category": category
                })
            
            return {"results": formatted_results}
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Error analyzing texts in batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-tweet")
async def analyze_tweet_with_comments(request: TweetCommentsRequest):
    """
    Analyze a tweet and its comments for sentiment and themes.
    
    This endpoint performs sentiment analysis on a tweet and its comments,
    and also provides a summary of the comments if BART is available.
    """
    try:
        if not model_container.advanced_model:
            raise HTTPException(status_code=503, detail="Advanced model not initialized")
            
        if not model_container.advanced_model.comment_analyzer:
            raise HTTPException(status_code=503, detail="Comment analyzer not available")
            
        # Analyze tweet and comments
        result = model_container.advanced_model.analyze_tweet_with_comments(
            request.tweet, 
            request.comments, 
            request.max_comments
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
    except Exception as e:
        logger.error(f"Error analyzing tweet with comments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_text(request: SummarizationRequest):
    """
    Generate a summary of a text using BART or fallback to a simpler method.
    
    This endpoint attempts to use the BART model if available, but falls back to
    a rule-based approach if the advanced model is not initialized.
    """
    try:
        # Try to use advanced model with BART if available
        if model_container.advanced_model and model_container.advanced_model.active_models.get("bart", False):
            # Generate summary using BART
            result = model_container.advanced_model.summarize_text(
                request.text,
                max_length=request.max_length,
                min_length=request.min_length,
                do_sample=request.do_sample
            )
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
                
            return result
        
        # Fallback to a simpler summarization approach
        logger.info("Advanced model with BART not available, using fallback summarization")
        
        # Simple extractive summarization fallback
        summary = simple_extractive_summarization(
            request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        return {
            "summary": summary,
            "method": "extractive_fallback",
            "model": "none"
        }
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def simple_extractive_summarization(text, max_sentences=3):
    """
    Simple extractive summarization fallback for when BART model is unavailable.
    Takes the first few sentences of the text as a simple summary.
    
    Args:
        text (str): The text to summarize
        max_sentences (int): Maximum number of sentences to include
        
    Returns:
        str: A simple extractive summary
    """
    # Check if text is too short to summarize
    if len(text) < 200:
        return text
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # If text has few sentences, return as is
    if len(sentences) <= max_sentences:
        return text
    
    # Take first few sentences based on text length
    num_sentences = min(max_sentences, max(2, int(len(sentences) / 5)))
    summary = ' '.join(sentences[:num_sentences])
    
    return summary

@app.post("/rag/add-knowledge")
async def add_to_knowledge_base(request: RagRequest, background_tasks: BackgroundTasks):
    """
    Add texts to the RAG knowledge base.
    
    This endpoint requires the RAG system to be enabled.
    The operation runs in the background so the API can respond quickly.
    """
    try:
        if not model_container.advanced_model:
            raise HTTPException(status_code=503, detail="Advanced model not initialized")
            
        if not model_container.advanced_model.active_models.get("rag", False):
            raise HTTPException(status_code=503, detail="RAG system not available")
            
        # Add to knowledge base in background
        background_tasks.add_task(
            model_container.advanced_model.add_to_knowledge_base,
            request.texts,
            request.metadatas
        )
        
        return {
            "status": "success",
            "message": f"Adding {len(request.texts)} texts to knowledge base in background"
        }
    except Exception as e:
        logger.error(f"Error adding to knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/query")
async def query_knowledge_base(request: TextRequest):
    """
    Query the RAG knowledge base for relevant context.
    
    This endpoint requires the RAG system to be enabled.
    """
    try:
        if not model_container.advanced_model:
            raise HTTPException(status_code=503, detail="Advanced model not initialized")
            
        if not model_container.advanced_model.active_models.get("rag", False):
            raise HTTPException(status_code=503, detail="RAG system not available")
            
        # Get relevant contexts
        contexts = model_container.advanced_model.get_context_for_text(request.text)
        
        # Format results
        results = []
        for doc in contexts:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, "metadata") else {}
            })
            
        return {
            "query": request.text,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-with-context")
async def analyze_with_context(request: TextRequest):
    """
    Analyze sentiment with additional context from the knowledge base.
    
    This endpoint leverages the RAG system to retrieve relevant context
    for the input text, and uses that context to enhance the sentiment analysis.
    """
    try:
        if not model_container.advanced_model:
            raise HTTPException(status_code=503, detail="Advanced model not initialized")
            
        if not model_container.advanced_model.active_models.get("rag", False):
            raise HTTPException(status_code=503, detail="RAG system not available")
            
        # Use the analyze_with_context method from EnhancedSentimentModel
        result = model_container.advanced_model.analyze_with_context(request.text)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
    except Exception as e:
        logger.error(f"Error analyzing with context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-batch-with-context")
async def analyze_batch_with_context(request: BatchContextRequest):
    """
    Analyze multiple texts with context enhancement from the RAG system.
    
    This endpoint processes a batch of texts, retrieving relevant context for each
    and providing enhanced sentiment analysis. Optionally includes topic detection
    and summarization.
    """
    try:
        if not model_container.advanced_model:
            raise HTTPException(status_code=503, detail="Advanced model not initialized")
            
        if not model_container.advanced_model.active_models.get("rag", False):
            raise HTTPException(status_code=503, detail="RAG system not available")
        
        # Process batch using our batch processor
        results = []
        
        # We'll process each text individually to provide detailed analysis
        for text in request.texts:
            # Get context-enhanced sentiment analysis
            result = model_container.advanced_model.analyze_with_context(text)
            
            if "error" in result:
                results.append({"text": text, "error": result["error"]})
                continue
                
            # Perform topic detection if requested
            if request.include_topics and hasattr(model_container.advanced_model, "topic_detector"):
                topics = model_container.advanced_model.detect_topics(text)
                result["topics"] = topics
            
            # Generate summary if requested and BART is available
            if (request.include_summaries and 
                model_container.advanced_model.active_models.get("bart", False) and 
                len(text.split()) > 10):
                summary_result = model_container.advanced_model.summarize_text(text)
                result["summary"] = summary_result.get("summary")
            
            results.append(result)
            
        return {"results": results}
    except Exception as e:
        logger.error(f"Error analyzing batch with context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-topics")
async def detect_topics(request: TopicRequest):
    """
    Detect topics in a text using the topic detection system.
    
    This endpoint analyzes the input text and returns the detected topics
    with confidence scores.
    """
    try:
        if not model_container.advanced_model:
            raise HTTPException(status_code=503, detail="Advanced model not initialized")
            
        if not hasattr(model_container.advanced_model, "topic_detector") or not model_container.advanced_model.topic_detector:
            raise HTTPException(status_code=503, detail="Topic detector not available")
            
        # Detect topics
        topics = model_container.advanced_model.detect_topics(
            request.text,
            threshold=request.threshold,
            max_topics=request.max_topics
        )
        
        return {
            "text": request.text,
            "topics": topics
        }
    except Exception as e:
        logger.error(f"Error detecting topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-custom-topic")
async def add_custom_topic(request: TopicCustomizationRequest):
    """
    Add a custom topic definition to the topic detection system.
    
    This endpoint allows users to define their own topics by providing
    a topic name and a list of keywords that represent the topic.
    """
    try:
        if not model_container.advanced_model:
            raise HTTPException(status_code=503, detail="Advanced model not initialized")
            
        if not hasattr(model_container.advanced_model, "topic_detector") or not model_container.advanced_model.topic_detector:
            raise HTTPException(status_code=503, detail="Topic detector not available")
            
        # Add custom topic
        success = model_container.advanced_model.add_custom_topic(request.topic_name, request.keywords)
        
        if not success:
            raise HTTPException(
                status_code=400, 
                detail="Failed to add custom topic. Ensure you provided a valid topic name and keywords."
            )
            
        return {
            "status": "success",
            "message": f"Added custom topic: {request.topic_name}"
        }
    except Exception as e:
        logger.error(f"Error adding custom topic: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache-stats")
async def get_cache_statistics():
    """
    Get statistics about the sentiment cache system.
    
    This endpoint provides information about the cache usage, hit rate,
    and current size.
    """
    try:
        if not model_container.advanced_model:
            raise HTTPException(status_code=503, detail="Advanced model not initialized")
            
        if not hasattr(model_container.advanced_model, "cache") or not model_container.advanced_model.cache:
            raise HTTPException(status_code=503, detail="Cache system not available")
            
        # Get cache statistics
        stats = model_container.advanced_model.cache.get_stats()
        
        return stats
    except Exception as e:
        logger.error(f"Error getting cache statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-emotions")
async def analyze_emotions(request: TextRequest):
    """
    Analyze text to detect sub-sentiments/emotions like happiness, sadness, anger, etc.
    
    This endpoint provides a more nuanced emotional analysis beyond basic
    positive/negative/neutral sentiment classification.
    """
    try:
        if not model_container.advanced_model:
            raise HTTPException(status_code=503, detail="Advanced model not initialized")
            
        result = model_container.advanced_model.predict_with_emotions(request.text)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return {
            "text": request.text,
            "sentiment": result["sentiment"],
            "emotions": result["emotions"],
            "topics": result.get("topics", {}),
            "meta": result.get("meta", {})
        }
    except Exception as e:
        logger.error(f"Error analyzing emotions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-model")
async def load_model(request: LoadModelRequest):
    """
    Explicitly load a model without performing analysis.
    
    This endpoint allows users to trigger model loading separately from analysis,
    which can improve user experience by allowing them to choose when to load models.
    
    Returns information about loading time and model status.
    """
    try:
        model_type = request.model_type.lower()
        wait = request.wait_for_loading
        
        # Start timing
        start_time = time.time()
        
        # Track model loading status
        model_loaded = False
        loading_time = None
        
        # Log request
        logger.info(f"Received model loading request: model={model_type}, wait={wait}")
        
        # Check if model is already loaded
        if model_type == "ensemble":
            if model_container.ensemble_model is not None:
                logger.info("Ensemble model already loaded")
                model_loaded = True
            else:
                if wait:
                    # Load model and wait for completion
                    model_container.load_ensemble_model()
                    loading_time = time.time() - start_time
                    model_loaded = model_container.ensemble_model is not None
                else:
                    # Start loading in background
                    background_tasks = BackgroundTasks()
                    background_tasks.add_task(model_container.load_ensemble_model)
        
        elif model_type == "attention":
            if model_container.attention_model is not None:
                logger.info("Attention model already loaded")
                model_loaded = True
            else:
                if wait:
                    # Load model and wait for completion
                    model_container.load_attention_model()
                    loading_time = time.time() - start_time
                    model_loaded = model_container.attention_model is not None
                else:
                    # Start loading in background
                    background_tasks = BackgroundTasks()
                    background_tasks.add_task(model_container.load_attention_model)
        
        elif model_type == "neutral":
            if model_container.neutral_finetuner is not None:
                logger.info("Neutral model already loaded")
                model_loaded = True
            else:
                if wait:
                    # Load model and wait for completion
                    model_container.load_neutral_finetuner()
                    loading_time = time.time() - start_time
                    model_loaded = model_container.neutral_finetuner is not None
                else:
                    # Start loading in background
                    background_tasks = BackgroundTasks()
                    background_tasks.add_task(model_container.load_neutral_finetuner)
                    
        elif model_type == "advanced":
            if model_container.advanced_model is not None:
                logger.info("Advanced model already loaded")
                model_loaded = True
            else:
                if wait:
                    # Load model and wait for completion
                    model_container.load_advanced_model()
                    loading_time = time.time() - start_time
                    model_loaded = model_container.advanced_model is not None
                else:
                    # Start loading in background
                    background_tasks = BackgroundTasks()
                    background_tasks.add_task(model_container.load_advanced_model)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
        
        # Prepare response
        response = {
            "model_type": model_type,
            "model_loaded": model_loaded,
            "wait_completed": wait,
            "loading_time": loading_time if loading_time else None,
            "timestamp": time.time()
        }
        
        if not wait and not model_loaded:
            # If not waiting and model wasn't already loaded, it's now loading in background
            response["status"] = "loading_started"
            return JSONResponse(content=response, background=background_tasks)
        else:
            # Model was either already loaded or we waited for it to load
            response["status"] = "complete" if model_loaded else "error"
            return response
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log API requests and their timing with detailed information."""
    start_time = time.time()
    
    # Get client IP and request method/path
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path
    
    # Generate unique request ID for tracking
    request_id = f"{int(time.time() * 1000)}-{hash(str(time.time()))}"[-8:]
    
    # Prepare to capture request body
    request_body = None
    
    # Log the request details
    try:
        if method == "POST":
            # Clone the request to read the body
            body_bytes = await request.body()
            # Reset the request body stream position
            request._body = body_bytes
            
            # Try to decode the body as JSON
            try:
                request_body = json.loads(body_bytes)
                # Truncate long text fields for readability
                if isinstance(request_body, dict) and "text" in request_body and isinstance(request_body["text"], str):
                    if len(request_body["text"]) > 100:
                        request_body["text"] = request_body["text"][:100] + "..."
                
                logger.info(f"[REQ-{request_id}] {method} {path} | Client: {client_ip} | Body: {json.dumps(request_body)}")
            except:
                # If not JSON, just log the raw text (truncated)
                request_body = body_bytes.decode('utf-8', errors='replace')
                logger.info(f"[REQ-{request_id}] {method} {path} | Client: {client_ip} | Body: {request_body[:100]}")
        else:
            # For non-POST requests, just log the basics
            logger.info(f"[REQ-{request_id}] {method} {path} | Client: {client_ip}")
            
    except Exception as e:
        logger.warning(f"[REQ-{request_id}] Failed to log request body: {str(e)}")
    
    # Process the request and capture the response
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Try to capture and log response body for API endpoints
    response_body = None
    original_response_body = None
    
    try:
        # Only intercept JSON responses from specific endpoints
        if path.startswith(("/analyze", "/extension")) and "application/json" in response.headers.get("content-type", ""):
            # Get the original response content
            original_response_body = b""
            async for chunk in response.body_iterator:
                original_response_body += chunk
            
            # Try to decode and log the response body
            try:
                response_body = json.loads(original_response_body)
                # Truncate long text fields for readability
                if isinstance(response_body, dict):
                    for key in response_body:
                        if isinstance(response_body[key], str) and len(response_body[key]) > 100:
                            response_body[key] = response_body[key][:100] + "..."
                
                logger.info(
                    f"[RES-{request_id}] {method} {path} | Status: {response.status_code} | "
                    f"Time: {process_time:.3f}s | Body: {json.dumps(response_body)}"
                )
            except:
                # If not valid JSON, log as is
                logger.info(
                    f"[RES-{request_id}] {method} {path} | Status: {response.status_code} | "
                    f"Time: {process_time:.3f}s | Body: [binary or invalid JSON]"
                )
            
            # Create a new response with the original body
            return Response(
                content=original_response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        else:
            # For non-intercepted endpoints, just log basic response info
            logger.info(
                f"[RES-{request_id}] {method} {path} | Status: {response.status_code} | Time: {process_time:.3f}s"
            )
    except Exception as e:
        logger.warning(f"[RES-{request_id}] Failed to capture response body: {str(e)}")
    
    # Return the original response if we didn't create a new one
    return response

@app.get("/extension/model-status")
async def extension_model_status():
    """
    Specialized endpoint for browser extensions to check model availability.
    This endpoint is highly optimized to minimize server load and avoid triggering model loading.
    """
    # Only report on the simple model which is always available and the status of other models
    # without attempting to load them
    return {
        "status": "online",
        "timestamp": time.time(),
        "models": {
            "simple": {
                "loaded": True,
                "available": True,
                "recommended": True,
                "loading_time": 0
            },
            "ensemble": {
                "loaded": model_container.ensemble_model is not None,
                "available": True,
                "recommended": False,
                "loading_time": model_container.ensemble_loading_time
            },
            "attention": {
                "loaded": model_container.attention_model is not None,
                "available": True,
                "recommended": False,
                "loading_time": model_container.attention_loading_time
            }
        },
        "recommended_model": "simple"
    }

@app.post("/extension/model-status")
async def get_extension_model_status(request: ExtensionStatusRequest):
    """
    Specialized endpoint for browser extensions to efficiently check model status.
    This endpoint is optimized to be lightweight and not trigger model loading.
    It helps browser extensions make smart decisions about which models to use.
    """
    try:
        # Start with simple model which is always available
        available_models = ["simple"]
        recommended_model = "simple"
        loading_times = {
            "simple": 0  # Simple model loads instantly
        }
        
        # Check which models are already loaded (without triggering loading)
        if model_container.ensemble_model is not None:
            available_models.append("ensemble")
            loading_times["ensemble"] = model_container.ensemble_loading_time or 0
            
        if model_container.attention_model is not None:
            available_models.append("attention")
            loading_times["attention"] = model_container.attention_loading_time or 0
            
        if model_container.neutral_finetuner is not None:
            available_models.append("neutral")
            loading_times["neutral"] = model_container.neutral_loading_time or 0
            
        if model_container.advanced_model is not None:
            available_models.append("advanced")
            loading_times["advanced"] = model_container.advanced_loading_time or 0
            
        # Determine recommended model for the extension
        # If any advanced model is loaded, recommend it over simple
        if "ensemble" in available_models:
            recommended_model = "ensemble"
        elif "neutral" in available_models:
            recommended_model = "neutral"
            
        # Check device capabilities if requested
        device_info = {}
        if request.require_advanced_features:
            device_info = {
                "cuda_available": torch.cuda.is_available() if 'torch' in sys.modules else False,
                "device_count": torch.cuda.device_count() if 'torch' in sys.modules and torch.cuda.is_available() else 0
            }
            
        return {
            "available_models": available_models,
            "recommended_model": recommended_model,
            "loading_times": loading_times,
            "device_info": device_info,
            "model_states": {
                name: state for name, state in model_container.model_states.items() 
                if name in ["simple", "ensemble", "neutral"]  # Only return the models an extension would likely use
            },
            "extension_mode": "offline" if request.check_offline_available else "online"
        }
    except Exception as e:
        logger.error(f"Error getting extension model status: {e}")
        # Return a basic response even in case of error
        return {
            "available_models": ["simple"],
            "recommended_model": "simple",
            "extension_mode": "offline",
            "error": str(e)
        }

@app.post("/extension/analyze")
async def extension_analyze(request: TextRequest):
    """
    Specialized endpoint for browser extensions to analyze text efficiently.
    This endpoint prioritizes speed and offline availability, using the simple model by default.
    It avoids loading heavy models unless explicitly requested.
    """
    try:
        text = request.text
        model_type = request.model_type.lower() if hasattr(request, "model_type") else "simple"
        
        # Log the extension request
        logger.info(f"Browser extension analyze request: model={model_type}, text='{text[:50]}...'")
        
        # Always default to the simple model for extension use unless explicitly requested otherwise
        if model_type != "simple" and not request.model_type:
            model_type = "simple"
        
        # Simple rule-based approach is always available and very fast
        if model_type == "simple":
            text_lower = text.lower()
            
            # Check for obviously positive sentiment
            positive_words = ["love", "amazing", "excellent", "fantastic", "great", "awesome", "exceeded", "happy", "glad"]
            if any(word in text_lower for word in positive_words):
                result = {
                    "label": "positive",
                    "score": 2.0,
                    "category": 2,
                    "confidence": 0.95,
                    "model_used": "simple_rule_based"
                }
                logger.info(f"Extension simple model response: {result}")
                return result
                
            # Check for obviously negative sentiment
            negative_words = ["hate", "terrible", "awful", "horrible", "worst", "disappointing", "bad", "sad", "angry"]
            if any(word in text_lower for word in negative_words):
                result = {
                    "label": "negative",
                    "score": 0.0,
                    "category": 0,
                    "confidence": 0.95,
                    "model_used": "simple_rule_based"
                }
                logger.info(f"Extension simple model response: {result}")
                return result
            
            # For more nuanced sentiment analysis, use additional patterns
            positive_patterns = ["this is good", "works well", "recommend", "helpful", "useful"]
            if any(pattern in text_lower for pattern in positive_patterns):
                result = {
                    "label": "positive",
                    "score": 1.5,
                    "category": 2,
                    "confidence": 0.8,
                    "model_used": "simple_rule_based"
                }
                logger.info(f"Extension simple model response: {result}")
                return result
            
            negative_patterns = ["this is bad", "doesn't work", "waste", "useless", "broken"]
            if any(pattern in text_lower for pattern in negative_patterns):
                result = {
                    "label": "negative",
                    "score": 0.5,
                    "category": 0,
                    "confidence": 0.8,
                    "model_used": "simple_rule_based"
                }
                logger.info(f"Extension simple model response: {result}")
                return result
            
            # Default to neutral for simple model
            result = {
                "label": "neutral",
                "score": 1.0,
                "category": 1,
                "confidence": 0.5,
                "model_used": "simple_rule_based"
            }
            logger.info(f"Extension simple model response: {result}")
            return result
        
        # If not using simple model, delegate to the main analyze endpoint
        # This allows extensions to use heavier models when needed
        return await analyze_text(request)
            
    except Exception as e:
        logger.error(f"Error in extension analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return a neutral result as fallback on error to ensure extension doesn't break
        return {
            "label": "neutral",
            "score": 1.0,
            "category": 1,
            "confidence": 0.3,
            "model_used": "fallback",
            "error": str(e)
        }

@app.post("/extension/status")
async def extension_status(request: ExtensionStatusRequest):
    """
    Provide detailed status information specifically for browser extensions.
    
    This endpoint is optimized for browser extensions to quickly determine:
    1. Which models are available and loaded
    2. Recommended model to use based on server conditions
    3. Whether offline processing is recommended
    
    The browser extension can use this information to make intelligent decisions
    about which model to use or whether to fall back to local processing.
    """
    try:
        # Check if import torch is available before trying to use it
        import importlib.util
        torch_available = importlib.util.find_spec("torch") is not None
        
        # Get current system stats
        import psutil
        system_stats = {
            "memory_available_mb": psutil.virtual_memory().available / (1024 * 1024),
            "cpu_percent": psutil.cpu_percent(),
            "cpu_count": psutil.cpu_count()
        }
        
        # Check GPU availability if torch is available
        device_info = {
            "cuda_available": torch.cuda.is_available() if torch_available else False,
            "device_count": torch.cuda.device_count() if torch_available and torch.cuda.is_available() else 0
        }
        
        # Update model states with current information 
        model_states = {
            "ensemble": {
                "loaded": model_container.ensemble_model is not None,
                "loading": False,
                "error": model_container.load_errors[-1] if model_container.load_errors else None,
                "last_loaded": time.time() if model_container.ensemble_model else None,
                "loading_time": model_container.ensemble_loading_time
            },
            "attention": {
                "loaded": model_container.attention_model is not None,
                "loading": False,
                "error": None,
                "last_loaded": time.time() if model_container.attention_model else None,
                "loading_time": model_container.attention_loading_time
            },
            "neutral": {
                "loaded": model_container.neutral_finetuner is not None,
                "loading": False,
                "error": None,
                "last_loaded": time.time() if model_container.neutral_finetuner else None,
                "loading_time": model_container.neutral_loading_time
            },
            "advanced": {
                "loaded": model_container.advanced_model is not None,
                "loading": False,
                "error": None,
                "last_loaded": time.time() if model_container.advanced_model else None,
                "loading_time": model_container.advanced_loading_time
            },
            "simple": {
                "loaded": True,  # Simple model is always available
                "loading": False,
                "error": None,
                "last_loaded": time.time(),
                "loading_time": 0
            }
        }
        
        # Determine available and loaded models
        available_models = ["simple"]  # Simple model is always available
        loaded_models = ["simple"]     # Simple model is always loaded
        
        # Check which advanced models are available and/or loaded
        if model_container.ensemble_model is not None:
            loaded_models.append("ensemble")
        elif "ensemble" not in model_container.load_errors:
            available_models.append("ensemble")
            
        if model_container.attention_model is not None:
            loaded_models.append("attention")
        elif "attention" not in model_container.load_errors:
            available_models.append("attention")
            
        if model_container.neutral_finetuner is not None:
            loaded_models.append("neutral")
        elif "neutral" not in model_container.load_errors:
            available_models.append("neutral")
            
        if model_container.advanced_model is not None:
            loaded_models.append("advanced")
        elif "advanced" not in model_container.load_errors:
            available_models.append("advanced")
        
        # Determine recommended model based on system conditions and request
        # For browser extensions, we generally want to balance speed and accuracy
        
        # Start with simple model as the default recommendation
        recommended_model = "simple"
        
        # If advanced models are requested, try to use them if they're loaded
        if request.require_advanced_features:
            if "advanced" in loaded_models:
                recommended_model = "advanced"
            elif "ensemble" in loaded_models:
                recommended_model = "ensemble"
            elif "neutral" in loaded_models:
                recommended_model = "neutral"
        else:
            # For basic sentiment analysis in extension, ensemble is good balance
            if "ensemble" in loaded_models:
                recommended_model = "ensemble"
            elif system_stats["memory_available_mb"] > 1000 and "ensemble" in available_models:
                # Ensemble is lightweight enough to load if we have memory
                recommended_model = "ensemble"
        
        # Special case: if system is under heavy load, recommend simple model
        if system_stats["cpu_percent"] > 80 or system_stats["memory_available_mb"] < 500:
            recommended_model = "simple"
        
        # Prepare detailed loading status for extension to display
        model_loading_status = {}
        for model_name, state in model_states.items():
            if model_name != "simple":  # Don't need loading status for simple model
                if state["loaded"]:
                    loading_time = state["loading_time"] or 0
                    model_loading_status[model_name] = {
                        "status": "loaded",
                        "loading_time_seconds": loading_time,
                        "last_loaded": state["last_loaded"]
                    }
                elif state["error"]:
                    model_loading_status[model_name] = {
                        "status": "error",
                        "error": state["error"]
                    }
                else:
                    model_loading_status[model_name] = {
                        "status": "available",
                        "estimated_loading_time": state["loading_time"] or "unknown"
                    }
        
        # Estimate approximate memory usage
        memory_usage = {}
        if model_container.ensemble_model:
            memory_usage["ensemble"] = "~50MB"
        if model_container.attention_model:
            memory_usage["attention"] = "~300MB"
        if model_container.neutral_finetuner:
            memory_usage["neutral"] = "~200MB"
        if model_container.advanced_model:
            memory_usage["advanced"] = "~500MB"
        memory_usage["simple"] = "<1MB"  # Simple model is always available
        
        # Create response content
        response_content = {
            "status": "online",
            "timestamp": time.time(),
            "api_version": "1.0.0",
            "available_models": available_models,
            "loaded_models": loaded_models,
            "model_loading_status": model_loading_status,
            "recommended_model": recommended_model,
            "offline_processing_available": True,  # Extension always has simple model
            "system_stats": system_stats,
            "device_info": device_info,
            "memory_usage": memory_usage
        }
        
        # Create a direct Response to avoid any issues with middleware
        return Response(
            content=json.dumps(response_content),
            media_type="application/json",
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error in extension status endpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Even on error, provide minimal information for the extension in a consistent format
        error_content = {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e),
            "available_models": ["simple"],
            "loaded_models": ["simple"],
            "recommended_model": "simple",
            "offline_processing_available": True
        }
        
        # Create a direct Response to avoid any issues with middleware
        return Response(
            content=json.dumps(error_content),
            media_type="application/json",
            status_code=500
        )

# Helper functions for extension status endpoint
def get_current_request_rate():
    """Get the current API request rate (requests per minute)"""
    global request_timestamps
    
    # Clean old timestamps (older than 1 minute)
    current_time = time.time()
    request_timestamps = [ts for ts in request_timestamps if current_time - ts <= 60]
    
    # Return requests per minute
    return len(request_timestamps)

def get_cpu_usage():
    """Get current CPU load average if available"""
    try:
        return os.getloadavg()[0]  # 1-minute load average
    except (AttributeError, OSError):
        return None

def get_memory_usage():
    """Get current memory usage percentage"""
    try:
        import psutil
        return psutil.virtual_memory().percent
    except ImportError:
        return None

def estimate_model_loading_time(model_name):
    """Estimate loading time for a model based on historical data or defaults"""
    # Use historical loading times if available
    if model_name in model_container.model_loading_times:
        return model_container.model_loading_times[model_name]
    
    # Default estimates based on model complexity
    default_times = {
        "simple": 0.1,  # Almost instant
        "roberta": 5.0,  # 5 seconds
        "distilbert": 3.0,  # 3 seconds
        "ensemble": 8.0  # 8 seconds
    }
    
    return default_times.get(model_name, 5.0)  # Default to 5 seconds if unknown

@app.post("/analyze-with-summary")
async def analyze_with_summary(request: TextRequest):
    """
    Analyze the sentiment of a text and generate a summary in a single request.
    
    This optimizes the experience for clients that need both sentiment analysis
    and text summarization, especially for browser extensions.
    """
    try:
        # First perform sentiment analysis
        sentiment_result = await analyze_text(request)
        
        # Check if text is long enough to warrant summarization (at least 50 words)
        if len(request.text.split()) < 50:
            # For short texts, just use the original
            sentiment_result["summary"] = request.text
            sentiment_result["summarization_method"] = "original_text"
            return sentiment_result
        
        # Then generate a summary
        try:
            # Try to use BART summarizer if available
            if model_container.advanced_model and model_container.advanced_model.active_models.get("bart", False):
                # Use the advanced model's BART summarizer
                summary_result = model_container.advanced_model.summarize_text(
                    request.text,
                    max_length=150,
                    min_length=50
                )
                sentiment_result["summary"] = summary_result.get("summary")
                sentiment_result["summarization_method"] = "bart"
            else:
                # Use the fallback summarization method
                summary = simple_extractive_summarization(request.text)
                sentiment_result["summary"] = summary
                sentiment_result["summarization_method"] = "extractive_fallback"
            
            return sentiment_result
            
        except Exception as e:
            # If summarization fails, return just the sentiment analysis
            logger.error(f"Error in summarization: {e}")
            sentiment_result["summary"] = request.text[:200] + "..." if len(request.text) > 200 else request.text
            sentiment_result["summarization_method"] = "truncated_text"
            sentiment_result["summarization_error"] = str(e)
            return sentiment_result
            
    except Exception as e:
        logger.error(f"Error in analyze with summary: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extension/analyze-with-summary")
async def extension_analyze_with_summary(request: TextRequest):
    """
    Specialized endpoint for browser extensions to analyze text sentiment and generate a summary.
    
    This endpoint combines sentiment analysis with text summarization in a single request,
    optimizing the experience for browser extensions by reducing network requests.
    """
    try:
        text = request.text
        model_type = request.model_type.lower() if hasattr(request, "model_type") and request.model_type else "ensemble"
        
        # Log the extension request
        logger.info(f"Browser extension analyze-with-summary request: model={model_type}, text length={len(text)}")
        
        # First perform sentiment analysis
        start_time = time.time()
        
        # If using simple model, we can analyze directly
        if model_type == "simple":
            text_lower = text.lower()
            
            # Check for obviously positive sentiment
            positive_words = ["love", "amazing", "excellent", "fantastic", "great", "awesome", "exceeded", "happy", "glad"]
            if any(word in text_lower for word in positive_words):
                sentiment_result = {
                    "label": "positive",
                    "score": 2.0,
                    "category": 2,
                    "confidence": 0.95,
                    "model_used": "simple_rule_based"
                }
            # Check for obviously negative sentiment
            elif any(word in text_lower for word in ["hate", "terrible", "awful", "horrible", "worst", "disappointing", "bad", "sad", "angry"]):
                sentiment_result = {
                    "label": "negative",
                    "score": 0.0,
                    "category": 0,
                    "confidence": 0.95,
                    "model_used": "simple_rule_based"
                }
            # Default to neutral
            else:
                sentiment_result = {
                    "label": "neutral",
                    "score": 1.0,
                    "category": 1,
                    "confidence": 0.5,
                    "model_used": "simple_rule_based"
                }
        else:
            # Use more advanced models via the analyze_text endpoint
            text_request = TextRequest(text=text, model_type=model_type)
            sentiment_result = await analyze_text(text_request)
        
        sentiment_time = time.time() - start_time
        logger.info(f"Sentiment analysis completed in {sentiment_time:.2f}s")
        
        # Generate summary based on text length
        # Skip for very short texts
        if len(text.split()) < 30:
            sentiment_result["summary"] = text
            sentiment_result["processing_time"] = sentiment_time
            sentiment_result["summary_method"] = "original_text"
            logger.info("Text too short for summarization, returning original")
            return sentiment_result
        
        # Then generate a summary
        try:
            # Try with BART summarizer if available in advanced model
            if (model_container.advanced_model and 
                model_container.advanced_model.active_models and 
                model_container.advanced_model.active_models.get("bart", False)):
                
                start_time = time.time()
                summary_result = model_container.advanced_model.summarize_text(
                    text, max_length=150, min_length=50
                )
                summary = summary_result.get("summary", "")
                summarization_time = time.time() - start_time
                
                logger.info(f"Generated BART summary in {summarization_time:.2f}s")
                sentiment_result["summary"] = summary
                sentiment_result["summary_method"] = "bart"
                sentiment_result["processing_time"] = sentiment_time + summarization_time
                
            else:
                # Try standalone BART summarizer
                try:
                    # Import only when needed
                    import sys
                    from pathlib import Path
                    sys.path.append(str(Path(__file__).parent.parent))
                    
                    from models.bart_summariser.bart_summariser import BartSummarizer
                    
                    start_time = time.time()
                    summarizer = BartSummarizer()
                    summary = summarizer.summarize(text, max_length=120, min_length=30)
                    summarization_time = time.time() - start_time
                    
                    logger.info(f"Generated standalone BART summary in {summarization_time:.2f}s")
                    sentiment_result["summary"] = summary
                    sentiment_result["summary_method"] = "standalone_bart"
                    sentiment_result["processing_time"] = sentiment_time + summarization_time
                    
                except Exception as bart_error:
                    # If standalone BART fails, use fallback
                    logger.warning(f"Standalone BART summarization failed: {str(bart_error)}. Using fallback.")
                    raise
            
        except Exception as e:
            # If BART fails, use fallback extractive summarization
            logger.warning(f"BART summarization failed: {str(e)}. Using fallback summarization.")
            start_time = time.time()
            summary = simple_extractive_summarization(text)
            summarization_time = time.time() - start_time
            
            logger.info(f"Generated extractive summary in {summarization_time:.2f}s")
            sentiment_result["summary"] = summary
            sentiment_result["summary_method"] = "extractive_fallback"
            sentiment_result["processing_time"] = sentiment_time + summarization_time
        
        return sentiment_result
            
    except Exception as e:
        logger.error(f"Error in extension analyze with summary: {e}")
        logger.error(traceback.format_exc())
        
        # Return a fallback result that includes both sentiment and summary
        return {
            "label": "neutral",
            "score": 1.0,
            "category": 1,
            "confidence": 0.3,
            "model_used": "fallback",
            "summary": text[:200] + "..." if len(text) > 200 else text,
            "summary_method": "error_fallback",
            "error": str(e)
        }

# Run the application using uvicorn
if __name__ == "__main__":
    import socket
    import uvicorn
    import subprocess
    import sys
    
    # Try to check if port 5000 is available
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 5000))
        sock.close()
        print("Port 5000 is available, starting server...")
        uvicorn.run("sentiment_api:app", host="127.0.0.1", port=5000)
    except socket.error:
        # Port 5000 is unavailable
        print("Port 5000 is already in use, trying to free it...")
        
        try:
            # Try to kill the process (Windows-specific)
            if sys.platform.startswith('win'):
                result = subprocess.run(
                    ["netstat", "-ano", "-p", "TCP"], 
                    capture_output=True, 
                    text=True
                )
                
                for line in result.stdout.splitlines():
                    if ":5000" in line and "LISTENING" in line:
                        pid = line.strip().split()[-1]
                        print(f"Found process using port 5000: PID {pid}")
                        subprocess.run(["taskkill", "/F", "/PID", pid])
                        print(f"Attempted to kill process {pid}")
                
                # If we couldn't free port 5000, fall back to 5001
                print("Could not free port 5000, using port 5001 instead...")
                uvicorn.run("sentiment_api:app", host="127.0.0.1", port=5001)
            else:
                # Non-Windows platforms, just use alternative port
                print("Using port 5001 instead...")
                uvicorn.run("sentiment_api:app", host="127.0.0.1", port=5001)
        except Exception as e:
            print(f"Error handling port conflict: {e}")
            print("Using port 5001 instead...")
            uvicorn.run("sentiment_api:app", host="127.0.0.1", port=5001)