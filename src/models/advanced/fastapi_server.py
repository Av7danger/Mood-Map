"""
FastAPI implementation for the enhanced sentiment analysis model.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field
from src.models.advanced import EnhancedSentimentModel
from src.utils.logging_utils import setup_logging

# Setup logging
logger = setup_logging("logs/api_logs.log")

# Initialize the app
app = FastAPI(
    title="Enhanced Sentiment Analysis API",
    description="API for advanced sentiment analysis using RoBERTa, BART, and RAG",
    version="1.0.0"
)

# Initialize the model
model = None

# Model configuration
MODEL_CONFIG = {
    "use_roberta": True,
    "use_bart": True,
    "use_rag": True,
    "model_type": "distilbert"
}

# Request and response models
class SentimentRequest(BaseModel):
    """Request model for sentiment analysis."""
    texts: List[str] = Field(..., min_items=1, description="List of texts to analyze")

class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""
    sentiments: List[Dict[str, Any]] = Field(..., description="List of sentiment analysis results")
    processing_time: float = Field(..., description="Processing time in seconds")

class SummarizationRequest(BaseModel):
    """Request model for text summarization."""
    text: str = Field(..., min_length=100, description="Text to summarize")
    max_length: Optional[int] = Field(150, description="Maximum summary length")
    min_length: Optional[int] = Field(50, description="Minimum summary length")
    do_sample: Optional[bool] = Field(False, description="Whether to use sampling")

class SummarizationResponse(BaseModel):
    """Response model for text summarization."""
    summary: str = Field(..., description="Generated summary")
    original_length: int = Field(..., description="Length of original text in words")
    summary_length: int = Field(..., description="Length of summary in words")
    compression_ratio: float = Field(..., description="Compression ratio")

class KnowledgeBaseRequest(BaseModel):
    """Request model for adding to knowledge base."""
    texts: List[str] = Field(..., min_items=1, description="List of texts to add")
    metadatas: Optional[List[Dict[str, Any]]] = Field(None, description="Optional metadata for each text")

class KnowledgeBaseResponse(BaseModel):
    """Response model for knowledge base operations."""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Status message")
    ids: Optional[List[str]] = Field(None, description="IDs of added documents")

class ContextRequest(BaseModel):
    """Request model for context retrieval."""
    query: str = Field(..., description="Query text")
    num_results: Optional[int] = Field(5, description="Number of results to retrieve")

class ContextResponse(BaseModel):
    """Response model for context retrieval."""
    documents: List[Dict[str, Any]] = Field(..., description="Retrieved documents")
    query: str = Field(..., description="Original query")

class CommentAnalysisRequest(BaseModel):
    """Request model for tweet comment analysis."""
    tweet_text: str = Field(..., description="The tweet text")
    comments: List[str] = Field(..., min_items=1, description="List of comment texts")
    max_comments: Optional[int] = Field(50, description="Maximum number of comments to analyze")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model
    try:
        logger.info("Initializing Enhanced Sentiment Model for API...")
        model = EnhancedSentimentModel(
            use_roberta=MODEL_CONFIG["use_roberta"],
            use_bart=MODEL_CONFIG["use_bart"],
            use_rag=MODEL_CONFIG["use_rag"],
            model_type=MODEL_CONFIG["model_type"]
        )
        logger.info("Enhanced Sentiment Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Enhanced Sentiment Analysis API",
        "version": "1.0.0",
        "description": "API for advanced sentiment analysis using RoBERTa, BART, and RAG",
        "models": {
            "roberta": model.active_models["roberta"] if model else MODEL_CONFIG["use_roberta"],
            "bart": model.active_models["bart"] if model else MODEL_CONFIG["use_bart"],
            "rag": model.active_models["rag"] if model else MODEL_CONFIG["use_rag"]
        },
        "endpoints": [
            "/sentiment", 
            "/summarize", 
            "/knowledge-base/add", 
            "/knowledge-base/query",
            "/comments/analyze"
        ]
    }

# Sentiment analysis endpoint
@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of provided texts."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        import time
        start_time = time.time()
        
        # Get predictions
        prediction_indices = model.predict(request.texts)
        
        # Format response
        sentiments = []
        for i, idx in enumerate(prediction_indices):
            sentiment_label = model.get_sentiment_label(idx)
            sentiments.append({
                "text": request.texts[i],
                "sentiment_index": idx,
                "sentiment": sentiment_label,
                "is_positive": idx >= 3,
                "is_negative": idx <= 1,
                "is_neutral": idx == 2
            })
        
        processing_time = time.time() - start_time
        logger.info(f"Processed sentiment analysis for {len(request.texts)} texts in {processing_time:.2f}s")
        
        return {
            "sentiments": sentiments,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing sentiment: {str(e)}")

# Text summarization endpoint
@app.post("/summarize", response_model=SummarizationResponse)
async def summarize_text(request: SummarizationRequest):
    """Summarize text using BART model."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    if not model.active_models["bart"]:
        raise HTTPException(status_code=503, detail="BART model not available")
    
    try:
        result = model.summarize_text(
            request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            do_sample=request.do_sample
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        logger.info(f"Generated summary for text of length {result['original_length']} words")
        return result
    except Exception as e:
        logger.error(f"Error in text summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Error summarizing text: {str(e)}")

# Add to knowledge base endpoint
@app.post("/knowledge-base/add", response_model=KnowledgeBaseResponse)
async def add_to_knowledge_base(request: KnowledgeBaseRequest):
    """Add texts to the RAG knowledge base."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    if not model.active_models["rag"]:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        result = model.add_to_knowledge_base(request.texts, request.metadatas)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
            
        logger.info(f"Added {len(request.texts)} texts to knowledge base")
        return result
    except Exception as e:
        logger.error(f"Error adding to knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding to knowledge base: {str(e)}")

# Query knowledge base endpoint
@app.post("/knowledge-base/query", response_model=ContextResponse)
async def query_knowledge_base(request: ContextRequest):
    """Retrieve relevant context from the knowledge base."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    if not model.active_models["rag"]:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        docs = model.get_context_for_text(request.query, request.num_results)
        
        # Convert documents to serializable format
        doc_list = []
        for doc in docs:
            doc_list.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
        logger.info(f"Retrieved {len(docs)} documents for query")
        return {
            "documents": doc_list,
            "query": request.query
        }
    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying knowledge base: {str(e)}")

# Comment analysis endpoint
@app.post("/comments/analyze")
async def analyze_comments(request: CommentAnalysisRequest):
    """Analyze a tweet and its comments for sentiment and themes."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        result = model.analyze_tweet_with_comments(
            request.tweet_text,
            request.comments,
            request.max_comments
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        logger.info(f"Analyzed {len(request.comments)} comments for tweet")
        return result
    except Exception as e:
        logger.error(f"Error analyzing comments: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing comments: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the service is healthy."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Check if all active models are operational
    health_status = {
        "status": "healthy",
        "models": model.active_models
    }
    
    return health_status