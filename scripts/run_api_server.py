#!/usr/bin/env python
"""
Script to run the MoodMap Sentiment API server.

This script starts the FastAPI server for the MoodMap sentiment analysis and
summarization API using the new package structure.
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the MoodMap Sentiment API Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind the server to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload on code changes')
    args = parser.parse_args()
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Import and run the API server
    import uvicorn
    from mood_map.api.improved_api import app
    
    # Print startup message
    print(f"Starting MoodMap Sentiment API server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    # Run the server
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()