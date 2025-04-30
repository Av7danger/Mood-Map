from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import sys
import os
import datetime
import json
import logging
import traceback

# Fix the import paths by properly adding the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Now import from src after fixing the path
try:
    from src.utils.input_validation import validate_input
    print("✅ Successfully imported utility functions")
except ImportError as e:
    print(f"⚠️ Warning: Could not import utility functions: {e}")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
logging.basicConfig(
    filename="backend_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set up request logging
def log_request(route, req_data=None, status=None, error=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "route": route,
        "method": request.method,
        "remote_addr": request.remote_addr,
        "user_agent": request.headers.get('User-Agent', 'Unknown'),
        "status": status,
    }
    
    if req_data:
        # Truncate long text data for cleaner logs
        if 'text' in req_data and len(req_data['text']) > 100:
            truncated_data = req_data.copy()
            truncated_data['text'] = req_data['text'][:100] + "... [truncated]"
            log_entry["data"] = truncated_data
        else:
            log_entry["data"] = req_data
    
    if error:
        log_entry["error"] = str(error)
    
    print(f"REQUEST LOG: {json.dumps(log_entry, indent=2)}")
    
    # Also log to file for persistence
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"api_requests_{datetime.datetime.now().strftime('%Y-%m-%d')}.log")
    
    with open(log_file, 'a') as f:
        f.write(f"{json.dumps(log_entry)}\n")

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "sentiment_api_config.json")
if not os.path.exists(config_path):
    # Try to copy from config directory if not found in backend
    src_config = os.path.join(parent_dir, "config", "sentiment_api_config.json")
    if os.path.exists(src_config):
        import shutil
        print(f"Copying config from {src_config} to {config_path}")
        shutil.copy2(src_config, config_path)
    else:
        print("Warning: Could not find sentiment_api_config.json. Using default configuration.")
        config = {
            "positive_keywords": ["love", "happy", "great", "excellent", "good", "best", "amazing"],
            "negative_keywords": ["hate", "bad", "terrible", "awful", "worst", "horrible", "disappointing"]
        }

if os.path.exists(config_path):
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    print("Configuration loaded successfully.")
else:
    print("Using default configuration.")
    config = {
        "positive_keywords": ["love", "happy", "great", "excellent", "good", "best", "amazing"],
        "negative_keywords": ["hate", "bad", "terrible", "awful", "worst", "horrible", "disappointing"]
    }

@app.route('/')
def home():
    log_request('/', status="200 OK")
    return jsonify({"status": "ok", "message": "Backend is running!"})

@app.route('/health')
def health_check():
    """Health check endpoint for the browser extension to test API connectivity."""
    log_request('/health', status="200 OK")
    return jsonify({
        "status": "ok", 
        "message": "API is healthy",
        "version": "1.0.0",
        "timestamp": datetime.datetime.now().isoformat()
    })

def get_sentiment_percentage(text):
    """
    A simple rule-based sentiment analyzer that returns a percentage.
    This is a placeholder for the ML model.
    """
    positive_count = 0
    negative_count = 0
    
    # Count positive and negative keywords
    for word in config.get("positive_keywords", []):
        if word.lower() in text.lower():
            positive_count += 1
            
    for word in config.get("negative_keywords", []):
        if word.lower() in text.lower():
            negative_count += 1
    
    # If no sentiment words found, return neutral
    if positive_count == 0 and negative_count == 0:
        return 50
        
    # Calculate the percentage (0-100)
    total = positive_count + negative_count
    positive_percentage = int((positive_count / total) * 100)
    
    return positive_percentage

def get_sentiment_category(percentage):
    """Convert percentage to category"""
    if percentage < 20:
        return 0  # Overwhelmingly negative
    elif percentage < 40:
        return 1  # Negative
    elif percentage < 60:
        return 2  # Neutral  
    elif percentage < 80:
        return 3  # Positive
    else:
        return 4  # Overwhelmingly positive

def get_sentiment_emojis(category_index, sentiment_percentage=None):
    """
    Generate appropriate emojis based on sentiment category and intensity.
    
    Args:
        category_index: Integer from 0-4 representing sentiment category
        sentiment_percentage: Optional percentage value for more granular emoji selection
        
    Returns:
        Dict containing primary and secondary emojis representing the sentiment
    """
    # Define emoji mappings for each sentiment category with intensity variations
    emoji_mappings = {
        0: {  # Overwhelmingly negative
            "primary": ["😠", "😡", "🤬", "😤"],
            "secondary": ["💔", "👎", "🚫", "❌", "😖"]
        },
        1: {  # Negative
            "primary": ["🙁", "😕", "😒", "😞"],
            "secondary": ["👎", "💢", "🤦", "🙄"]
        },
        2: {  # Neutral
            "primary": ["😐", "🤔", "😶", "🫤"],
            "secondary": ["⚖️", "🤷", "📊", "📝"]
        },
        3: {  # Positive
            "primary": ["🙂", "😊", "👍", "😀"],
            "secondary": ["👌", "✅", "💯", "🔆"]
        },
        4: {  # Overwhelmingly positive
            "primary": ["😁", "🤩", "😍", "🥰"],
            "secondary": ["🎉", "🎊", "⭐", "💖", "🔥"]
        }
    }
    
    # Default to middle intensity if no percentage provided
    if sentiment_percentage is None:
        primary_emoji = emoji_mappings[category_index]["primary"][1]
        secondary_emoji = emoji_mappings[category_index]["secondary"][1]
        return {"primary": primary_emoji, "secondary": secondary_emoji}
    
    # Use percentage to determine intensity within the category
    primary_emojis = emoji_mappings[category_index]["primary"]
    secondary_emojis = emoji_mappings[category_index]["secondary"]
    
    # Map percentage to emoji index
    if category_index in [0, 1]:  # Negative categories
        # For negative emotions, higher percentage means more intense negative
        index = min(len(primary_emojis) - 1, int(sentiment_percentage / 25))
    elif category_index in [3, 4]:  # Positive categories
        # For positive emotions, higher percentage means more intense positive
        index = min(len(primary_emojis) - 1, int(sentiment_percentage / 25))
    else:  # Neutral category
        # For neutral, use fixed index
        index = 1
    
    primary_emoji = primary_emojis[index]
    secondary_emoji = secondary_emojis[min(index, len(secondary_emojis)-1)]
    
    return {"primary": primary_emoji, "secondary": secondary_emoji}

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    log_request('/analyze', data, status="Processing")

    if not data or 'text' not in data:
        error_msg = "Invalid input. Please provide 'text' in the request body."
        log_request('/analyze', data, status="400 Bad Request", error=error_msg)
        return jsonify({"error": error_msg}), 400

    text = data['text']
    
    try:
        # Simple rule-based sentiment analysis (placeholder for ML model)
        sentiment_percentage = get_sentiment_percentage(text)
        category_index = get_sentiment_category(sentiment_percentage)
        
        sentiment_labels = {
            0: "overwhelmingly negative",
            1: "negative",
            2: "neutral",
            3: "positive", 
            4: "overwhelmingly positive"
        }
        
        sentiment_label = sentiment_labels[category_index]
        
        # Get sentiment emojis based on category and percentage
        sentiment_emojis = get_sentiment_emojis(category_index, sentiment_percentage)

        # Add percentage to the label
        display_label = f"{sentiment_label} ({sentiment_percentage}%) {sentiment_emojis['primary']} {sentiment_emojis['secondary']}"

        # Prepare the response with detailed sentiment information
        response = {
            "text": text, 
            "prediction": int(category_index),
            "sentiment": display_label,
            "sentiment_percentage": sentiment_percentage,
            "sentiment_emojis": sentiment_emojis,
            "sentiment_category": {
                "0": "overwhelmingly negative",
                "1": "negative",
                "2": "neutral",
                "3": "positive", 
                "4": "overwhelmingly positive"
            }
        }
        
        log_request('/analyze', data, status="200 Success", error=None)
        return jsonify(response)
    except Exception as e:
        log_request('/analyze', data, status="500 Error", error=str(e))
        print(f"Error in sentiment analysis: {e}")
        return jsonify({"error": f"An error occurred during sentiment analysis: {str(e)}"}), 500

@app.route('/analyze_roberta', methods=['POST'])
def analyze_roberta():
    """Placeholder for RoBERTa model analysis"""
    # Redirect to the simple analyzer since we've removed the ML components
    return analyze()

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    log_request('/summarize', data, status="Processing")
    
    if not data or 'text' not in data:
        error_msg = "Invalid input. Please provide 'text' in the request body."
        log_request('/summarize', data, status="400 Bad Request", error=error_msg)
        return jsonify({"error": error_msg}), 400

    text = data['text']
    
    try:
        # Create a simple summary by taking the first 2 sentences or 100 chars
        sentences = text.split('.')
        simple_summary = '.'.join(sentences[:2]) + '.' if len(sentences) > 1 else text[:100]
        
        # Generate placeholder opinion based on the length of the text
        opinion = "This text appears to be " + ("short and concise." if len(text) < 200 else "detailed and informative.")
        
        enhanced_summary = (
            f"Summary: {simple_summary}\n\n"
            f"Model's Opinion: {opinion}\n"
            f"Reason: This is a simplified summary as the ML components have been removed."
        )
        
        response = {
            "original_text": text,
            "summary": enhanced_summary,
            "original_length": len(text),
            "summary_length": len(enhanced_summary),
            "processing_time_seconds": 0.1
        }
        
        log_request('/summarize', data, status="200 Success")
        return jsonify(response)
    except Exception as e:
        log_request('/summarize', data, status="500 Error", error=str(e))
        print(f"Error in text summarization: {e}")
        return jsonify({"error": f"An error occurred during summarization: {str(e)}"}), 500

@app.route("/api/sentiment", methods=["POST"])
def sentiment_analysis():
    try:
        data = request.get_json()
        if not validate_input(data):
            logging.warning("Invalid input received.")
            return jsonify({"error": "Invalid input."}), 400

        # Redirect to the simple analyzer
        return analyze()

    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        return jsonify({"error": "Internal server error."}), 500

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" MOOD MAP API SERVER ".center(80, "="))
    print("=" * 80)
    
    print("\n" + "*" * 80)
    print(" STARTING SERVER - HTTP MODE ".center(80, "*"))
    print("*" * 80)
    print("\n🚀 Server starting at http://127.0.0.1:5000")
    print("📢 IMPORTANT: The server is now running! Press Ctrl+C to stop.")
    print("🌐 Access the API at: http://127.0.0.1:5000")
    print("\n" + "*" * 80 + "\n")
    app.run(host="127.0.0.1", port=5000)