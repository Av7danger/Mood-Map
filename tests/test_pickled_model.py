#!/usr/bin/env python
"""
Script to test the pickled sentiment analysis model.
This script loads the model.pkl file directly and tests it on sample texts.
"""
import os
import sys
import pickle
import time
from pathlib import Path
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

print("=" * 60)
print("PICKLED SENTIMENT MODEL TEST TOOL")
print("=" * 60)
print(f"Project root: {project_root}")

# Define paths to check for model.pkl
model_paths = [
    os.path.join(project_root, "model.pkl"),  # Root folder
    os.path.join(project_root, "backend", "model.pkl")  # Backend folder
]

# Test data for sentiment analysis
test_texts = [
    "I love this product! It's amazing and works perfectly.",  # Positive
    "This is the worst experience ever. I hate it.",           # Negative
    "The package arrived on time and contained all items.",    # Neutral
    "This movie was great but the ending could have been better.", # Mixed
    "I'm not sure if I like this or not.",                     # Uncertain
]

# Expected sentiments
expected_sentiments = ["positive", "negative", "neutral", "mixed", "neutral"]

# Function to test a model on the test data
def test_model(model, model_type="scikit-learn"):
    print(f"\nTesting model (type: {model_type})...")
    
    results = []
    
    for i, text in enumerate(test_texts):
        try:
            start_time = time.time()
            
            # Different handling based on model type
            if hasattr(model, 'predict'):
                # Most scikit-learn models expect this format
                prediction = model.predict([text])[0]
                
                # Try to get probability if available
                confidence = 0.0
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba([text])[0]
                        confidence = max(proba)
                except Exception as e:
                    print(f"  Note: Could not get probability: {e}")
                
                # Map prediction to human-readable label
                # Assume 0=negative, 1=neutral, 2=positive if numeric
                label = prediction
                if isinstance(prediction, (int, np.integer)):
                    label_map = {0: "negative", 1: "neutral", 2: "positive"}
                    # Binary classifiers might use 0/1 instead of 0/2
                    if prediction == 1 and 2 not in label_map:
                        label = "positive"
                    else:
                        label = label_map.get(prediction, f"unknown-{prediction}")
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                print(f"  Text {i+1}: '{text[:50]}...'")
                print(f"    → Predicted: {label} (raw: {prediction}, confidence: {confidence:.4f})")
                print(f"    → Expected:  {expected_sentiments[i]}")
                print(f"    → Time:      {processing_time*1000:.2f}ms")
                
                results.append({
                    "text": text,
                    "prediction": label,
                    "raw_prediction": prediction,
                    "confidence": confidence,
                    "time": processing_time,
                    "expected": expected_sentiments[i]
                })
                
            elif callable(model):
                # Function-like model
                result = model(text)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                print(f"  Text {i+1}: '{text[:50]}...'")
                print(f"    → Predicted: {result}")
                print(f"    → Expected:  {expected_sentiments[i]}")
                print(f"    → Time:      {processing_time*1000:.2f}ms")
                
                results.append({
                    "text": text,
                    "prediction": result,
                    "time": processing_time,
                    "expected": expected_sentiments[i]
                })
                
            else:
                print(f"  ❌ Unsupported model type: {type(model)}")
                return False
                
        except Exception as e:
            print(f"  ❌ Error processing text {i+1}: {e}")
    
    # Calculate accuracy
    correct = 0
    for r in results:
        pred = r["prediction"]
        expected = r["expected"]
        
        # Compare predictions, handling string vs numeric cases
        is_correct = False
        if isinstance(pred, str) and isinstance(expected, str):
            is_correct = pred.lower() == expected.lower()
        elif isinstance(pred, (int, np.integer)) and expected == "positive":
            is_correct = pred > 0
        elif isinstance(pred, (int, np.integer)) and expected == "negative":
            is_correct = pred < 0
        
        if is_correct:
            correct += 1
    
    accuracy = correct / len(test_texts) if test_texts else 0
    print(f"\nAccuracy: {accuracy*100:.1f}% ({correct}/{len(test_texts)} correct)")
    
    # Calculate average processing time
    avg_time = sum(r["time"] for r in results) / len(results) if results else 0
    print(f"Average processing time: {avg_time*1000:.2f}ms per text")
    
    return accuracy >= 0.6  # Consider it successful if accuracy is at least 60%

# Main execution
print("\nSearching for pickled model...")
model_found = False

for path in model_paths:
    if os.path.exists(path):
        print(f"✅ Found model at: {path}")
        try:
            # Load the model
            print(f"Loading model from {path}...")
            start_time = time.time()
            with open(path, 'rb') as f:
                model = pickle.load(f)
            load_time = time.time() - start_time
            
            print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
            print(f"Model type: {type(model)}")
            
            # Attempt to get more info about the model
            if hasattr(model, 'classes_'):
                print(f"Model classes: {model.classes_}")
            if hasattr(model, 'n_features_in_'):
                print(f"Number of features: {model.n_features_in_}")
            
            # Test the model
            success = test_model(model)
            if success:
                print("\n✅ Model testing completed successfully!")
            else:
                print("\n⚠️ Model testing completed with issues")
            
            model_found = True
            break
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print(f"❌ No model found at: {path}")

if not model_found:
    print("\n❌ No pickled model found!")
    print("Please make sure model.pkl exists in the project root or backend directory.")
    sys.exit(1)

# Try to check if the model is being used in the streamlined API
try:
    import requests
    print("\nTesting model through the streamlined API...")
    
    # Try to connect to the API
    api_running = False
    
    for port in [5000, 5001, 5002]:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                api_running = True
                api_port = port
                print(f"✅ API server found running on port {port}")
                break
        except:
            continue
    
    if api_running:
        # Test with the pickled model
        test_text = "I really love this new product!"
        try:
            response = requests.post(
                f"http://localhost:{api_port}/analyze", 
                json={"text": test_text, "model_type": "pickled"},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"API test result:")
                print(f"  Text: '{test_text}'")
                print(f"  Sentiment: {result.get('label')}")
                print(f"  Score: {result.get('score')}")
                print(f"  Confidence: {result.get('confidence')}")
                print(f"  Model used: {result.get('model_used')}")
                
                # Check if it's using the pickled model
                if "pickled" in result.get('model_used', ''):
                    print("\n✅ Streamlined API is successfully using the pickled model!")
                else:
                    print("\n⚠️ API responded but may not be using the pickled model.")
                    print(f"   Model reported: {result.get('model_used')}")
            else:
                print(f"\n❌ API returned error: {response.status_code}")
        except Exception as e:
            print(f"\n❌ Error testing API: {e}")
    else:
        print("\n⚠️ Could not connect to API server. Make sure it's running to test integration.")
except ImportError:
    print("\n⚠️ Could not import requests library. Skipping API integration test.")
except Exception as e:
    print(f"\n❌ Error during API testing: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)