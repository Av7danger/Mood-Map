import os
import sys
import time
import subprocess
import webbrowser
import argparse

def print_section(title):
    """Print a section header to make output more readable."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def verify_python_environment():
    """Verify that the Python environment is properly set up."""
    print_section("Verifying Python Environment")
    
    try:
        import torch
        import transformers
        import flask
        import pandas
        import numpy
        import joblib
        
        print("✅ All required Python packages are installed.")
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"✅ CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA is not available. Training will run on CPU (slower).")
        
        return True
    
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install all required packages using: pip install -r requirements.txt")
        return False

def retrain_model(dataset_path=None, use_subset=True, epochs=10, batch_size=32):
    """Retrain the sentiment model with the specified parameters."""
    print_section("Retraining Sentiment Model")
    
    # Build the command
    command = [sys.executable, "src/training/train_sentiment_model.py"]
    
    # Add arguments if specified
    if dataset_path:
        command.extend(["--dataset_path", dataset_path])
    if not use_subset:
        command.append("--full_dataset")
    if epochs != 10:
        command.extend(["--epochs", str(epochs)])
    if batch_size != 32:
        command.extend(["--batch_size", str(batch_size)])
    
    print(f"Running command: {' '.join(command)}")
    print("Training may take a while, please be patient...\n")
    
    # Set up the environment with the proper Python path
    env = os.environ.copy()
    # Add the current directory to PYTHONPATH so 'src' module can be found
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = os.getcwd()
    
    # Using subprocess.Popen with input to automatically provide 'yes' to the prompt
    try:
        process = subprocess.Popen(
            command,
            text=True,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )
        
        # Function to handle output 
        def read_output():
            for line in iter(process.stdout.readline, ''):
                print(line.rstrip())
                if "Use new dataset? (yes/no):" in line:
                    process.stdin.write("yes\n")
                    process.stdin.flush()
            
        # Start reading output in a separate thread
        import threading
        thread = threading.Thread(target=read_output)
        thread.daemon = True
        thread.start()
        
        # Wait for the process to finish
        returncode = process.wait()
        thread.join(1.0)  # Wait up to 1 second for the thread to finish
        
        if returncode == 0:
            print("\n✅ Model training completed successfully.")
            return True
        else:
            print("\n❌ Model training failed.")
            return False
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return False

def copy_model_to_backend():
    """Copy the trained model to the backend directory."""
    print_section("Copying Model to Backend")
    
    import shutil
    
    try:
        # Check if the model exists
        src_model = os.path.join("model.pkl")
        if not os.path.exists(src_model):
            src_model = os.path.join("src", "models", "model.pkl")
            if not os.path.exists(src_model):
                print("❌ Could not find trained model file.")
                return False
        
        # Copy the model to the backend directory
        dst_model = os.path.join("backend", "model.pkl")
        shutil.copy2(src_model, dst_model)
        print(f"✅ Model copied to {dst_model}")
        return True
    
    except Exception as e:
        print(f"❌ Error copying model: {e}")
        return False

def start_backend_server():
    """Start the Flask backend server."""
    print_section("Starting Backend API Server")
    
    # Check if the model exists
    model_path = os.path.join("backend", "model.pkl")
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        print("Please make sure the model has been trained and copied correctly.")
        return False
    
    # Check for config file
    config_path = os.path.join("config", "sentiment_api_config.json")
    if not os.path.exists(config_path):
        print(f"⚠️ Config file not found at {config_path}")
        print("Using default configuration.")
    else:
        # Copy config to backend directory
        import shutil
        backend_config_path = os.path.join("backend", "sentiment_api_config.json")
        shutil.copy2(config_path, backend_config_path)
        print(f"✅ Copied config file to {backend_config_path}")
    
    # Check for certificate files
    cert_path = os.path.join("backend", "cert.pem")
    key_path = os.path.join("backend", "key.pem")
    ssl_available = False
    
    if not (os.path.exists(cert_path) and os.path.exists(key_path)):
        print("⚠️ SSL certificate files not found.")
        try:
            # Try to import OpenSSL
            try:
                from OpenSSL import crypto
                ssl_available = True
            except ImportError:
                print("⚠️ OpenSSL module not found. To enable HTTPS, install it with:")
                print("   pip install pyopenssl")
                print("⚠️ Continuing without SSL (HTTP only mode)...")
            
            # Create certificates only if OpenSSL is available
            if ssl_available:
                print("Creating self-signed certificates...")
                # Create a key pair
                k = crypto.PKey()
                k.generate_key(crypto.TYPE_RSA, 2048)
                
                # Create a self-signed cert
                cert = crypto.X509()
                cert.get_subject().C = "US"
                cert.get_subject().ST = "California"
                cert.get_subject().L = "Silicon Valley"
                cert.get_subject().O = "Mood Map"
                cert.get_subject().OU = "Development"
                cert.get_subject().CN = "localhost"
                cert.set_serial_number(1000)
                cert.gmtime_adj_notBefore(0)
                cert.gmtime_adj_notAfter(365*24*60*60)  # Valid for one year
                cert.set_issuer(cert.get_subject())
                cert.set_pubkey(k)
                cert.sign(k, 'sha256')
                
                # Save the certificate and private key
                with open(cert_path, "wb") as f:
                    f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
                
                with open(key_path, "wb") as f:
                    f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
                    
                print("✅ Created self-signed certificates")
        except Exception as e:
            print(f"❌ Error creating certificates: {e}")
            print("Running server without SSL...")
    
    # Start the server as a subprocess
    print("\nStarting the backend API server...")
    print("The server will run in the background.")
    print("Press Ctrl+C to stop the server when you're done.")
    
    server_env = os.environ.copy()
    server_env["FLASK_ENV"] = "development"
    server_env["PYTHONPATH"] = os.getcwd()  # Add the current directory to PYTHONPATH
    
    # CD into the backend directory to ensure paths work correctly
    os.chdir("backend")
    
    try:
        server_process = subprocess.Popen(
            [sys.executable, "sentiment_api.py"],
            env=server_env
        )
        print(f"\n✅ Server started with PID: {server_process.pid}")
        
        if ssl_available and os.path.exists(cert_path) and os.path.exists(key_path):
            print("The API is available at https://localhost:5000")
        else:
            print("The API is available at http://localhost:5000")
        
        return server_process
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Mood Map - Retrain model and run everything")
    parser.add_argument("--dataset", choices=["sentimentdataset", "large"],
                        help="Dataset to use for training (sentimentdataset.csv or the large training dataset)")
    parser.add_argument("--full", action="store_true", help="Use the full dataset instead of a subset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--skip_training", action="store_true", help="Skip model training")
    
    args = parser.parse_args()
    
    # Verify Python environment
    if not verify_python_environment():
        sys.exit(1)
    
    # Retrain the model if not skipping
    if not args.skip_training:
        # Determine dataset path
        dataset_path = None
        if args.dataset == "sentimentdataset":
            dataset_path = os.path.join("data", "raw", "sentimentdataset.csv")
        elif args.dataset == "large":
            dataset_path = os.path.join("data", "raw", "training.1600000.processed.noemoticon.csv")
        
        # Retrain the model
        if not retrain_model(dataset_path, not args.full, args.epochs, args.batch_size):
            print("❌ Model training failed. Exiting.")
            sys.exit(1)
        
        # Copy the model to the backend directory
        if not copy_model_to_backend():
            print("❌ Failed to copy model to backend directory. Exiting.")
            sys.exit(1)
    
    # Start the backend server
    server_process = start_backend_server()
    if not server_process:
        print("❌ Failed to start the backend server. Exiting.")
        sys.exit(1)
    
    try:
        # Give the server some time to start
        time.sleep(3)
        
        # Open the browser
        print("\nOpening browser to access the API...")
        webbrowser.open("http://localhost:5000")
        
        # Keep the script running until user interrupts
        print("\nPress Ctrl+C to stop the server and exit.")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        if server_process:
            server_process.terminate()
        print("Goodbye!")

if __name__ == "__main__":
    main()