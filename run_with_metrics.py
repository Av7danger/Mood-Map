#!/usr/bin/env python
"""
Run the Mood Map backend with enhanced terminal output
showing model loading times and other metrics in real-time.
"""
import os
import sys
import time
import signal
import subprocess
import datetime
import threading
import re
from pathlib import Path

# Set colored terminal output
COLORS = {
    "RESET": "\033[0m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m"
}

def print_header():
    """Print a styled header for the terminal"""
    print(f"\n{COLORS['BOLD']}{COLORS['BLUE']}=" * 80)
    print(f"{COLORS['BOLD']}{COLORS['MAGENTA']}MOOD MAP BACKEND{COLORS['RESET']}{COLORS['BOLD']} - Enhanced Terminal Display")
    print(f"{COLORS['BLUE']}=" * 80 + f"{COLORS['RESET']}\n")

def print_section(title):
    """Print a section header"""
    print(f"\n{COLORS['BOLD']}{COLORS['CYAN']}▶ {title}{COLORS['RESET']}")
    print(f"{COLORS['CYAN']}-" * 50 + f"{COLORS['RESET']}")

def get_timestamp():
    """Get a formatted timestamp for logging"""
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

def format_time(seconds):
    """Format time in seconds to a readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes}m {seconds:.2f}s"

def parse_log_line(line):
    """Parse a log line for metrics of interest"""
    metrics = {}
    
    # Look for model loading time
    loading_match = re.search(r"(Ensemble|Attention|Neutral|Advanced) model (?:initialized|loaded) successfully in (\d+\.\d+) seconds", line)
    if loading_match:
        model_type = loading_match.group(1).lower()
        load_time = float(loading_match.group(2))
        metrics["model_load"] = {
            "model": model_type,
            "time": load_time
        }
    
    # Look for API request processing time
    request_match = re.search(r"\[RES-[^\]]+\] \w+ /\S+ \| Status: \d+ \| Time: (\d+\.\d+)s", line)
    if request_match:
        metrics["api_request_time"] = float(request_match.group(1))
    
    # Look for memory usage
    memory_match = re.search(r"memory_available_mb\": (\d+\.\d+)", line)
    if memory_match:
        metrics["memory_available"] = float(memory_match.group(1))
    
    return metrics

def display_model_loading_metrics():
    """Display a summary of model loading metrics"""
    model_metrics = {}
    total_start_time = time.time()
    
    # Wait for models to start loading
    print(f"{COLORS['YELLOW']}Waiting for model loading to begin...{COLORS['RESET']}")
    time.sleep(3)  # Give the backend process time to start
    
    loading_complete = False
    loading_in_progress = False
    
    # Continuously check for loading metrics until all models are loaded
    while not loading_complete and (time.time() - total_start_time < 300):  # Timeout after 5 minutes
        try:
            with open("logs/api_requests.log", "r") as log_file:
                # Read the last 50 lines (should contain recent loading info)
                lines = log_file.readlines()[-50:] if len(log_file.readlines()) > 50 else log_file.readlines()
                
                # Look for loading messages
                for line in lines:
                    metrics = parse_log_line(line)
                    if "model_load" in metrics:
                        model_info = metrics["model_load"]
                        model_type = model_info["model"]
                        load_time = model_info["time"]
                        
                        if model_type not in model_metrics:
                            loading_in_progress = True
                            print(f"{get_timestamp()} {COLORS['GREEN']}✓ {model_type.capitalize()} model loaded in {COLORS['BOLD']}{load_time:.2f}s{COLORS['RESET']}")
                            model_metrics[model_type] = load_time
                    
                    # Check for "Model loading complete!" message
                    if "Model loading complete!" in line:
                        loading_complete = True
                        break
                
                # If we found at least one model loaded but not all expected models,
                # continue waiting
                if loading_in_progress and not loading_complete:
                    expected_models = ["ensemble", "attention", "neutral", "advanced"]
                    loaded_models = list(model_metrics.keys())
                    missing_models = [model for model in expected_models if model not in loaded_models]
                    
                    if missing_models:
                        print(f"{COLORS['YELLOW']}Waiting for remaining models: {', '.join(missing_models)}{COLORS['RESET']}")
                    
        except FileNotFoundError:
            # Log file not created yet
            print(f"{COLORS['YELLOW']}Waiting for log file to be created...{COLORS['RESET']}")
        
        # Only sleep if we're still waiting for models
        if not loading_complete:
            time.sleep(2)
    
    print_section("MODEL LOADING SUMMARY")
    
    if model_metrics:
        # Print model loading summary
        total_load_time = sum(model_metrics.values())
        print(f"{COLORS['BOLD']}Total models loaded: {len(model_metrics)}{COLORS['RESET']}")
        print(f"{COLORS['BOLD']}Total loading time: {format_time(total_load_time)}{COLORS['RESET']}")
        print("\nIndividual model loading times:")
        
        # Sort models by loading time (descending)
        sorted_models = sorted(model_metrics.items(), key=lambda x: x[1], reverse=True)
        
        for model, load_time in sorted_models:
            color = COLORS['RED'] if load_time > 10 else COLORS['YELLOW'] if load_time > 5 else COLORS['GREEN']
            print(f"  - {model.capitalize():10} : {color}{load_time:.2f}s{COLORS['RESET']}")
    
    elif time.time() - total_start_time >= 300:
        print(f"{COLORS['RED']}Timed out waiting for model loading to complete.{COLORS['RESET']}")
    else:
        print(f"{COLORS['YELLOW']}No model loading metrics found in the logs.{COLORS['RESET']}")

def monitor_backend(process):
    """Monitor the backend process and display metrics"""
    display_model_loading_metrics()
    
    # After model loading, start monitoring API requests
    print_section("API MONITORING (Press Ctrl+C to stop)")
    
    start_time = time.time()
    request_times = []
    
    try:
        while process.poll() is None:
            try:
                # Read the last few lines of the log file
                with open("logs/api_requests.log", "r") as log_file:
                    # Read the last 10 lines (should contain recent requests)
                    lines = log_file.readlines()[-10:] if len(log_file.readlines()) > 10 else log_file.readlines()
                    
                    for line in lines:
                        metrics = parse_log_line(line)
                        
                        if "api_request_time" in metrics:
                            request_time = metrics["api_request_time"]
                            request_times.append(request_time)
                            
                            # Only keep the last 20 requests for average calculation
                            if len(request_times) > 20:
                                request_times.pop(0)
                            
                            avg_time = sum(request_times) / len(request_times)
                            
                            # Only print if the request took more than 0.1s
                            if request_time > 0.1:
                                color = COLORS['RED'] if request_time > 1.0 else COLORS['YELLOW'] if request_time > 0.5 else COLORS['GREEN']
                                print(f"{get_timestamp()} API request processed in {color}{request_time:.3f}s{COLORS['RESET']} (avg: {avg_time:.3f}s)")
                
            except FileNotFoundError:
                # Log file not created yet
                pass
            
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print(f"\n{COLORS['YELLOW']}Monitoring stopped. Backend still running.{COLORS['RESET']}")
        return

def run_backend():
    """Run the backend with enhanced terminal output"""
    print_header()
    
    # Get the path to the sentiment API
    api_script = os.path.join(os.getcwd(), "backend", "sentiment_api.py")
    
    if not os.path.exists(api_script):
        print(f"{COLORS['RED']}Error: Could not find backend script at {api_script}{COLORS['RESET']}")
        print(f"Make sure you're running this script from the mood-map root directory.")
        return
    
    print_section("STARTING BACKEND")
    print(f"Starting backend from: {api_script}")
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Start the API in a subprocess
    process = subprocess.Popen(
        [sys.executable, api_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )
    
    print(f"{COLORS['GREEN']}Backend process started with PID: {process.pid}{COLORS['RESET']}")
    
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_backend, args=(process,))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Also display log output
    print_section("BACKEND LOG OUTPUT")
    
    try:
        # Continuously read and display stdout/stderr from the process
        while process.poll() is None:
            stdout_line = process.stdout.readline()
            if stdout_line:
                print(f"{COLORS['BLUE']}[LOG] {COLORS['RESET']}{stdout_line.strip()}")
            
            stderr_line = process.stderr.readline()
            if stderr_line:
                print(f"{COLORS['RED']}[ERR] {COLORS['RESET']}{stderr_line.strip()}")
            
            # Small sleep to reduce CPU usage
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print(f"\n{COLORS['YELLOW']}Received interrupt signal. Shutting down...{COLORS['RESET']}")
        
        try:
            # Send SIGTERM to the process and its children
            os.kill(process.pid, signal.SIGTERM)
            print(f"{COLORS['GREEN']}Sent termination signal to backend process.{COLORS['RESET']}")
        except:
            print(f"{COLORS['RED']}Failed to terminate the backend process gracefully.{COLORS['RESET']}")
            process.terminate()
    
    finally:
        # Wait for the process to finish
        exit_code = process.wait()
        
        if exit_code == 0:
            print(f"\n{COLORS['GREEN']}Backend process exited successfully.{COLORS['RESET']}")
        else:
            print(f"\n{COLORS['RED']}Backend process exited with code: {exit_code}{COLORS['RESET']}")

if __name__ == "__main__":
    run_backend()