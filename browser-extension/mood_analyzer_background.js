// Background service worker for MoodMap extension
console.log("MoodMap background script initializing...");

// Global error handler for extension context invalidation
let extensionContextValid = true;

// Listen for extension context invalidation
chrome.runtime.onSuspend.addListener(() => {
  console.log("Extension context being suspended, marking as invalid");
  extensionContextValid = false;
});

// Wrapper for chrome API calls to check for invalid context
function safeChromeApiCall(apiCall) {
  return function(...args) {
    if (!extensionContextValid) {
      console.warn("Extension context invalidated, skipping API call");
      return Promise.reject(new Error("Extension context invalidated"));
    }
    
    try {
      return apiCall(...args);
    } catch (e) {
      if (e.message.includes("Extension context invalidated")) {
        extensionContextValid = false;
        console.warn("Extension context invalidated during API call");
      }
      throw e;
    }
  };
}

// Wrap chrome API calls that could fail due to extension context invalidation
// Define these before using them below
const safeStorageGet = (keys, callback) => {
  try {
    chrome.storage.local.get(keys, (result) => {
      if (chrome.runtime.lastError) {
        console.error("Storage get error:", chrome.runtime.lastError);
        if (chrome.runtime.lastError.message.includes("Extension context invalidated")) {
          extensionContextValid = false;
        }
        callback({});
      } else {
        callback(result);
      }
    });
  } catch (error) {
    console.error("Error in storage get:", error);
    if (error.message.includes("Extension context invalidated")) {
      extensionContextValid = false;
    }
    callback({});
  }
};

const safeStorageSet = (items, callback) => {
  try {
    chrome.storage.local.set(items, () => {
      if (chrome.runtime.lastError) {
        console.error("Storage set error:", chrome.runtime.lastError);
        if (chrome.runtime.lastError.message.includes("Extension context invalidated")) {
          extensionContextValid = false;
        }
      }
      if (callback) callback();
    });
  } catch (error) {
    console.error("Error in storage set:", error);
    if (error.message.includes("Extension context invalidated")) {
      extensionContextValid = false;
    }
    if (callback) callback();
  }
};

const safeTabsSendMessage = (tabId, message) => {
  return new Promise((resolve, reject) => {
    try {
      chrome.tabs.sendMessage(tabId, message, (response) => {
        if (chrome.runtime.lastError) {
          console.warn("Tab message error:", chrome.runtime.lastError);
          reject(chrome.runtime.lastError);
        } else {
          resolve(response);
        }
      });
    } catch (error) {
      console.error("Error sending tab message:", error);
      reject(error);
    }
  });
};

// Debug network request function - defined at the top to avoid reference errors
function debugNetworkRequest(details) {
  console.log("Network request:", details);
  return details;
}

// Global error handler for unhandled exceptions
self.onerror = function(message, source, lineno, colno, error) {
  console.error("Unhandled error in background script:", message, error);
  
  // If the error is about extension context, mark it as invalid
  if (message && message.includes && message.includes("Extension context invalidated")) {
    extensionContextValid = false;
    console.warn("Extension context has been invalidated, will require reload");
  }
  
  return true;
};

// Listen for requests from content scripts and popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  try {
    console.log("Background script received message:", request);
    
    // Check if extension context is valid before proceeding
    if (!extensionContextValid) {
      console.warn("Extension context invalidated, cannot process message");
      sendResponse({ error: "Extension context invalidated. Please reload the extension." });
      return false;
    }

    // Respond to ping messages immediately (for testing communication)
    if (request.type === 'ping') {
      console.log("Received ping, sending response");
      sendResponse({ type: 'pong', status: 'alive', timestamp: Date.now() });
      return true;
    }
    
    // Handle different message types
    if (request.type === 'analyzeSentiment') {
      // Process sentiment analysis request
      analyzeSentiment(request.text, request.options)
        .then(result => {
          console.log("Analysis result:", result);
          if (extensionContextValid) {
            sendResponse(result);
          }
        })
        .catch(error => {
          console.error("Error in sentiment analysis:", error);
          if (extensionContextValid) {
            sendResponse({ error: error.message });
          }
        });
      
      // Return true to indicate we'll send a response asynchronously
      return true;
    }
    
    else if (request.type === 'updateDefaultModel') {
      // Update the default model setting
      safeStorageSet({ selectedModel: request.model }, () => {
        if (extensionContextValid) {
          sendResponse({ success: true, model: request.model });
        }
      });
      
      // Return true to indicate we'll send a response asynchronously
      return true;
    }
    
    else if (request.type === 'getNetworkRequest') {
      // This is a debug function to get network request data
      sendResponse({ 
        success: true, 
        message: "Network request monitoring is active"
      });
      
      return true;
    }
    
    else if (request.type === 'getBackgroundStatus') {
      // Used to check if background script is running properly
      getStoredSettings().then(settings => {
        if (extensionContextValid) {
          sendResponse({
            status: "OK",
            timestamp: Date.now(),
            apiUrl: settings.apiUrl,
            model: settings.selectedModel,
            apiStatus: settings.apiStatus
          });
        }
      });
      return true;
    }
    
    // For unhandled message types
    sendResponse({ error: "Unhandled message type", type: request.type });
    return true;
  } catch (error) {
    console.error("Error handling message:", error);
    
    // Check for extension context invalidation
    if (error.message.includes("Extension context invalidated")) {
      extensionContextValid = false;
      console.warn("Extension context has been invalidated");
    }
    
    // Try to send error response
    try {
      sendResponse({ error: error.message });
    } catch (responseError) {
      console.error("Could not send error response:", responseError);
    }
    
    return false;
  }
});

// Create context menu for analyzing selected text
chrome.runtime.onInstalled.addListener(() => {
  try {
    chrome.contextMenus.create({
      id: "analyzeSentiment",
      title: "Analyze sentiment with MoodMap",
      contexts: ["selection"]
    });
    
    console.log("MoodMap: Created context menu for sentiment analysis");
    
    // Initialize default settings if not already set
    safeStorageGet(['apiUrl', 'selectedModel'], (data) => {
      if (!data.apiUrl) {
        safeStorageSet({ apiUrl: 'http://localhost:5000' });
      }
      if (!data.selectedModel) {
        safeStorageSet({ selectedModel: 'simple' });
      }
      console.log("Initialized default settings:", data);
    });

    // Preload all available models
    getStoredSettings().then(settings => {
      preloadAllModels(settings.apiUrl);
    });
  } catch (e) {
    console.error("Error during installation:", e);
  }
});

// Listen for context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "analyzeSentiment" && info.selectionText) {
    // Send the selected text to the content script for analysis
    safeTabsSendMessage(tab.id, {
      type: "analyzeSelectedText",
      text: info.selectionText
    }).catch(error => {
      console.error("Error sending message to content script:", error);
      // If content script is not available, try to analyze directly
      analyzeAndShowNotification(info.selectionText);
    });
    
    console.log("Sent selected text for analysis:", info.selectionText.substring(0, 50) + "...");
  }
});

// Function to show analysis result in a notification if content script isn't available
function analyzeAndShowNotification(text) {
  analyzeSentiment(text).then(result => {
    let sentimentText = "Neutral";
    if (result.sentiment < -0.3) sentimentText = "Negative";
    else if (result.sentiment > 0.3) sentimentText = "Positive";
    
    chrome.notifications.create({
      type: "basic",
      iconUrl: chrome.runtime.getURL("assets/icon128.png"),
      title: "MoodMap Sentiment Analysis",
      message: `The text has a ${sentimentText} sentiment (Score: ${result.sentiment.toFixed(2)})`
    });
  });
}

// Function to analyze sentiment using the API or local processing
async function analyzeSentiment(text, options = {}) {
  console.log("Analyzing sentiment for text:", text.substring(0, 50) + "...", "options:", options);
  
  try {
    // Get settings from storage
    const { apiUrl, selectedModel, apiStatus } = await getStoredSettings();
    
    // Get model to use - either from options or from stored settings
    const modelToUse = options.model || selectedModel;
    
    // Optimize for short text analysis - always use simple model for very short text
    if (text.length < 30 && !text.includes('?') && !options.summarize) {
      console.log('Using offline processing with simple model (short text optimization)');
      return processSimpleAnalysis(text);
    }
    
    // Only use simple model if explicitly selected and not summarizing
    if (modelToUse === 'simple' && !options.summarize) {
      console.log('Using offline processing with simple model (user selected)');
      return processSimpleAnalysis(text);
    }
    
    // If API is known to be offline and summarization is not requested, fall back to simple model
    if (apiStatus === 'offline' && !options.summarize) {
      console.log('API is offline, falling back to simple model');
      return processSimpleAnalysis(text);
    }
    
    console.log(`Using API with model: ${modelToUse}${options.summarize ? ' with summarization' : ''}`);
    
    // New: Use specialized endpoint for simple model when appropriate
    const cleanApiUrl = apiUrl.endsWith('/') ? apiUrl.slice(0, -1) : apiUrl;
    let endpoint = `${cleanApiUrl}/analyze`;
    
    // For simple model without summarization, use the specialized fast endpoint
    if (modelToUse === 'simple' && !options.summarize) {
      endpoint = `${cleanApiUrl}/extension/analyze_simple`;
    }
    
    // If summarization is requested, use the advanced endpoint
    if (options.summarize) {
      endpoint = `${cleanApiUrl}/extension/analyze_advanced`;
    }
    
    // Make API request - the server will lazy-load the required model
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        text: text,
        model_type: modelToUse,
        summarize: options.summarize || false,
        features: {
          sentiment: true,
          summarization: options.summarize || false
        }
      })
    });
    
    if (!response.ok) {
      throw new Error(`API returned ${response.status}: ${response.statusText}`);
    }
    
    const result = await response.json();
    console.log("API returned result:", result);
    
    // Update API status to online since we got a successful response
    safeStorageSet({ apiStatus: 'online' });
    
    // Add loading info if available
    if (result.model_loading_time) {
      result.modelWasJustLoaded = true;
      result.loadingTimeSeconds = result.model_loading_time;
    }
    
    return result;
    
  } catch (error) {
    console.error("Error analyzing sentiment:", error);
    
    // Mark API as offline if we couldn't connect
    if (error.message.includes('Failed to fetch') || 
        error.message.includes('NetworkError') ||
        error.message.includes('ECONNREFUSED')) {
      console.log('API connection failed, marking as offline');
      safeStorageSet({ apiStatus: 'offline' });
    }
    
    // Only fall back to simple analysis when not summarizing
    if (!options.summarize) {
      console.log("Falling back to offline processing due to error");
      return processSimpleAnalysis(text);
    }
    
    // Return error response for summarization requests
    return {
      error: "Could not connect to backend for advanced analysis and summarization",
      sentiment: 0,
      category: 1,
      label: "neutral",
      confidence: 0.5
    };
  }
}

// Simple offline text analysis
function processSimpleAnalysis(text) {
  console.log("Processing with simple offline analysis");
  text = text.toLowerCase();
  
  // Check for obviously positive sentiment
  const positiveWords = ["love", "amazing", "excellent", "fantastic", "great", "awesome", "happy", "joy"];
  const negativeWords = ["hate", "terrible", "awful", "horrible", "worst", "bad", "sad", "angry"];
  
  let positiveCount = 0;
  let negativeCount = 0;
  
  positiveWords.forEach(word => {
    if (text.includes(word)) positiveCount++;
  });
  
  negativeWords.forEach(word => {
    if (text.includes(word)) negativeCount++;
  });
  
  let sentiment = 0;
  let category = 1;
  let label = 'neutral';
  
  if (positiveCount > negativeCount) {
    sentiment = 0.5 + (0.5 * (positiveCount / (positiveCount + negativeCount)));
    category = 2;
    label = 'positive';
  } else if (negativeCount > positiveCount) {
    sentiment = -0.5 - (0.5 * (negativeCount / (positiveCount + negativeCount)));
    category = 0;
    label = 'negative';
  }
  
  return {
    sentiment: sentiment,
    category: category,
    label: label,
    confidence: 0.6,
    emotions: {}
  };
}

// Helper function to safely get stored settings
async function getStoredSettings() {
  try {
    const data = await new Promise((resolve, reject) => {
      safeStorageGet(['apiUrl', 'selectedModel', 'apiStatus'], (result) => {
        resolve(result);
      });
    });
    
    return {
      apiUrl: data.apiUrl || 'http://localhost:5000',
      selectedModel: data.selectedModel || 'simple',
      apiStatus: data.apiStatus || 'unknown'
    };
  } catch (error) {
    console.error("Error getting stored settings:", error);
    
    // If there's a context invalidated error, mark the context
    if (error.message && error.message.includes("Extension context invalidated")) {
      extensionContextValid = false;
    }
    
    // Return defaults
    return {
      apiUrl: 'http://localhost:5000',
      selectedModel: 'simple',
      apiStatus: 'unknown'
    };
  }
}

// Monitor network requests for debugging
try {
  chrome.webRequest.onSendHeaders.addListener(
    debugNetworkRequest,
    { urls: ["*://localhost/*", "*://127.0.0.1/*"] }
  );
} catch (e) {
  console.error("Error setting up webRequest listener:", e);
}

// Set periodic health check
setInterval(() => {
  console.log("Running periodic health check");
  
  // Check API health if necessary
  getStoredSettings().then(settings => {
    // Only check API if not using simple model
    if (settings.selectedModel !== 'simple') {
      checkApiHealth(settings.apiUrl);
    }
  });
}, 5 * 60 * 1000); // Every 5 minutes

// Function to check API health with enhanced status information
async function checkApiHealth(apiUrl) {
  try {
    const cleanApiUrl = apiUrl.endsWith('/') ? apiUrl.slice(0, -1) : apiUrl;
    
    // Use the new extension-specific status endpoint that provides detailed information
    const response = await fetch(`${cleanApiUrl}/extension/status`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        check_offline_available: true,
        require_advanced_features: false,
        client_info: {
          client_type: "browser_extension",
          version: chrome.runtime.getManifest().version,
          platform: "web"
        }
      })
    });
    
    if (response.ok) {
      const statusData = await response.json();
      console.log("API enhanced status:", statusData);
      
      // Store detailed API status information
      safeStorageSet({ 
        apiStatus: 'online',
        lastHealthCheckTime: Date.now(),
        availableModels: statusData.available_models || [],
        loadedModels: statusData.loaded_models || [],
        modelLoadingStatus: statusData.model_loading_status || {},
        recommendedModel: statusData.recommended_model || 'simple',
        apiVersion: statusData.api_version
      });
      
      // If API recommends a different model than currently selected, consider switching
      getStoredSettings().then(settings => {
        if (settings.selectedModel !== 'simple' && 
            !statusData.loaded_models.includes(settings.selectedModel) && 
            statusData.recommended_model && 
            statusData.recommended_model !== settings.selectedModel) {
          
          console.log(`API recommends switching from ${settings.selectedModel} to ${statusData.recommended_model}`);
          
          // Don't auto-switch to simple if user selected a more advanced model
          // This avoids downgrading the user experience without their consent
          if (statusData.recommended_model !== 'simple') {
            safeStorageSet({ 
              recommendedModelSwitch: statusData.recommended_model,
              recommendedModelReason: "API recommendation for better performance"
            });
          }
        }
      });
      
    } else {
      // Fall back to basic status
      safeStorageSet({ apiStatus: 'offline' });
      console.log("API health check failed with status:", response.status);
    }
  } catch (error) {
    safeStorageSet({ apiStatus: 'offline' });
    console.error("API health check error:", error);
  }
}

// Function to preload all available models from the API server
async function preloadAllModels(apiUrl) {
  try {
    console.log("Starting to preload all sentiment analysis models...");
    const cleanApiUrl = apiUrl.endsWith('/') ? apiUrl.slice(0, -1) : apiUrl;
    
    // Step 1: Check which models are available via the extension status endpoint
    const statusResponse = await fetch(`${cleanApiUrl}/extension/status`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        check_offline_available: true,
        require_advanced_features: true
      })
    });
    
    if (!statusResponse.ok) {
      throw new Error(`API status check failed: ${statusResponse.status}`);
    }
    
    const statusData = await statusResponse.json();
    console.log("Available models:", statusData.available_models);
    
    // Models to load (exclude 'simple' as it's always available)
    const modelsToLoad = statusData.available_models.filter(model => 
      model !== 'simple' && !statusData.loaded_models.includes(model)
    );
    
    if (modelsToLoad.length === 0) {
      console.log("All models are already loaded, no preloading needed");
      return { success: true, status: "all_models_already_loaded" };
    }
    
    console.log(`Models to preload: ${modelsToLoad.join(', ')}`);
    
    // Step 2: Request each model to load one by one
    const results = {};
    
    for (const model of modelsToLoad) {
      console.log(`Preloading model: ${model}`);
      
      try {
        // Use the load-model endpoint to explicitly load the model
        const loadResponse = await fetch(`${cleanApiUrl}/load-model`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            model_type: model,
            wait_for_loading: false  // Don't block on loading, start it in background
          })
        });
        
        if (loadResponse.ok) {
          const loadResult = await loadResponse.json();
          console.log(`Model ${model} loading triggered:`, loadResult);
          results[model] = loadResult.status || "loading_started";
        } else {
          console.warn(`Failed to load model ${model}: ${loadResponse.status}`);
          results[model] = "load_failed";
        }
      } catch (error) {
        console.error(`Error loading model ${model}:`, error);
        results[model] = "error";
      }
      
      // Short delay between model loading requests to avoid overloading the server
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    // Store the model loading status
    safeStorageSet({ 
      modelLoadingStatus: results,
      lastModelLoadingTime: Date.now()
    });
    
    return {
      success: true,
      status: "loading_triggered",
      models: results
    };
  } catch (error) {
    console.error("Error preloading models:", error);
    return {
      success: false,
      error: error.message
    };
  }
}

// Run initial health check
getStoredSettings().then(settings => {
  checkApiHealth(settings.apiUrl);
});

// Log initialization complete
console.log("MoodMap background script initialized successfully");