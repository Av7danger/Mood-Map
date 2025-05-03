// MoodMap Browser Extension Background Script
// Handles API communication with the sentiment analysis backend

// Configuration
const DEFAULT_API_URL = "http://localhost:5000"; // Updated to correct port 5000
let API_URL = DEFAULT_API_URL;
let preferredModel = 'ensemble'; // Default to ensemble which is our best model
const API_TIMEOUT_MS = 8000; // 8 second timeout for API calls
let availableModels = {ensemble: true, simple: true}; // Default to ensure at least ensemble is available
let apiAvailable = false; // Track API availability

// Function to check if API is currently available
function isApiAvailable() {
  return apiAvailable;
}

// Initialize extension on install
chrome.runtime.onInstalled.addListener(() => {
  console.log("MoodMap extension installed");
  
  // Initialize storage with default settings if not set
  chrome.storage.local.set({ 
    defaultModel: 'ensemble',
    apiUrl: DEFAULT_API_URL,
    apiStatus: 'unknown',
    availableModels: {ensemble: true, simple: true}
  }, () => {
    console.log("Settings initialized with:", { 
      defaultModel: 'ensemble', 
      apiUrl: DEFAULT_API_URL 
    });
  });
  
  // Create context menu item
  chrome.contextMenus.create({
    id: "analyzeWithMoodMap",
    title: "Analyze with Mood Map",
    contexts: ["selection"]
  });
});

// Load settings and check API on service worker startup
chrome.storage.local.get(['defaultModel', 'apiUrl', 'apiStatus'], (result) => {
  if (result.defaultModel) {
    preferredModel = result.defaultModel;
    console.log("Loaded preferred model:", preferredModel);
  }
  
  if (result.apiUrl) {
    API_URL = result.apiUrl;
    console.log("Loaded API URL:", API_URL);
  }
  
  if (result.apiStatus === 'online') {
    apiAvailable = true;
  }
  
  // Check API and available models on startup
  checkApiAndModels();
  
  // Log the configured values
  console.log("Mood Map configuration loaded:", { 
    apiUrl: API_URL, 
    defaultModel: preferredModel 
  });
});

// Function to check API and available models
function checkApiAndModels() {
  console.log("Checking API health at:", `${API_URL}/health`);
  
  fetchWithTimeout(`${API_URL}/health`, {
    method: 'GET'
  }, 5000)
  .then(response => {
    if (!response.ok) {
      throw new Error(`API returned ${response.status}`);
    }
    return response.json();
  })
  .then(data => {
    console.log("API health check successful:", data);
    apiAvailable = true;
    
    // Store available models from API
    if (data.models_status) {
      availableModels = data.models_status;
      // Always keep simple model available as it works offline
      availableModels.simple = true;
      
      // If preferred model is not available, switch to ensemble or simple
      if (!availableModels[preferredModel]) {
        preferredModel = availableModels.ensemble ? 'ensemble' : 'simple';
        chrome.storage.local.set({ defaultModel: preferredModel });
        console.log(`Preferred model switched to ${preferredModel} based on availability`);
      }
      
      // Store available models in storage for popup
      chrome.storage.local.set({ 
        availableModels: availableModels,
        apiStatus: 'online'
      });
    }
  })
  .catch(error => {
    console.error("API health check failed:", error.message);
    apiAvailable = false;
    
    chrome.storage.local.set({ 
      apiStatus: 'offline',
      availableModels: { simple: true } // Only simple model works offline
    });
    
    // Default to simple model when API is down
    preferredModel = 'simple';
    chrome.storage.local.set({ defaultModel: 'simple' });
  });
}

// Helper function for fetch with timeout
async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    if (error.name === 'AbortError') {
      throw new Error('Request timeout');
    }
    throw error;
  }
}

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "analyzeWithMoodMap" && info.selectionText) {
    console.log("Context menu clicked, sending text to content script");
    // Send message to content script with the selected text
    chrome.tabs.sendMessage(tab.id, {
      type: 'analyzeSelectedText',
      text: info.selectionText
    });
  }
});

// Message handling from content scripts and popup
// FIXED: Modified message handler to properly support asynchronous responses
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log("Background script received message:", request.type);
  
  if (request.type === 'analyzeSelectedText' || request.type === 'analyzeSentiment') {
    const text = request.text;
    const model = request.model || preferredModel; // Use specified model or default
    
    if (!text || text.trim() === '') {
      console.log("Empty text received for analysis");
      sendResponse({ error: "No text provided for analysis" });
      return true;
    }
    
    console.log(`Analyzing text (${text.length} chars) with model: ${model}`);
    
    // Choose between online API analysis or offline analysis
    if (model === 'simple' || !isApiAvailable()) {
      // Use offline analysis for simple model or if API is down
      const result = performOfflineSentimentAnalysis(text);
      console.log("Offline analysis result:", result);
      sendResponse(result);
    } else {
      // Use Promise to handle the async API call
      (async () => {
        try {
          const response = await fetchWithTimeout(`${API_URL}/analyze`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
              text: text,
              model_type: model // Use model_type parameter name
            })
          }, API_TIMEOUT_MS);
          
          if (!response.ok) {
            console.error(`API error status: ${response.status}`);
            throw new Error(`API error: ${response.status}`);
          }
          
          const data = await response.json();
          console.log("Sentiment analysis result:", data);
          // Convert API response to a consistent format if needed
          const result = formatApiResponse(data);
          sendResponse(result);
        } catch (error) {
          console.error("Error in sentiment analysis:", error.message);
          // Fall back to offline analysis if API fails
          const fallbackResult = performOfflineSentimentAnalysis(text);
          sendResponse(fallbackResult);
        }
      })();
      
      return true; // Keep the message channel open for async response
    }
    
    return true; // Keep the message channel open for async response
  }
  
  else if (request.type === 'summarizeText') {
    const text = request.text;
    const sentiment = request.sentiment_category;
    
    // Skip summarization if API is not available
    if (!isApiAvailable()) {
      sendResponse({ 
        error: "API not available",
        summary: "Summarization requires API connection."
      });
      return true;
    }
    
    // Use Promise for async handling
    (async () => {
      try {
        const response = await fetchWithTimeout(`${API_URL}/summarize`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            text: text,
            sentiment: sentiment
          })
        }, API_TIMEOUT_MS);
        
        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        console.log("Summarization result:", data);
        sendResponse(data);
      } catch (error) {
        console.error("Error in summarization:", error.message);
        sendResponse({ 
          error: error.message,
          summary: "Could not generate summary. Try again later."
        });
      }
    })();
    
    return true; // Keep the message channel open for async response
  }
  
  else if (request.type === 'updateApiUrl') {
    const newUrl = request.url;
    API_URL = newUrl;
    chrome.storage.local.set({ apiUrl: newUrl });
    
    // Use Promise for async handling
    (async () => {
      try {
        const response = await fetchWithTimeout(`${newUrl}/health`, {
          method: 'GET'
        }, API_TIMEOUT_MS);
        
        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        apiAvailable = true;
        chrome.storage.local.set({ apiStatus: 'online' });
        sendResponse({ success: true, message: "API URL updated and connection verified" });
      } catch (error) {
        console.error("Error connecting to API:", error.message);
        apiAvailable = false;
        chrome.storage.local.set({ apiStatus: 'offline' });
        sendResponse({ success: false, error: error.message });
      }
    })();
    
    return true; // Keep the message channel open for async response
  }
  
  else if (request.type === 'checkBackend') {
    checkApiAndModels();
    
    // Use Promise for async handling
    (async () => {
      try {
        const response = await fetchWithTimeout(`${API_URL}/health`, {
          method: 'GET'
        }, 5000);
        
        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        apiAvailable = true;
        chrome.storage.local.set({ apiStatus: 'online' });
        sendResponse({ isAvailable: true, models: data.models_status });
      } catch (error) {
        console.error("Error checking API:", error.message);
        apiAvailable = false;
        chrome.storage.local.set({ apiStatus: 'offline' });
        sendResponse({ isAvailable: false, error: error.message });
      }
    })();
    
    return true; // Keep the message channel open for async response
  }
  
  else if (request.type === 'updateDefaultModel') {
    // Update the preferred model only if it's available
    if (request.model === 'simple' || availableModels[request.model]) {
      preferredModel = request.model;
      chrome.storage.local.set({ defaultModel: request.model });
      console.log("Default model updated to:", preferredModel);
      sendResponse({ success: true });
    } else {
      console.error(`Requested model ${request.model} is not available`);
      sendResponse({ success: false, error: "Model not available" });
    }
    return true;
  }
  
  else if (request.type === 'getPreferredModel') {
    // Return the current preferred model and available models
    sendResponse({ 
      model: preferredModel, 
      availableModels: availableModels,
      apiStatus: apiAvailable ? 'online' : 'offline'
    });
    return true;
  }
});

// Function to keep content script ready 
function keepAlive() {
  setInterval(() => {
    console.log("Background service worker keeping alive");
    // Periodically check API availability
    if (Math.random() < 0.2) {  // 20% chance to avoid too many requests
      checkApiAndModels();
    }
  }, 30000);  // Every 30 seconds
}

keepAlive();

// Improved offline sentiment analysis
function performOfflineSentimentAnalysis(text) {
  console.log('Performing offline sentiment analysis');
  
  const positiveWords = ['good', 'great', 'excellent', 'happy', 'love', 'nice', 
                         'wonderful', 'awesome', 'fantastic', 'positive', 'best',
                         'amazing', 'brilliant', 'perfect', 'delighted', 'joy',
                         'beautiful', 'favorite', 'liked', 'win', 'winning'];
                         
  const negativeWords = ['bad', 'terrible', 'awful', 'sad', 'hate', 'poor', 
                         'negative', 'horrible', 'wrong', 'fail', 'worst',
                         'disappointed', 'frustrating', 'useless', 'annoying',
                         'dislike', 'problem', 'difficult', 'trouble', 'worry'];
  
  let positiveScore = 0;
  let negativeScore = 0;
  
  // Convert to lowercase for case-insensitive matching
  text = text.toLowerCase();
  
  // Count positive words
  positiveWords.forEach(word => {
    const regex = new RegExp('\\b' + word + '\\b', 'gi');
    const matches = text.match(regex);
    if (matches) positiveScore += matches.length;
  });
  
  // Count negative words
  negativeWords.forEach(word => {
    const regex = new RegExp('\\b' + word + '\\b', 'gi');
    const matches = text.match(regex);
    if (matches) negativeScore += matches.length;
  });
  
  // Calculate overall sentiment
  let sentimentScore = 0;
  if (positiveScore === 0 && negativeScore === 0) {
    sentimentScore = 50; // Neutral if no sentiment words found
  } else {
    const total = positiveScore + negativeScore;
    sentimentScore = Math.round((positiveScore / total) * 100);
  }
  
  // Map to 3-category system (0=negative, 1=neutral, 2=positive)
  let prediction = 1; // Default to neutral
  if (sentimentScore >= 60) prediction = 2;  // Positive
  else if (sentimentScore <= 40) prediction = 0;  // Negative
  
  // Map sentiment score to text label
  let sentimentLabel = 'Neutral';
  if (prediction === 2) sentimentLabel = 'Positive';
  else if (prediction === 0) sentimentLabel = 'Negative';
  
  return {
    sentiment: sentimentLabel,
    prediction: prediction,
    sentiment_percentage: sentimentScore,
    label: sentimentLabel,
    score: (sentimentScore - 50) / 50, // Convert 0-100 to -1 to 1
    offline: true
  };
}

// Format API response to ensure consistency
function formatApiResponse(data) {
  // If response is already in our expected format, return as is
  if (data.sentiment && data.prediction !== undefined) {
    return data;
  }
  
  // Create formatted response from various API response formats
  let result = {
    offline: false
  };
  
  // Extract sentiment label
  if (data.label) {
    result.sentiment = data.label;
    result.label = data.label;
  } else if (data.sentiment) {
    result.label = data.sentiment;
  } else {
    result.sentiment = 'Neutral';
    result.label = 'Neutral';
  }
  
  // Extract prediction value (0=negative, 1=neutral, 2=positive)
  if (data.prediction !== undefined) {
    result.prediction = data.prediction;
  } else if (data.category !== undefined) {
    result.prediction = data.category;
  } else if (data.score !== undefined) {
    // Convert -1 to 1 score to category
    const score = data.score;
    if (score < -0.3) result.prediction = 0;
    else if (score > 0.3) result.prediction = 2;
    else result.prediction = 1;
  } else {
    result.prediction = 1; // Default to neutral
  }
  
  // Extract or calculate score
  if (data.score !== undefined) {
    result.score = data.score;
  } else if (data.sentiment_percentage !== undefined) {
    result.score = (data.sentiment_percentage - 50) / 50; // Convert 0-100 to -1 to 1
  } else if (data.prediction !== undefined) {
    // Convert prediction to approximate score
    switch(data.prediction) {
      case 0: result.score = -0.7; break; // Negative
      case 2: result.score = 0.7; break;  // Positive
      default: result.score = 0; break;   // Neutral
    }
  }
  
  // Extract or calculate percentage
  if (data.sentiment_percentage !== undefined) {
    result.sentiment_percentage = data.sentiment_percentage;
  } else if (data.score !== undefined) {
    result.sentiment_percentage = Math.round((data.score + 1) / 2 * 100);
  } else if (data.prediction !== undefined) {
    // Convert prediction to approximate percentage
    switch(data.prediction) {
      case 0: result.sentiment_percentage = 25; break; // Negative
      case 2: result.sentiment_percentage = 75; break; // Positive
      default: result.sentiment_percentage = 50; break; // Neutral
    }
  }
  
  return result;
}

// Helper function to get API URL
function getApiUrl(endpoint) {
  // Ensure endpoint starts with a slash
  if (!endpoint.startsWith('/')) {
    endpoint = '/' + endpoint;
  }
  
  // Remove trailing slash from API_URL if present
  const baseUrl = API_URL.endsWith('/') ? API_URL.slice(0, -1) : API_URL;
  
  return `${baseUrl}${endpoint}`;
}