// MoodMap Browser Extension Background Script
// Handles API communication with the sentiment analysis backend

// Configuration
const DEFAULT_API_URL = "https://localhost:5000";
let API_URL = DEFAULT_API_URL;
let preferredModel = 'roberta'; // Default value, will be updated from storage

// Initialize values from storage
chrome.storage.local.get(['defaultModel', 'apiUrl'], (result) => {
  if (result.defaultModel) {
    preferredModel = result.defaultModel;
  }
  
  if (result.apiUrl) {
    API_URL = result.apiUrl;
  }
});

// Create context menu for text selection
chrome.runtime.onInstalled.addListener(() => {
  console.log("MoodMap extension installed");
  
  // Initialize storage with default settings if not set
  chrome.storage.local.get(['defaultModel', 'apiUrl'], (result) => {
    if (!result.defaultModel) {
      chrome.storage.local.set({ defaultModel: 'roberta' });
    }
    
    if (!result.apiUrl) {
      chrome.storage.local.set({ apiUrl: DEFAULT_API_URL });
    }
  });
  
  // Create context menu item
  chrome.contextMenus.create({
    id: "analyzeWithMoodMap",
    title: "Analyze with Mood Map",
    contexts: ["selection"]
  });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "analyzeWithMoodMap" && info.selectionText) {
    // Send message to content script with the selected text
    chrome.tabs.sendMessage(tab.id, {
      type: 'analyzeSelectedText',
      text: info.selectionText
    });
  }
});

// Message handling from content scripts and popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log("Background script received message:", request.type);
  
  if (request.type === 'analyzeSelectedText' || request.type === 'analyzeSentiment') {
    const text = request.text;
    const model = request.model || preferredModel; // Use specified model or default
    
    if (!text || text.trim() === '') {
      sendResponse({ error: "No text provided for analysis" });
      return true;
    }
    
    // Make API request to sentiment analysis endpoint
    fetch(`${API_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        text: text,
        model: model // Pass the model to the API
      })
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log("Sentiment analysis result:", data);
      sendResponse(data);
    })
    .catch(error => {
      console.error("Error in sentiment analysis:", error);
      sendResponse({ error: error.message });
    });
    
    return true; // Keep the message channel open for async response
  }
  
  else if (request.type === 'summarizeText') {
    const text = request.text;
    const sentiment = request.sentiment_category;
    
    fetch(`${API_URL}/summarize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        text: text,
        sentiment: sentiment
      })
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log("Summarization result:", data);
      sendResponse(data);
    })
    .catch(error => {
      console.error("Error in summarization:", error);
      sendResponse({ error: error.message });
    });
    
    return true; // Keep the message channel open for async response
  }
  
  else if (request.type === 'updateApiUrl') {
    const newUrl = request.url;
    API_URL = newUrl;
    chrome.storage.local.set({ apiUrl: newUrl });
    
    // Test the new URL
    fetch(`${newUrl}/health`, {
      method: 'GET'
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      sendResponse({ success: true, message: "API URL updated and connection verified" });
    })
    .catch(error => {
      console.error("Error connecting to API:", error);
      sendResponse({ success: false, error: error.message });
    });
    
    return true; // Keep the message channel open for async response
  }
  
  else if (request.type === 'checkBackend') {
    fetch(`${API_URL}/health`, {
      method: 'GET'
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      sendResponse({ isAvailable: true });
    })
    .catch(error => {
      console.error("Error checking API:", error);
      sendResponse({ isAvailable: false, error: error.message });
    });
    
    return true; // Keep the message channel open for async response
  }
  
  else if (request.type === 'updateDefaultModel') {
    // Update the preferred model
    preferredModel = request.model;
    chrome.storage.local.set({ defaultModel: request.model });
    console.log("Default model updated to:", preferredModel);
    sendResponse({ success: true });
    return true;
  }
  
  else if (request.type === 'getPreferredModel') {
    // Return the current preferred model
    sendResponse({ model: preferredModel });
    return true;
  }
});

// Function to keep content script ready (in case we implement periodic sentiment checking)
function keepAlive() {
  setInterval(() => {
    console.log("Background service worker keeping alive");
  }, 20000);
}

keepAlive();