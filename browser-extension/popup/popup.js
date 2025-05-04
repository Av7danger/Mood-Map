// MoodMap extension popup.js - Completely fixed version

// Add extension context invalidation error handling
let extensionContextValid = true;

// Wrapper function to safely make chrome API calls
function safeChromeApiCall(apiCall, fallback = null) {
  return function(...args) {
    try {
      if (!extensionContextValid) {
        console.warn("Extension context already invalidated, aborting API call");
        return typeof fallback === 'function' ? fallback(...args) : fallback;
      }
      
      return apiCall(...args);
    } catch (error) {
      console.error("Chrome API call failed:", error);
      
      if (error.message.includes("Extension context invalidated")) {
        handleExtensionContextInvalidated();
      }
      
      return typeof fallback === 'function' ? fallback(...args) : fallback;
    }
  };
}

// Handle extension context invalidation 
function handleExtensionContextInvalidated() {
  if (extensionContextValid) {
    extensionContextValid = false;
    console.warn("Extension context has been invalidated");
    
    // Show user-friendly error message
    const content = document.getElementById('content');
    const loading = document.getElementById('loading');
    
    if (content) {
      content.innerHTML = `
      <div class="error-container">
        <h2>Extension Error</h2>
        <p>The extension context has been invalidated. This usually happens when the extension is updated, reloaded, or the browser is restarted during an operation.</p>
        <p>Please close this popup and try again. If the problem persists, try reloading the extension:</p>
        <ol>
          <li>Go to your browser's extension management page</li>
          <li>Find Mood Map and click "Reload" or toggle it off and on</li>
          <li>Try using the extension again</li>
        </ol>
        <button id="close-popup-btn" class="primary-button">Close Popup</button>
      </div>`;
      
      // Add event listener for the close button
      setTimeout(() => {
        const closeBtn = document.getElementById('close-popup-btn');
        if (closeBtn) {
          closeBtn.addEventListener('click', () => window.close());
        }
      }, 0);
    }
    
    if (loading) {
      loading.style.display = 'none';
    }
  }
}

// Safely send message to background script with retry and error handling
function safeSendMessage(message, callback) {
  if (!extensionContextValid) {
    console.warn("Extension context invalidated, not sending message:", message);
    if (typeof callback === 'function') {
      callback({ error: "Extension context invalidated" });
    }
    return;
  }
  
  try {
    chrome.runtime.sendMessage(message, (response) => {
      // Check for runtime errors
      const lastError = chrome.runtime.lastError;
      if (lastError) {
        console.error("Runtime error:", lastError.message);
        
        // Handle extension context invalidation
        if (lastError.message.includes("Extension context invalidated")) {
          handleExtensionContextInvalidated();
        }
        
        // Call callback with error
        if (typeof callback === 'function') {
          callback({ error: lastError.message });
        }
        return;
      }
      
      // Call callback with response
      if (typeof callback === 'function') {
        callback(response);
      }
    });
  } catch (error) {
    console.error("Error sending message:", error);
    
    // Handle extension context invalidation
    if (error.message.includes("Extension context invalidated")) {
      handleExtensionContextInvalidated();
    }
    
    // Call callback with error
    if (typeof callback === 'function') {
      callback({ error: error.message });
    }
  }
}

// Global error handler
window.addEventListener('error', function(event) {
  console.error("Global error:", event.error);
  
  // Check if this is an extension context invalidation error
  if (event.error && event.error.message && event.error.message.includes("Extension context invalidated")) {
    handleExtensionContextInvalidated();
  }
  
  // Don't prevent default error handling
  return false;
});

// Safe storage access functions
const safeStorageGet = (keys, callback) => {
  try {
    chrome.storage.local.get(keys, (result) => {
      if (chrome.runtime.lastError) {
        console.error("Storage get error:", chrome.runtime.lastError);
        // Check for context invalidation
        if (chrome.runtime.lastError.message.includes("Extension context invalidated")) {
          handleExtensionContextInvalidated();
        }
        callback({});
      } else {
        callback(result);
      }
    });
  } catch (error) {
    console.error("Error in storage get:", error);
    if (error.message.includes("Extension context invalidated")) {
      handleExtensionContextInvalidated();
    }
    callback({});
  }
};

const safeStorageSet = (items, callback) => {
  try {
    chrome.storage.local.set(items, () => {
      if (chrome.runtime.lastError) {
        console.error("Storage set error:", chrome.runtime.lastError);
        // Check for context invalidation
        if (chrome.runtime.lastError.message.includes("Extension context invalidated")) {
          handleExtensionContextInvalidated();
        }
      }
      if (callback) callback();
    });
  } catch (error) {
    console.error("Error in storage set:", error);
    if (error.message.includes("Extension context invalidated")) {
      handleExtensionContextInvalidated();
    }
    if (callback) callback();
  }
};

// MoodMap extension popup.js - With extension context invalidation handling
document.addEventListener('DOMContentLoaded', function() {
  console.log('==== MOOD MAP DEBUG LOG ====');
  console.log('DOM content loaded, initializing Mood Map extension...');
  
  // Try to ping the background script to verify extension context is valid
  safeSendMessage({ type: 'ping' }, (response) => {
    if (response && response.error) {
      console.error("Background script communication error:", response.error);
      if (response.error.includes("Extension context invalidated")) {
        handleExtensionContextInvalidated();
        return;
      }
    } else if (response && response.type === 'pong') {
      console.log("Background script communication verified:", response);
      // Continue with extension initialization
      initializeExtension();
    } else {
      console.warn("Unexpected ping response:", response);
      // Try to continue anyway
      initializeExtension();
    }
  });
  
  // Main extension initialization function
  function initializeExtension() {
    // Force content to display and hide loading screen immediately
    const content = document.getElementById('content');
    const loading = document.getElementById('loading');
  
    if (content && loading) {
      console.log('Showing content and hiding loading screen');
      content.style.display = 'block';
      loading.style.display = 'none';
    }
  
    // Add direct debug information to the UI
    const summaryText = document.getElementById('summary-text');
    if (summaryText) {
      summaryText.textContent = 'Extension loaded. Checking API connection...';
    }
  
    // Global variables
    const API_BASE_URL = 'http://localhost:5000';
    let currentTab = 'analyze';
  
    // Helper functions
    function getElement(id) {
      return document.getElementById(id);
    }
  
    function getApiUrl(endpoint) {
      // Get from storage or use default
      const apiUrlDisplay = getElement('api-url-display');
      const apiUrl = apiUrlDisplay ? apiUrlDisplay.textContent.trim() : API_BASE_URL;
    
      // Remove trailing slash from base URL if present
      const cleanBaseUrl = apiUrl.endsWith('/') ? apiUrl.slice(0, -1) : apiUrl;
      // Add leading slash to endpoint if not present
      const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    
      return `${cleanBaseUrl}${cleanEndpoint}`;
    }
  
    function logToDebug(message) {
      console.log(message);
      const debugOutput = getElement('debug-output');
      if (debugOutput) {
        const timestamp = new Date().toLocaleTimeString();
        debugOutput.innerHTML += `<div class="debug-entry"><span class="timestamp">[${timestamp}]</span> ${message}</div>`;
        debugOutput.scrollTop = debugOutput.scrollHeight;
      }
    }
    
    // Initialize UI tabs
    function initializeTabs() {
      const tabButtons = document.querySelectorAll('.tab-btn');
      const tabContents = document.querySelectorAll('.tab-content');
      
      tabButtons.forEach(button => {
        button.addEventListener('click', () => {
          const tabId = button.getAttribute('data-tab');
          
          // Hide all tabs
          tabContents.forEach(tab => {
            tab.classList.remove('active');
          });
          
          // Deactivate all buttons
          tabButtons.forEach(btn => {
            btn.classList.remove('active');
          });
          
          // Activate the selected tab and button
          const selectedTab = document.getElementById(`${tabId}-tab`);
          if (selectedTab) {
            selectedTab.classList.add('active');
          }
          
          button.classList.add('active');
          
          // Update current tab
          currentTab = tabId;
          console.log(`Switched to tab: ${tabId}`);
          
          // Initialize the appropriate tab
          if (tabId === 'history') {
            initializeHistoryTab();
          } else if (tabId === 'visualize') {
            initializeVisualizationTab();
          }
        });
      });
      
      // Start with analyze tab active
      const analyzeTab = document.getElementById('analyze-tab');
      const analyzeBtn = document.querySelector('.tab-btn[data-tab="analyze"]');
      
      if (analyzeTab && analyzeBtn) {
        analyzeTab.classList.add('active');
        analyzeBtn.classList.add('active');
      }
    }
    
    // Test API connection
    function testApiConnection() {
      const apiStatus = getElement('api-status');
      const apiMessage = getElement('api-message');
      
      if (apiStatus) {
        apiStatus.className = 'api-status unknown';
        apiStatus.textContent = 'Testing...';
      }
      
      if (apiMessage) {
        apiMessage.className = 'api-message';
        apiMessage.textContent = '';
      }
      
      console.log('Testing API connection...');
      
      // Make a request to the health endpoint
      fetch(getApiUrl('health'), {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      })
      .then(response => {
        console.log(`API responded with status: ${response.status}`);
        
        if (!response.ok) {
          throw new Error(`API returned status ${response.status}`);
        }
        
        return response.json();
      })
      .then(data => {
        console.log('API health check successful:', data);
        
        if (apiStatus) {
          apiStatus.className = 'api-status online';
          apiStatus.textContent = 'Online';
        }
        
        if (apiMessage) {
          apiMessage.className = 'api-message success';
          apiMessage.textContent = 'Connection successful! API is responding correctly.';
        }
        
        if (summaryText) {
          summaryText.textContent = 'API connection successful. Ready to analyze text.';
        }
        
        // Save API status to storage
        safeStorageSet({ 
          apiStatus: 'online',
          lastHealthCheckTime: Date.now()
        });
        
        // Update model selector if we have model status information
        updateModelSelector(data.models_status || {simple: true, ensemble: true});
      })
      .catch(error => {
        console.error('API connection error:', error);
        
        if (apiStatus) {
          apiStatus.className = 'api-status offline';
          apiStatus.textContent = 'Offline';
        }
        
        if (apiMessage) {
          apiMessage.className = 'api-message error';
          apiMessage.textContent = `Connection failed: ${error.message}. Is the backend server running?`;
        }
        
        if (summaryText) {
          summaryText.textContent = 'No connection to API. Using simple offline mode.';
        }
        
        // Save API status to storage
        safeStorageSet({ apiStatus: 'offline' });
        
        // Default to simple model
        updateModelSelector({simple: true});
      });
    }
    
    // Update model selector based on available models
    function updateModelSelector(modelsStatus) {
      const modelSelector = getElement('model-selector');
      if (!modelSelector) return;
      
      // Save current selection
      const currentSelection = modelSelector.value;
      
      // Clear existing options
      modelSelector.innerHTML = '';
      
      // Add available models
      if (modelsStatus.ensemble) {
        const loadingState = modelsStatus.ensemble_loaded ? ' (loaded)' : ' (needs loading)';
        const loadingTime = modelsStatus.ensemble_loading_time ? ` (~${Math.round(modelsStatus.ensemble_loading_time)}s)` : '';
        addModelOption(modelSelector, 'ensemble', `Ensemble Model (Best Overall)${loadingState}${loadingTime}`);
      }
      
      if (modelsStatus.attention) {
        const loadingState = modelsStatus.attention_loaded ? ' (loaded)' : ' (needs loading)';
        const loadingTime = modelsStatus.attention_loading_time ? ` (~${Math.round(modelsStatus.attention_loading_time)}s)` : '';
        addModelOption(modelSelector, 'attention', `Attention Model (Detail Focused)${loadingState}${loadingTime}`);
      }
      
      if (modelsStatus.neutral_finetuner) {
        const loadingState = modelsStatus.neutral_loaded ? ' (loaded)' : ' (needs loading)';
        const loadingTime = modelsStatus.neutral_loading_time ? ` (~${Math.round(modelsStatus.neutral_loading_time)}s)` : '';
        addModelOption(modelSelector, 'neutral', `Neutral Model (Balanced)${loadingState}${loadingTime}`);
      }
      
      if (modelsStatus.advanced) {
        const loadingState = modelsStatus.advanced_loaded ? ' (loaded)' : ' (needs loading)';
        const loadingTime = modelsStatus.advanced_loading_time ? ` (~${Math.round(modelsStatus.advanced_loading_time)}s)` : '';
        addModelOption(modelSelector, 'advanced', `Advanced Model (with RAG & BART)${loadingState}${loadingTime}`);
      }
      
      // Always add simple option for offline use
      addModelOption(modelSelector, 'simple', 'Simple Model (Offline Mode - Always Available)');
      
      // Try to restore previous selection
      let selected = false;
      for (let i = 0; i < modelSelector.options.length; i++) {
        if (modelSelector.options[i].value === currentSelection) {
          modelSelector.value = currentSelection;
          selected = true;
          break;
        }
      }
      
      // Default to best available model if previous selection not available
      if (!selected) {
        if (modelsStatus.advanced && modelsStatus.advanced_loaded) {
          modelSelector.value = 'advanced';
        } else if (modelsStatus.ensemble && modelsStatus.ensemble_loaded) {
          modelSelector.value = 'ensemble';
        } else {
          modelSelector.value = 'simple';
        }
      }
      
      // Save the selected model
      safeStorageSet({ selectedModel: modelSelector.value });
      console.log(`Model set to: ${modelSelector.value}`);
    }
    
    // Helper to add option to model selector
    function addModelOption(selector, value, text) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = text;
      selector.appendChild(option);
    }
    
    // Analyze text sentiment
    function analyzeSentiment() {
      const textInput = getElement('text-input');
      if (!textInput || !textInput.value.trim()) {
        alert('Please enter some text to analyze.');
        return;
      }
      
      const text = textInput.value.trim();
      const modelSelector = getElement('model-selector');
      const modelType = modelSelector ? modelSelector.value : 'simple';
      
      console.log(`Analyzing text with model: ${modelType}`);
      
      // Clear previous results
      resetResultDisplay();
      
      // Show loading state
      const sentimentLabel = getElement('sentiment-label');
      if (sentimentLabel) {
        sentimentLabel.textContent = 'Analyzing...';
      }
      
      // Check if we should use API or local processing
      if (modelType === 'simple') {
        // Use simple offline processing
        const result = processLocalSentiment(text);
        updateResultDisplay(result);
        saveAnalysisToHistory(text, result);
      } else {
        // Use API for processing
        fetch(getApiUrl('analyze'), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify({
            text: text,
            model_type: modelType
          })
        })
        .then(response => {
          console.log(`API responded with status: ${response.status}`);
          
          if (!response.ok) {
            throw new Error(`API returned status ${response.status}`);
          }
          
          return response.json();
        })
        .then(result => {
          console.log('Analysis result:', result);
          updateResultDisplay(result);
          saveAnalysisToHistory(text, result);
        })
        .catch(error => {
          console.error('API analysis error:', error);
          
          // Show error
          if (sentimentLabel) {
            sentimentLabel.textContent = 'Analysis failed. Falling back to offline mode.';
          }
          
          // Fall back to local processing
          const result = processLocalSentiment(text);
          updateResultDisplay(result);
          saveAnalysisToHistory(text, result);
        });
      }
    }
    
    // Simple offline sentiment analysis
    function processLocalSentiment(text) {
      text = text.toLowerCase();
      
      // Expanded lists of sentiment words
      const positiveWords = [
        "love", "loving", "loved", "likes", "like", "liked", 
        "amazing", "excellent", "exceptional", "outstanding",
        "fantastic", "great", "good", "wonderful", "terrific", 
        "awesome", "happy", "happiness", "joy", "joyful", 
        "beautiful", "delightful", "pleasant", "glad", 
        "pleased", "satisfying", "satisfied", "impressive",
        "perfect", "brilliant", "superb", "splendid", "enjoy"
      ];
      
      const negativeWords = [
        "hate", "hates", "hated", "hating",
        "terrible", "awful", "horrible", "worst", 
        "bad", "disappointing", "disappointed", 
        "sad", "unhappy", "angry", "furious", "upset", 
        "annoying", "annoyed", "frustrating", "frustrated",
        "poor", "mediocre", "inadequate", "inferior",
        "boring", "worthless", "waste", "disgusting",
        "dislike", "dislikes", "disliked", "sucks"
      ];
      
      let positiveCount = 0;
      let negativeCount = 0;
      
      // Improved word matching using word boundaries
      const words = text.split(/\b/);
      
      // Count positive and negative words
      words.forEach(word => {
        word = word.trim();
        if (word.length < 2) return; // Skip single characters and empty strings
        
        if (positiveWords.includes(word)) {
          positiveCount++;
          console.log(`Found positive word: ${word}`);
        }
        
        if (negativeWords.includes(word)) {
          negativeCount++;
          console.log(`Found negative word: ${word}`);
        }
      });
      
      console.log(`Word counts - Positive: ${positiveCount}, Negative: ${negativeCount}`);
      
      let score = 0;
      let category = 1;
      let label = 'neutral';
      
      // Improved thresholds for sentiment classification
      if (positiveCount > 0 && positiveCount > negativeCount) {
        // Calculate positive score based on ratio and strength
        score = 0.3 + (0.7 * (positiveCount / (positiveCount + negativeCount || 1)));
        category = 2;
        label = 'positive';
      } else if (negativeCount > 0 && negativeCount >= positiveCount) {
        // Calculate negative score based on ratio and strength
        score = -0.3 - (0.7 * (negativeCount / (positiveCount + negativeCount || 1)));
        category = 0;
        label = 'negative';
      }
      
      // If no sentiment words found, keep as neutral
      if (positiveCount === 0 && negativeCount === 0) {
        score = 0;
        category = 1;
        label = 'neutral';
      }
      
      return {
        score: score,
        category: category,
        label: label,
        confidence: positiveCount + negativeCount > 0 ? 0.5 + (0.1 * (positiveCount + negativeCount)) : 0.5,
        model_used: 'simple_offline'
      };
    }
    
    // Reset result display
    function resetResultDisplay() {
      const sentimentLabel = getElement('sentiment-label');
      const summaryText = getElement('summary-text');
      
      if (sentimentLabel) {
        sentimentLabel.textContent = 'Waiting for analysis...';
        sentimentLabel.className = 'sentiment-label';
      }
      
      if (summaryText) {
        summaryText.textContent = '';
      }
      
      // Reset sentiment bars
      const negativeBar = getElement('negative-bar');
      const neutralBar = getElement('neutral-bar');
      const positiveBar = getElement('positive-bar');
      
      if (negativeBar) negativeBar.style.width = '0%';
      if (neutralBar) neutralBar.style.width = '0%';
      if (positiveBar) positiveBar.style.width = '0%';
      
      const negativeValue = getElement('negative-value');
      const neutralValue = getElement('neutral-value');
      const positiveValue = getElement('positive-value');
      
      if (negativeValue) negativeValue.textContent = '0%';
      if (neutralValue) neutralValue.textContent = '0%';
      if (positiveValue) positiveValue.textContent = '0%';
    }
    
    // Update result display with sentiment analysis
    function updateResultDisplay(result) {
      console.log('Updating display with result:', result);
      
      const sentimentLabel = getElement('sentiment-label');
      const summaryText = getElement('summary-text');
      const sentimentScore = getElement('sentiment-score');
      const sentimentCircle = getElement('sentiment-circle');
      
      // Normalize result format
      const normalizedResult = {
        score: result.score !== undefined ? result.score : 0,
        category: result.category !== undefined ? result.category : 1,
        label: result.label || 'neutral',
        confidence: result.confidence || 0.5,
        summary: result.summary || null,
        model_used: result.model_used || 'unknown'
      };
      
      // Update sentiment circle
      if (sentimentCircle) {
        // Determine emoji based on category
        let emoji = '😐'; // Neutral
        if (normalizedResult.category === 0) {
          emoji = '😞'; // Negative
        } else if (normalizedResult.category === 2) {
          emoji = '😊'; // Positive
        }
        
        if (sentimentScore) {
          sentimentScore.textContent = emoji;
        }
      }
      
      // Update sentiment label
      if (sentimentLabel) {
        sentimentLabel.textContent = normalizedResult.label.charAt(0).toUpperCase() + normalizedResult.label.slice(1);
        
        // Add appropriate class
        sentimentLabel.className = 'sentiment-label';
        if (normalizedResult.category === 0) {
          sentimentLabel.classList.add('negative');
        } else if (normalizedResult.category === 1) {
          sentimentLabel.classList.add('neutral');
        } else if (normalizedResult.category === 2) {
          sentimentLabel.classList.add('positive');
        }
      }
      
      // Update sentiment bars
      updateSentimentBars(normalizedResult);
      
      // Update summary text
      if (summaryText) {
        let summary = `This text has a ${normalizedResult.label} sentiment`;
        
        if (normalizedResult.confidence) {
          summary += ` with ${Math.round(normalizedResult.confidence * 100)}% confidence`;
        }
        
        summary += `.`;
        
        if (normalizedResult.model_used) {
          summary += ` Analyzed using ${normalizedResult.model_used} model.`;
        }
        
        // Add BART summary if available
        if (normalizedResult.summary) {
          summary += `\n\nText summary: ${normalizedResult.summary}`;
        }
        
        summaryText.textContent = summary;
      }
    }
    
    // Update sentiment bars
    function updateSentimentBars(result) {
      // Convert to bar percentages
      let negativeValue = 0;
      let neutralValue = 0;
      let positiveValue = 0;
      
      // Map category to bar values
      if (result.category === 0) {
        // Negative
        negativeValue = 80;
        neutralValue = 20;
        positiveValue = 10;
      } else if (result.category === 1) {
        // Neutral
        negativeValue = 20;
        neutralValue = 70;
        positiveValue = 20;
      } else if (result.category === 2) {
        // Positive
        negativeValue = 10;
        neutralValue = 20;
        positiveValue = 80;
      }
      
      // Fine-tune with confidence if available
      if (result.confidence) {
        const confidence = result.confidence;
        
        if (result.category === 0) {
          negativeValue = Math.min(100, negativeValue * confidence * 1.25);
        } else if (result.category === 1) {
          neutralValue = Math.min(100, neutralValue * confidence * 1.25);
        } else if (result.category === 2) {
          positiveValue = Math.min(100, positiveValue * confidence * 1.25);
        }
      }
      
      // Update the bars
      const negativeBar = getElement('negative-bar');
      const neutralBar = getElement('neutral-bar');
      const positiveBar = getElement('positive-bar');
      
      if (negativeBar) negativeBar.style.width = `${negativeValue}%`;
      if (neutralBar) neutralBar.style.width = `${neutralValue}%`;
      if (positiveBar) positiveBar.style.width = `${positiveValue}%`;
      
      // Update the values
      const negativeValue_el = getElement('negative-value');
      const neutralValue_el = getElement('neutral-value');
      const positiveValue_el = getElement('positive-value');
      
      if (negativeValue_el) negativeValue_el.textContent = `${Math.round(negativeValue)}%`;
      if (neutralValue_el) neutralValue_el.textContent = `${Math.round(neutralValue)}%`;
      if (positiveValue_el) positiveValue_el.textContent = `${Math.round(positiveValue)}%`;
    }
    
    // Save analysis to history
    function saveAnalysisToHistory(text, result) {
      safeStorageGet(['analysisHistory'], function(data) {
        const history = data.analysisHistory || [];
        
        // Create new entry
        const newEntry = {
          id: Date.now(),
          timestamp: new Date().toISOString(),
          text: text.length > 150 ? text.substring(0, 150) + '...' : text,
          result: result
        };
        
        // Add to beginning of history
        history.unshift(newEntry);
        
        // Keep only recent 50 entries
        const trimmedHistory = history.slice(0, 50);
        
        // Save back to storage
        safeStorageSet({ analysisHistory: trimmedHistory });
        console.log('Saved analysis to history');
      });
    }
    
    // Initialize history tab
    function initializeHistoryTab() {
      const historyList = getElement('history-list');
      if (!historyList) return;
      
      // Load history from storage
      safeStorageGet(['analysisHistory'], function(data) {
        const history = data.analysisHistory || [];
        
        if (history.length === 0) {
          historyList.innerHTML = '<div class="empty-history-message">No history items yet. Analyze some text to see it here.</div>';
          return;
        }
        
        // Clear list
        historyList.innerHTML = '';
        
        // Add each history item
        history.forEach(item => {
          const historyItem = document.createElement('div');
          historyItem.className = 'history-item';
          
          // Format date
          const date = new Date(item.timestamp);
          const formattedDate = `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
          
          // Get sentiment info
          const sentiment = item.result.label || 'neutral';
          const sentimentClass = sentiment.toLowerCase();
          
          // Create HTML for item
          historyItem.innerHTML = `
            <div class="history-item-header">
              <span class="history-item-date">${formattedDate}</span>
              <span class="history-item-sentiment ${sentimentClass}">${sentiment}</span>
            </div>
            <div class="history-item-text">${item.text}</div>
          `;
          
          // Add to list
          historyList.appendChild(historyItem);
          
          // Add click handler to reanalyze
          historyItem.addEventListener('click', function() {
            const textInput = getElement('text-input');
            if (textInput) {
              textInput.value = item.text;
              
              // Switch to analyze tab
              const analyzeBtn = document.querySelector('.tab-btn[data-tab="analyze"]');
              if (analyzeBtn) {
                analyzeBtn.click();
              }
              
              // Analyze the text
              analyzeSentiment();
            }
          });
        });
      });
    }
    
    // Initialize settings tab
    function initializeSettings() {
      const themeSelector = getElement('theme-selector');
      const autoAnalyzeToggle = getElement('auto-analyze-toggle');
      const updateApiUrlBtn = getElement('update-api-url-btn');
      const testApiBtn = getElement('test-api-btn');
      const apiUrlInput = getElement('api-url-input');
      const apiUrlDisplay = getElement('api-url-display');
      
      // Load settings from storage
      safeStorageGet(['theme', 'autoAnalyze', 'apiUrl'], function(data) {
        // Set theme
        if (themeSelector && data.theme) {
          themeSelector.value = data.theme;
          applyTheme(data.theme);
        }
        
        // Set auto-analyze
        if (autoAnalyzeToggle && data.autoAnalyze !== undefined) {
          autoAnalyzeToggle.checked = data.autoAnalyze;
        }
        
        // Set API URL
        if (apiUrlInput && data.apiUrl) {
          // Extract host part for input
          const url = new URL(data.apiUrl);
          apiUrlInput.value = `${url.hostname}${url.port ? ':' + url.port : ''}`;
        } else if (apiUrlInput) {
          apiUrlInput.value = 'localhost:5000';
        }
        
        // Update display
        if (apiUrlDisplay) {
          apiUrlDisplay.textContent = data.apiUrl || 'http://localhost:5000';
        }
      });
      
      // Theme selector change handler
      if (themeSelector) {
        themeSelector.addEventListener('change', function() {
          const theme = themeSelector.value;
          applyTheme(theme);
          safeStorageSet({ theme: theme });
        });
      }
      
      // Auto-analyze toggle handler
      if (autoAnalyzeToggle) {
        autoAnalyzeToggle.addEventListener('change', function() {
          safeStorageSet({ autoAnalyze: autoAnalyzeToggle.checked });
        });
      }
      
      // Update API URL button handler
      if (updateApiUrlBtn && apiUrlInput) {
        updateApiUrlBtn.addEventListener('click', function() {
          const protocolSelect = getElement('api-protocol-select');
          const protocol = protocolSelect ? protocolSelect.value : 'http';
          const host = apiUrlInput.value.trim().replace(/^https?:\/\//, '');
          
          const newApiUrl = `${protocol}://${host}`;
          
          // Update display
          if (apiUrlDisplay) {
            apiUrlDisplay.textContent = newApiUrl;
          }
          
          // Save to storage
          safeStorageSet({ apiUrl: newApiUrl }, function() {
            console.log(`API URL updated to: ${newApiUrl}`);
            // Test the connection
            testApiConnection();
          });
        });
      }
      
      // Test API button handler
      if (testApiBtn) {
        testApiBtn.addEventListener('click', function() {
          testApiConnection();
        });
      }
    }
    
    // Apply theme
    function applyTheme(theme) {
      const body = document.body;
      
      // Remove existing theme classes
      body.classList.remove('light-theme', 'dark-theme');
      
      // Add appropriate class
      if (theme === 'dark') {
        body.classList.add('dark-theme');
      } else if (theme === 'light') {
        body.classList.add('light-theme');
      } else if (theme === 'system') {
        // Check system preference
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
          body.classList.add('dark-theme');
        } else {
          body.classList.add('light-theme');
        }
      }
    }
    
    // Initialize debug panel
    function initializeDebugPanel() {
      const toggleDebugBtn = getElement('toggle-debug');
      const debugContainer = getElement('debug-container');
      const clearDebugBtn = getElement('clear-debug-btn');
      
      if (toggleDebugBtn && debugContainer) {
        toggleDebugBtn.addEventListener('click', function() {
          debugContainer.style.display = debugContainer.style.display === 'none' ? 'block' : 'none';
        });
      }
      
      if (clearDebugBtn) {
        clearDebugBtn.addEventListener('click', function() {
          const debugOutput = getElement('debug-output');
          if (debugOutput) {
            debugOutput.innerHTML = '';
            logToDebug('Debug console cleared');
          }
        });
      }
    }
    
    // Initialize all UI components
    function initializeUI() {
      // Set up tabs
      initializeTabs();
      
      // Set up analyze button
      const analyzeButton = getElement('analyze-button');
      if (analyzeButton) {
        analyzeButton.addEventListener('click', analyzeSentiment);
      }
      
      // Set up model selector
      const modelSelector = getElement('model-selector');
      if (modelSelector) {
        modelSelector.addEventListener('change', function() {
          safeStorageSet({ selectedModel: modelSelector.value });
        });
      }
      
      // Set up history tab
      const historyTab = document.querySelector('.tab-btn[data-tab="history"]');
      if (historyTab) {
        historyTab.addEventListener('click', initializeHistoryTab);
      }
      
      // Initialize settings
      initializeSettings();
      
      // Initialize debug panel
      initializeDebugPanel();
      
      // Test API connection
      testApiConnection();
      
      console.log('UI initialization complete');
    }
    
    // Initialize visualization tab
    function initializeVisualizationTab() {
      console.log('Initializing visualization tab');
      
      // Get filter dropdown
      const visualizationFilter = getElement('visualization-filter');
      
      // Load history from storage and create visualizations
      safeStorageGet(['analysisHistory'], function(data) {
        const history = data.analysisHistory || [];
        
        if (history.length === 0) {
          // Show empty state for each visualization panel
          const emptyMessage = '<div class="empty-visualization-message">No data available. Analyze some text to see visualizations.</div>';
          document.querySelector('.timeline-container').innerHTML = emptyMessage;
          document.querySelector('.wordcloud-container').innerHTML = emptyMessage;
          document.querySelector('.interactive-graph-container').innerHTML = emptyMessage;
          return;
        }
        
        // Create the visualizations
        createSentimentTimeline(history);
        createWordCloud(history);
        createInteractiveSentimentGraph(history, 'pie'); // Default to pie chart
        
        // Set up filter change handler
        if (visualizationFilter) {
          visualizationFilter.addEventListener('change', function() {
            const filteredHistory = filterHistoryByTimeRange(history, visualizationFilter.value);
            createSentimentTimeline(filteredHistory);
            createWordCloud(filteredHistory);
            
            // Get current graph type
            const graphType = document.querySelector('input[name="graph-type"]:checked').value;
            createInteractiveSentimentGraph(filteredHistory, graphType);
          });
        }
        
        // Set up graph type radio buttons
        const graphTypeRadios = document.querySelectorAll('input[name="graph-type"]');
        graphTypeRadios.forEach(radio => {
          radio.addEventListener('change', function() {
            const filteredHistory = filterHistoryByTimeRange(history, visualizationFilter.value);
            createInteractiveSentimentGraph(filteredHistory, this.value);
          });
        });
      });
    }
    
    // Filter history by time range
    function filterHistoryByTimeRange(history, timeRange) {
      const now = new Date();
      
      switch(timeRange) {
        case 'today':
          // Filter for items from today
          return history.filter(item => {
            const itemDate = new Date(item.timestamp);
            return itemDate.getDate() === now.getDate() &&
                   itemDate.getMonth() === now.getMonth() &&
                   itemDate.getFullYear() === now.getFullYear();
          });
        case 'week':
          // Filter for items from this week (last 7 days)
          const weekAgo = new Date(now);
          weekAgo.setDate(now.getDate() - 7);
          return history.filter(item => new Date(item.timestamp) >= weekAgo);
        case 'month':
          // Filter for items from this month (last 30 days)
          const monthAgo = new Date(now);
          monthAgo.setDate(now.getDate() - 30);
          return history.filter(item => new Date(item.timestamp) >= monthAgo);
        default:
          // All history
          return history;
      }
    }
    
    // Create sentiment timeline chart
    function createSentimentTimeline(history) {
      const container = document.querySelector('.timeline-container');
      if (!container) return;
      
      // Clear previous chart
      container.innerHTML = '';
      
      if (history.length === 0) {
        container.innerHTML = '<div class="empty-visualization-message">No data available for the selected time range.</div>';
        return;
      }
      
      // Create canvas element
      const canvas = document.createElement('canvas');
      canvas.id = 'sentiment-timeline';
      container.appendChild(canvas);
      
      // Prepare data
      const timeData = [];
      const sentimentData = [];
      
      // Sort by timestamp
      history.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
      
      history.forEach(item => {
        // Convert timestamp to readable format
        const date = new Date(item.timestamp);
        timeData.push(date.toLocaleString());
        
        // Convert sentiment to numerical value (-1 to 1)
        let sentimentValue = 0;
        if (item.result && item.result.score !== undefined) {
          sentimentValue = item.result.score;
        } else if (item.result && item.result.category !== undefined) {
          // Map category to a score: negative=0, neutral=1, positive=2
          sentimentValue = item.result.category === 0 ? -0.7 : (item.result.category === 2 ? 0.7 : 0);
        }
        
        sentimentData.push(sentimentValue);
      });
      
      console.log('Timeline data prepared:', timeData.length, 'points');
      
      // Create the chart
      const ctx = canvas.getContext('2d');
      window.sentimentTimelineChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: timeData,
          datasets: [{
            label: 'Sentiment Score',
            data: sentimentData,
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 2,
            pointBackgroundColor: sentimentData.map(value => {
              // Color points based on sentiment: red for negative, gray for neutral, green for positive
              if (value < -0.3) return 'rgba(255, 99, 132, 1)';
              if (value > 0.3) return 'rgba(75, 192, 192, 1)';
              return 'rgba(153, 102, 255, 1)';
            }),
            pointBorderColor: '#fff',
            pointRadius: 5,
            tension: 0.1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              min: -1,
              max: 1,
              ticks: {
                callback: function(value) {
                  if (value === -1) return 'Very Negative';
                  if (value === 0) return 'Neutral';
                  if (value === 1) return 'Very Positive';
                  return '';
                }
              }
            },
            x: {
              ticks: {
                maxRotation: 45,
                minRotation: 45
              }
            }
          },
          plugins: {
            tooltip: {
              callbacks: {
                label: function(context) {
                  const value = context.raw;
                  let sentiment = 'Neutral';
                  if (value < -0.3) sentiment = 'Negative';
                  if (value > 0.3) sentiment = 'Positive';
                  return `Sentiment: ${sentiment} (${value.toFixed(2)})`;
                },
                afterLabel: function(context) {
                  const index = context.dataIndex;
                  return `Text: ${history[index].text.substring(0, 50)}...`;
                }
              }
            }
          }
        }
      });
    }
    
    // Create word cloud visualization
    function createWordCloud(history) {
      const container = document.querySelector('.wordcloud-container');
      if (!container) return;
      
      // Clear previous content
      container.innerHTML = '';
      
      if (history.length === 0) {
        container.innerHTML = '<div class="empty-visualization-message">No data available for the selected time range.</div>';
        return;
      }
      
      // Create placeholder message with info about the word cloud being disabled
      const placeholderDiv = document.createElement('div');
      placeholderDiv.className = 'word-cloud-placeholder';
      placeholderDiv.style.textAlign = 'center';
      placeholderDiv.style.padding = '40px 20px';
      placeholderDiv.style.color = 'var(--text-secondary)';
      placeholderDiv.style.fontSize = '14px';
      placeholderDiv.innerHTML = 'Sentiment Word Cloud has been disabled in this version.';
      
      container.appendChild(placeholderDiv);
      
      console.log('Word cloud visualization has been disabled');
    }
    
    // Create interactive sentiment graph
    function createInteractiveSentimentGraph(history, chartType) {
      const container = document.querySelector('.interactive-graph-container');
      if (!container) return;
      
      // Clear previous chart
      container.innerHTML = '';
      
      if (history.length === 0) {
        container.innerHTML = '<div class="empty-visualization-message">No data available for the selected time range.</div>';
        return;
      }
      
      // Create canvas element
      const canvas = document.createElement('canvas');
      canvas.id = 'interactive-sentiment-graph';
      container.appendChild(canvas);
      
      console.log('Creating interactive graph with chart type:', chartType);
      
      // Count sentiment categories
      let negativeCount = 0;
      let neutralCount = 0;
      let positiveCount = 0;
      
      history.forEach(item => {
        if (item.result && (
            item.result.category === 0 || 
            (item.result.score !== undefined && item.result.score < -0.3)
          )) {
          negativeCount++;
        } else if (item.result && (
            item.result.category === 2 || 
            (item.result.score !== undefined && item.result.score > 0.3)
          )) {
          positiveCount++;
        } else {
          neutralCount++;
        }
      });
      
      console.log(`Sentiment distribution - Positive: ${positiveCount}, Neutral: ${neutralCount}, Negative: ${negativeCount}`);
      
      // Only proceed if we have data to show
      if (positiveCount === 0 && neutralCount === 0 && negativeCount === 0) {
        container.innerHTML = '<div class="empty-visualization-message">No sentiment data available.</div>';
        return;
      }
      
      // Fix chart type naming - "polar" should be "polarArea" in Chart.js
      if (chartType === 'polar') {
        chartType = 'polarArea';
      }
      
      console.log(`Using corrected chart type: ${chartType}`);
      
      // Set up chart config based on type
      const ctx = canvas.getContext('2d');
      
      // Chart configuration
      let chartConfig = {
        type: chartType,
        data: {
          labels: ['Positive', 'Neutral', 'Negative'],
          datasets: [{
            label: 'Sentiment Distribution',
            data: [positiveCount, neutralCount, negativeCount],
            backgroundColor: [
              'rgba(76, 175, 80, 0.7)', // Green for positive
              'rgba(158, 158, 158, 0.7)', // Gray for neutral
              'rgba(244, 67, 54, 0.7)' // Red for negative
            ],
            borderColor: [
              'rgba(76, 175, 80, 1)',
              'rgba(158, 158, 158, 1)',
              'rgba(244, 67, 54, 1)'
            ],
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'top',
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  const label = context.label || '';
                  const value = context.raw || 0;
                  const total = negativeCount + neutralCount + positiveCount;
                  const percentage = Math.round((value / total) * 100);
                  return `${label}: ${value} (${percentage}%)`;
                }
              }
            }
          }
        }
      };
      
      // Customize options based on chart type
      if (chartType === 'radar' || chartType === 'polarArea') {
        chartConfig.options.scales = {
          r: {
            min: 0,
            ticks: {
              display: false
            }
          }
        };
      }
      
      // Create the chart
      try {
        window.interactiveSentimentChart = new Chart(ctx, chartConfig);
        console.log('Interactive chart created successfully');
      } catch (error) {
        console.error('Error creating chart:', error);
        container.innerHTML = `<div class="empty-visualization-message">Error creating chart: ${error.message}</div>`;
      }
    }
    
    // Start initialization
    try {
      initializeUI();
      logToDebug('MoodMap Extension initialized successfully');
    } catch (error) {
      console.error('Error initializing MoodMap:', error);
      logToDebug(`Initialization error: ${error.message}`);
      
      // Check for extension context invalidation
      if (error.message.includes("Extension context invalidated")) {
        handleExtensionContextInvalidated();
        return;
      }
      
      // Show error in the UI
      if (summaryText) {
        summaryText.textContent = `Error initializing extension: ${error.message}`;
      }
    }
  }
});