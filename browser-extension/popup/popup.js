// Set a maximum loading time to prevent the UI from being stuck
setTimeout(() => {
  const content = document.getElementById('content');
  const loading = document.getElementById('loading');
  if (content && loading && content.style.display === 'none') {
    console.log('Loading timeout reached - forcing UI to display');
    content.style.display = 'block';
    loading.style.display = 'none';
    
    // Add visible error message
    const analyzeContainer = document.querySelector('.analyze-container');
    if (analyzeContainer) {
      const errorMsg = document.createElement('div');
      errorMsg.className = 'api-message error';
      errorMsg.style.margin = '10px 0';
      errorMsg.textContent = 'Extension initialization timed out. Some features may not work correctly. Try clicking "Debug" for more information.';
      analyzeContainer.prepend(errorMsg);
    }
  }
}, 5000); // 5 second timeout

// Immediately wrap code in try/catch to catch initialization errors
try {
  // Function to map sentiment scores to categories
  function determineCategory(score) {
    // 3-category system:
    // negative: scores from -1 to -0.3
    // neutral: scores from -0.3 to 0.3
    // positive: scores from 0.3 to 1
    if (score < -0.3) {
      return {
        id: 'negative',
        label: 'Negative',
        emoji: '😞'
      };
    } else if (score < 0.3) {
      return {
        id: 'neutral',
        label: 'Neutral',
        emoji: '😐'
      };
    } else {
      return {
        id: 'positive',
        label: 'Positive',
        emoji: '😊'
      };
    }
  }

  // Helper function to safely get DOM elements
  function getElement(id) {
    return document.getElementById(id);
  }

  // Check if CORS headers have been correctly attached to the response
  function checkAndLogResponseHeaders(response) {
    console.log('Response headers received:');
    const headers = {};
    response.headers.forEach((value, name) => {
      console.log(`${name}: ${value}`);
      headers[name] = value;
    });
    
    // Check for common CORS-related headers
    if (!headers['access-control-allow-origin']) {
      console.warn('Warning: Response is missing Access-Control-Allow-Origin header');
    }
    
    return headers;
  }

  // Improved JSON parsing function with fallbacks
  function safeParseJson(data) {
    if (!data) return null;
    
    // If it's already an object, return it
    if (typeof data === 'object') return data;
    
    try {
      return JSON.parse(data);
    } catch (e) {
      console.error('JSON parsing error:', e);
      // Try to extract JSON from potential HTML or text response
      try {
        // Look for JSON pattern in string
        const jsonMatch = data.match(/(\{.*\})/s) || data.match(/(\[.*\])/s);
        if (jsonMatch && jsonMatch[1]) {
          return JSON.parse(jsonMatch[1]);
        }
      } catch (nestedError) {
        console.error('Nested JSON extraction failed:', nestedError);
      }
      return null;
    }
  }

  // Process API responses consistently
  async function processApiResponse(response, endpoint) {
    // Log headers for debugging
    const headers = checkAndLogResponseHeaders(response);
    
    // Check if response is OK
    if (!response.ok) {
      console.error(`API error: ${response.status} ${response.statusText}`);
      
      // Try to parse error message if available
      try {
        const errorData = await response.json();
        console.error('Error details:', errorData);
        throw new Error(errorData.detail || `API returned status ${response.status}`);
      } catch (parseError) {
        // If cannot parse as JSON, use text or status
        const errorText = await response.text().catch(() => 'Unknown error');
        throw new Error(errorText || `API returned status ${response.status}`);
      }
    }
    
    try {
      // Attempt to parse JSON response
      const data = await response.json();
      console.log(`Received ${endpoint} response:`, data);
      return data;
    } catch (parseError) {
      console.error(`Error parsing ${endpoint} response:`, parseError);
      const rawText = await response.text().catch(() => null);
      console.log('Raw response:', rawText);
      
      throw new Error(`Failed to parse API response as JSON: ${parseError.message}`);
    }
  }

  // Wait for DOM content to load before initializing popup
  document.addEventListener('DOMContentLoaded', function() {
    // Show initialization message directly (helpful for debugging)
    console.log('DOM content loaded, initializing Mood Map extension...');
    
    try {
      // Initialize theme first thing, so UI appears correct from the start
      initializeTheme();
      
      // Get UI elements - using the safer approach to avoid duplicate declarations
      const elements = {
        apiUrlInput: getElement('api-url-input'),
        testApiBtn: getElement('test-api-btn'),
        apiStatus: getElement('api-status'),
        apiMessage: getElement('api-message'),
        modelSelector: getElement('model-selector'),
        saveSettingsBtn: getElement('save-settings'),
        debugOutput: getElement('debug-output'),
        clearDebugBtn: getElement('clear-debug-btn'),
        content: getElement('content'),
        loading: getElement('loading')
      };
      
      // Helper function to log debug messages - both to console and debug area
      function logToDebug(message) {
        console.log(message); // Always log to console for debugging
        if (elements.debugOutput) {
          const timestamp = new Date().toLocaleTimeString();
          const logEntry = document.createElement('div');
          logEntry.className = 'debug-entry';
          logEntry.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;
          elements.debugOutput.appendChild(logEntry);
          elements.debugOutput.scrollTop = elements.debugOutput.scrollHeight;
        }
      }
      
      // Log initialization progress
      logToDebug('Mood Map extension initialization started');

      // Explicitly show content, hide loading right at the beginning
      if (elements.content) elements.content.style.display = 'block';
      if (elements.loading) elements.loading.style.display = 'none';
      logToDebug('Made content visible and removed loading screen');
      
      // Helper function to get full API URL - FIXED to handle trailing slashes properly
      function getApiUrl(endpoint) {
        const baseUrl = elements.apiUrlInput ? elements.apiUrlInput.value.trim() : 'http://localhost:5000';
        // Remove trailing slash from base URL if present
        const cleanBaseUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
        // Add leading slash to endpoint if not present
        const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
        
        const fullUrl = `${cleanBaseUrl}${cleanEndpoint}`;
        logToDebug(`Constructed API URL: ${fullUrl}`);
        return fullUrl;
      }
      
      // Load saved settings
      chrome.storage.local.get(['apiUrl', 'defaultModel', 'apiStatus', 'availableModels'], function(data) {
        // Set API URL input
        if (data.apiUrl && elements.apiUrlInput) {
          elements.apiUrlInput.value = data.apiUrl;
          logToDebug(`Loaded saved API URL: ${data.apiUrl}`);
        } else if (elements.apiUrlInput) {
          // Set default if not saved
          elements.apiUrlInput.value = 'http://localhost:5000';
          logToDebug('Using default API URL: http://localhost:5000');
        }
        
        // Update API status indicator if available
        if (data.apiStatus && elements.apiStatus) {
          elements.apiStatus.className = `api-status ${data.apiStatus}`;
          elements.apiStatus.textContent = data.apiStatus.charAt(0).toUpperCase() + data.apiStatus.slice(1);
          logToDebug(`API status: ${data.apiStatus}`);
        }
        
        // Update model selector with available models
        if (data.availableModels) {
          updateModelSelector(data.availableModels);
          logToDebug(`Available models: ${JSON.stringify(data.availableModels)}`);
        }
        
        // Set selected model
        if (data.defaultModel && elements.modelSelector) {
          // Check if this model option exists in the dropdown
          for (let i = 0; i < elements.modelSelector.options.length; i++) {
            if (elements.modelSelector.options[i].value === data.defaultModel) {
              elements.modelSelector.value = data.defaultModel;
              logToDebug(`Selected model: ${data.defaultModel}`);
              break;
            }
          }
        }
        
        // Test API connection on startup
        testApiConnection();
      });
      
      // Function to check API connection and available models
      function testApiConnection() {
        // Using model selector from elements object
        
        if (elements.apiStatus) {
          elements.apiStatus.className = 'api-status unknown';
          elements.apiStatus.textContent = 'Testing...';
        }
        
        if (elements.apiMessage) {
          elements.apiMessage.className = 'api-message';
          elements.apiMessage.textContent = '';
        }
        
        // Make sure we're using the correct API URL format
        const apiUrl = elements.apiUrlInput ? elements.apiUrlInput.value.trim() : 'http://localhost:5000';
        // Remove trailing slash from base URL if present
        const cleanBaseUrl = apiUrl.endsWith('/') ? apiUrl.slice(0, -1) : apiUrl;
        const healthUrl = `${cleanBaseUrl}/health`;
        
        logToDebug(`Testing API connection to ${healthUrl}...`);
        
        fetch(healthUrl, {
          method: 'GET',
          mode: 'cors',
          cache: 'no-cache',
          credentials: 'omit',
          headers: {
            'Accept': 'application/json',
          }
        })
          .then(response => {
            logToDebug(`API responded with status: ${response.status}`);
            if (!response.ok) {
              throw new Error(`API returned ${response.status}: ${response.statusText}`);
            }
            return response.json().catch(err => {
              logToDebug(`JSON parse error: ${err.message}`);
              throw new Error(`Failed to parse API response as JSON: ${err.message}`);
            });
          })
          .then(data => {
            logToDebug(`API health check response: ${JSON.stringify(data)}`);
            
            // Check if response has expected structure
            if (!data) {
              throw new Error('API response is empty');
            }
            
            if (elements.apiStatus) {
              elements.apiStatus.className = 'api-status online';
              elements.apiStatus.textContent = 'Online';
            }
            
            if (elements.apiMessage) {
              elements.apiMessage.className = 'api-message success';
              elements.apiMessage.textContent = 'Connection successful! API is responding correctly.';
            }
            
            // Handle different API response formats
            let modelsStatus = {};
            
            // More flexible response handling
            if (data.models_status && typeof data.models_status === 'object') {
              // Direct structure where models_status is a top-level property
              modelsStatus = data.models_status;
              logToDebug('Found models_status in response');
            } else if (data.data && data.data.models_status && typeof data.data.models_status === 'object') {
              // Nested structure where models_status is under data property
              modelsStatus = data.data.models_status;
              logToDebug('Found models_status nested under data property');
            } else if (data.status === 'online' && typeof data.models === 'object') {
              // Alternative structure where models is a top-level property
              modelsStatus = data.models;
              logToDebug('Found models object in response');
            } else if (data.status === 'online' || data.status === 'OK' || data.message === 'API is running') {
              // API is online but doesn't report model status - use defaults
              logToDebug('API is online but no model status reported, using defaults');
              modelsStatus = {
                ensemble: true,
                simple: true,
                attention: true
              };
            } else {
              // Fallback if structure is unknown
              logToDebug('Unknown API response structure, using default models');
              modelsStatus = {
                ensemble: true,
                simple: true
              };
            }
            
            // Update model selector based on available models
            if (elements.modelSelector) {
              updateModelSelector(modelsStatus);
            }
            
            // Save API status and URL
            chrome.storage.local.set({ 
              apiStatus: 'online', 
              availableModels: modelsStatus || {},
              apiUrl: cleanBaseUrl  // Save the cleaned URL
            });
          })
          .catch(error => {
            logToDebug(`API test failed: ${error.message}`);
            
            if (elements.apiStatus) {
              elements.apiStatus.className = 'api-status offline';
              elements.apiStatus.textContent = 'Offline';
            }
            
            if (elements.apiMessage) {
              elements.apiMessage.className = 'api-message error';
              elements.apiMessage.textContent = `Connection failed: ${error.message}. Check that the backend server is running.`;
            }
            
            // Save API status as offline
            chrome.storage.local.set({ 
              apiStatus: 'offline'
            });
            
            // Try alternative endpoint
            tryFallbackEndpoint();
          });
      }
      
      // Function to update model selector based on available models
      function updateModelSelector(modelsStatus) {
        if (!elements.modelSelector) {
          logToDebug('Model selector not found in DOM');
          return;
        }
        
        // Save current selection if possible
        const currentSelection = elements.modelSelector.value;
        
        // Clear existing options
        elements.modelSelector.innerHTML = '';
        
        // Add available model options
        if (modelsStatus.ensemble) {
          addModelOption(elements.modelSelector, 'ensemble', 'Ensemble Model (Best Overall)');
        }
        
        if (modelsStatus.attention) {
          addModelOption(elements.modelSelector, 'attention', 'Attention Model (Detail Focused)');
        }
        
        if (modelsStatus.neutral_finetuner) {
          addModelOption(elements.modelSelector, 'neutral', 'Neutral Model (Balanced)');
        }
        
        // Always add simple option for offline use
        addModelOption(elements.modelSelector, 'simple', 'Simple Model (Offline Mode)');
        
        // Try to restore previous selection if it's still available
        let modelFound = false;
        for (let i = 0; i < elements.modelSelector.options.length; i++) {
          if (elements.modelSelector.options[i].value === currentSelection) {
            elements.modelSelector.value = currentSelection;
            modelFound = true;
            break;
          }
        }
        
        // Default to ensemble if available or first option
        if (!modelFound) {
          if (modelsStatus.ensemble) {
            elements.modelSelector.value = 'ensemble';
          } else if (elements.modelSelector.options.length > 0) {
            elements.modelSelector.value = elements.modelSelector.options[0].value;
          }
        }
        
        // Save the selected model
        chrome.storage.local.set({ selectedModel: elements.modelSelector.value });
        logToDebug(`Model defaulted to: ${elements.modelSelector.value} based on available models`);
      }
      
      // Helper function to add option to selector
      function addModelOption(selector, value, text) {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = text;
        selector.appendChild(option);
      }

      // Try connecting to an alternative endpoint if main one fails
      function tryFallbackEndpoint() {
        // Try the root endpoint as fallback
        const apiUrl = elements.apiUrlInput ? elements.apiUrlInput.value.replace(/\/$/, '') : 'http://localhost:5000'; // Remove trailing slash if any
        
        logToDebug(`Trying fallback endpoint: ${apiUrl}/`);
        
        fetch(`${apiUrl}/`, {
          method: 'GET',
          mode: 'cors',
          cache: 'no-cache',
          credentials: 'omit',
          headers: {
            'Accept': 'text/html,application/json,*/*'
          }
        })
          .then(response => response.text())
          .then(text => {
            logToDebug(`Fallback endpoint response received: ${text.substring(0, 50)}...`);
            
            if (elements.apiMessage) {
              elements.apiMessage.className = 'api-message warning';
              elements.apiMessage.innerHTML = 'API server appears to be running but health endpoint is not responding.<br>Limited functionality may be available.';
            }
            
            // Set API status as partial
            if (elements.apiStatus) {
              elements.apiStatus.className = 'api-status partial';
              elements.apiStatus.textContent = 'Partial';
            }
            
            // Default to simple model when API is partially available
            const simpleOption = {simple: true};
            updateModelSelector(simpleOption);
          })
          .catch(error => {
            logToDebug(`Fallback endpoint failed: ${error.message}`);
            
            // Update UI to show API is completely offline
            if (elements.apiMessage) {
              elements.apiMessage.className = 'api-message error';
              elements.apiMessage.innerHTML = 'API server is not responding. Using offline mode only.<br>Start the backend server and retry.';
            }
            
            // Default to simple model when API is down
            const simpleOption = {simple: true};
            updateModelSelector(simpleOption);
          });
      }

      // Initialize all UI elements
      initializeUI();
      
      // Function to initialize all the UI elements and event listeners
      function initializeUI() {
        logToDebug('Initializing UI elements and event listeners');
        
        // Test API button click handler
        if (elements.testApiBtn) {
          elements.testApiBtn.addEventListener('click', function() {
            testApiConnection();
          });
        }

        // Update API URL button click handler - ADDED
        const updateApiUrlBtn = getElement('update-api-url-btn');
        if (updateApiUrlBtn) {
          updateApiUrlBtn.addEventListener('click', function() {
            if (elements.apiUrlInput) {
              const protocol = getElement('api-protocol-select').value;
              const host = elements.apiUrlInput.value.trim();
              // Construct full URL
              const fullUrl = `${protocol}://${host.replace(/^https?:\/\//, '')}`;
              // Save to storage
              chrome.storage.local.set({ apiUrl: fullUrl }, function() {
                logToDebug(`API URL updated to: ${fullUrl}`);
                // Update display
                const apiUrlDisplay = getElement('api-url-display');
                if (apiUrlDisplay) {
                  apiUrlDisplay.textContent = fullUrl;
                }
                // Test connection with new URL
                testApiConnection();
              });
            }
          });
        }
        
        // Analyze button click handler - ADDED
        const analyzeButton = getElement('analyze-button');
        if (analyzeButton) {
          analyzeButton.addEventListener('click', function() {
            const textInput = getElement('text-input');
            if (textInput) {
              const text = textInput.value.trim();
              if (text) {
                logToDebug(`Analyzing text: "${text.substring(0, 30)}..."`);
                
                // Call the analyze function
                analyzeEmotions(text)
                  .then(result => {
                    logToDebug(`Analysis complete: ${JSON.stringify(result)}`);
                    // Update UI with results
                    updateUIWithResults(result);
                  })
                  .catch(error => {
                    logToDebug(`Analysis failed: ${error.message}`);
                    // Show error in UI
                    const sentimentLabel = getElement('sentiment-label');
                    if (sentimentLabel) {
                      sentimentLabel.textContent = 'Analysis failed. Please try again.';
                      sentimentLabel.className = 'sentiment-label error';
                    }
                  });
              } else {
                logToDebug('No text to analyze');
                const sentimentLabel = getElement('sentiment-label');
                if (sentimentLabel) {
                  sentimentLabel.textContent = 'Please enter some text to analyze.';
                  sentimentLabel.className = 'sentiment-label warning';
                }
              }
            }
          });
        }

        // Initialize tabs
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabButtons.forEach(button => {
          button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            switchTab(tabId);
          });
        });
        
        // Initialize debug button if it exists
        const toggleDebugBtn = getElement('toggle-debug');
        if (toggleDebugBtn) {
          toggleDebugBtn.addEventListener('click', function() {
            const debugContainer = getElement('debug-container');
            if (debugContainer) {
              debugContainer.style.display = debugContainer.style.display === 'none' ? 'block' : 'none';
            }
          });
        }
        
        // Initialize clear debug button
        if (elements.clearDebugBtn) {
          elements.clearDebugBtn.addEventListener('click', function() {
            if (elements.debugOutput) {
              elements.debugOutput.innerHTML = '';
              logToDebug('Debug console cleared');
            }
          });
        }
        
        // Start by showing the analyze tab
        switchTab('analyze');
        logToDebug('UI initialization complete');
        
        // Auto test the analysis endpoint with a sample text - but only if not tested recently
        setTimeout(() => {
          logToDebug('Checking if analysis endpoint auto-test is needed...');
          
          // Get the last time we did an auto-test
          chrome.storage.local.get(['lastAnalysisTest', 'apiStatus'], function(data) {
            const now = Date.now();
            const lastTestTime = data.lastAnalysisTest || 0;
            const apiStatus = data.apiStatus || 'unknown';
            
            // Only auto-test if:
            // 1. We haven't tested in the last 10 minutes
            // 2. The API status is 'online' or 'unknown'
            if (now - lastTestTime > 600000 && (apiStatus === 'online' || apiStatus === 'unknown')) {
              logToDebug('Auto-testing analysis endpoint...');
              const testText = "This is a test message to verify the analyze endpoint is working correctly.";
              
              // Set the text in the input field
              const textInput = getElement('text-input');
              if (textInput) {
                textInput.value = testText;
                logToDebug('Set test text in input field');
              }
              
              // Directly call the analyze function to bypass UI interactions
              logToDebug('API is online, sending test analysis request...');
              analyzeEmotions(testText)
                .then(result => {
                  logToDebug(`TEST ANALYSIS COMPLETE: ${JSON.stringify(result)}`);
                  
                  // Save the time we did this test
                  chrome.storage.local.set({ lastAnalysisTest: now });
                  
                  // If successful, update API status to online and lastHealthCheckTime
                  if (result && !result.error) {
                    chrome.storage.local.set({ 
                      apiStatus: 'online',
                      lastHealthCheckTime: now
                    });
                    logToDebug('API status updated to online based on successful analysis test');
                  }
                })
                .catch(error => {
                  logToDebug(`TEST ANALYSIS FAILED: ${error.message}`);
                });
            } else {
              logToDebug(`Skipping auto-test: ${now - lastTestTime < 600000 ? 'tested recently' : 'API status is ' + apiStatus}`);
            }
          });
        }, 3000); // Wait 3 seconds after initialization before testing
      }
      
      // Switch tab function
      function switchTab(tabId) {
        logToDebug(`Switching to tab: ${tabId}`);
        
        // Hide all tabs
        const tabContents = document.querySelectorAll('.tab-content');
        tabContents.forEach(tab => {
          tab.classList.remove('active');
        });
        
        // Deactivate all buttons
        const tabButtons = document.querySelectorAll('.tab-btn');
        tabButtons.forEach(button => {
          button.classList.remove('active');
        });
        
        // Activate the selected tab
        const selectedTab = document.getElementById(`${tabId}-tab`);
        if (selectedTab) {
          selectedTab.classList.add('active');
        }
        
        // Activate the selected button
        const selectedButton = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
        if (selectedButton) {
          selectedButton.classList.add('active');
        }
      }
      
      // Helper function to update UI with analysis results
      function updateUIWithResults(result) {
        // Log the result for debugging
        console.log('Updating UI with result:', result);
        
        // Update sentiment score
        const sentimentScore = getElement('sentiment-score');
        if (sentimentScore) {
          // Normalize the sentiment score for display
          let displayScore;
          
          // Different backends use different scales, normalize to -1 to 1
          if (result.score !== undefined && (result.score < -1 || result.score > 1)) {
            // Map from 0-2 scale to -1 to 1 scale
            displayScore = (result.score - 1);
          } else {
            displayScore = result.sentiment;
          }
          
          // Display the sentiment score with 2 decimal places
          sentimentScore.textContent = displayScore.toFixed(2);
          
          // Add class based on sentiment
          sentimentScore.className = '';
          if (displayScore < -0.3) {
            sentimentScore.classList.add('negative');
          } else if (displayScore < 0.3) {
            sentimentScore.classList.add('neutral');
          } else {
            sentimentScore.classList.add('positive');
          }
        }
        
        // Update sentiment label and emoji
        const sentimentLabel = getElement('sentiment-label');
        if (sentimentLabel) {
          let category;
          if (result.score !== undefined && (result.score < -1 || result.score > 1)) {
            // Use category directly if available in the expected range
            if (result.category >= 0 && result.category <= 2) {
              const scoreMap = [-0.7, 0, 0.7]; // Map categories to scores
              category = determineCategory(scoreMap[result.category]);
            } else {
              // Map score to our format
              category = determineCategory((result.score - 1));
            }
          } else {
            // Use sentiment directly
            category = determineCategory(result.sentiment);
          }
          
          sentimentLabel.textContent = `${category.emoji} ${category.label}`;
          sentimentLabel.className = 'sentiment-label ' + category.id;
        }
        
        // Update the sentiment circle UI
        const sentimentCircle = getElement('sentiment-circle');
        if (sentimentCircle) {
          // Remove any existing sentiment classes
          sentimentCircle.classList.remove(
            'negative',
            'overwhelmingly-negative',
            'neutral',
            'positive',
            'overwhelmingly-positive'
          );
          
          // Add appropriate class based on sentiment
          if (result.sentiment < -0.7) {
            sentimentCircle.classList.add('overwhelmingly-negative');
          } else if (result.sentiment < -0.3) {
            sentimentCircle.classList.add('negative');
          } else if (result.sentiment < 0.3) {
            sentimentCircle.classList.add('neutral');
          } else if (result.sentiment < 0.7) {
            sentimentCircle.classList.add('positive');
          } else {
            sentimentCircle.classList.add('overwhelmingly-positive');
          }
        }
        
        // Update summary text
        const summaryText = getElement('summary-text');
        if (summaryText) {
          let summary = `This text has a ${result.label} sentiment with ${(result.confidence * 100).toFixed(0)}% confidence.`;
          
          // Add emotions if available
          if (result.emotions && Object.keys(result.emotions).length > 0) {
            const topEmotions = Object.entries(result.emotions)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 3);
              
            if (topEmotions.length > 0) {
              summary += ' Top emotions detected: ';
              summary += topEmotions
                .map(([emotion, score]) => `${emotion} (${(score * 100).toFixed(0)}%)`)
                .join(', ');
            }
          }
          
          summaryText.textContent = summary;
        }
        
        // Update sentiment bars
        updateSentimentBars(result);
      }
      
      // Helper function to update sentiment bars
      function updateSentimentBars(result) {
        // Convert sentiment score (-1 to 1) to percentages for each category
        let negativeValue = 0;
        let neutralValue = 0;
        let positiveValue = 0;
        
        // Simple mapping algorithm
        if (result.sentiment < -0.3) {
          // Negative sentiment
          negativeValue = Math.min(100, (Math.abs(result.sentiment) * 100));
          neutralValue = Math.max(0, (1 - Math.abs(result.sentiment)) * 50);
          positiveValue = Math.max(0, 10); // Always show a little positive for UX
        } else if (result.sentiment < 0.3) {
          // Neutral sentiment
          const neutralStrength = 1 - Math.abs(result.sentiment) * 3.33;
          neutralValue = Math.min(100, neutralStrength * 100);
          if (result.sentiment < 0) {
            negativeValue = Math.min(100, Math.abs(result.sentiment) * 100);
            positiveValue = Math.max(0, 10);
          } else {
            positiveValue = Math.min(100, result.sentiment * 100);
            negativeValue = Math.max(0, 10);
          }
        } else {
          // Positive sentiment
          positiveValue = Math.min(100, result.sentiment * 100);
          neutralValue = Math.max(0, (1 - result.sentiment) * 50);
          negativeValue = Math.max(0, 10); // Always show a little negative for UX
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
      
      // Theme handling function
      function initializeTheme() {
        // Get saved theme preference
        chrome.storage.local.get(['theme'], function(data) {
          const savedTheme = data.theme || 'light';
          const themeSelector = document.getElementById('theme-selector');
          
          // Update the theme selector if it exists
          if (themeSelector) {
            themeSelector.value = savedTheme;
          }
          
          // Apply the theme
          applyTheme(savedTheme);
          
          // Log the theme application
          console.log(`Applied saved theme: ${savedTheme}`);
        });
        
        // Add event listener to theme selector
        const themeSelector = document.getElementById('theme-selector');
        if (themeSelector) {
          themeSelector.addEventListener('change', function() {
            const selectedTheme = themeSelector.value;
            applyTheme(selectedTheme);
            
            // Save the theme preference
            chrome.storage.local.set({ theme: selectedTheme }, function() {
              console.log(`Theme preference saved: ${selectedTheme}`);
            });
          });
        }
      }
      
      // Function to apply theme
      function applyTheme(theme) {
        const body = document.body;
        
        // Remove any existing theme classes
        body.classList.remove('light-theme', 'dark-theme');
        
        // Add the appropriate theme class
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
          
          // Listen for system theme changes
          window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
            if (document.getElementById('theme-selector').value === 'system') {
              body.classList.remove('light-theme', 'dark-theme');
              body.classList.add(event.matches ? 'dark-theme' : 'light-theme');
            }
          });
        }
        
        console.log(`Theme applied: ${theme}`);
      }
      
      // Initialize theme
      initializeTheme();
      
    } catch (error) {
      console.error('Error in Mood Map initialization:', error);
      // Display error directly in popup for debugging
      document.body.innerHTML += `<div style="color:red; padding:20px; border:2px solid red;">
        <h3>Extension Error</h3>
        <p>${error.message}</p>
        <p>Stack trace: ${error.stack}</p>
      </div>`;
      
      // Try to show content anyway
      const content = document.getElementById('content');
      const loading = document.getElementById('loading');
      if (content) content.style.display = 'block';
      if (loading) loading.style.display = 'none';
    }
  });

  // Define global variables/API here
  const API_BASE_URL = 'http://localhost:5000';
  
  // Function to analyze text with emotions
  async function analyzeEmotions(text) {
    try {
      console.log('Analyzing text:', text);
      
      // Get the API URL and model selection from storage
      const [apiUrl, model, apiStatus] = await Promise.all([
        new Promise(resolve => {
          chrome.storage.local.get(['apiUrl'], function(data) {
            resolve(data.apiUrl || API_BASE_URL);
          });
        }),
        new Promise(resolve => {
          chrome.storage.local.get(['selectedModel'], function(data) {
            resolve(data.selectedModel || 'ensemble');
          });
        }),
        new Promise(resolve => {
          chrome.storage.local.get(['apiStatus'], function(data) {
            resolve(data.apiStatus || 'unknown');
          });
        })
      ]);
      
      // Check if we should use offline processing 
      if (model === 'simple') {
        console.log('Using local simple model as selected');
        return processSimpleAnalysis(text);
      }
      
      // Only check health if we don't already know the API is online
      if (apiStatus !== 'online') {
        console.log('API status unknown or offline, checking health before proceeding');
        // Use our cached health check to avoid redundant API calls
        const healthResult = await checkApiHealthWithCache();
        if (healthResult.status !== 'online') {
          console.log('API is not online, using simple model as fallback');
          return processSimpleAnalysis(text);
        }
      } else {
        console.log('Using cached API status: online');
      }
      
      // Construct URL properly
      let baseUrl = apiUrl.endsWith('/') ? apiUrl.slice(0, -1) : apiUrl;
      const fullUrl = `${baseUrl}/analyze`;
      
      console.log(`Sending analysis request to: ${fullUrl}`);
      console.log(`Using model: ${model}`);
      
      // Make the API request to analyze endpoint with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
      
      try {
        // Create request options
        const requestOptions = {
          method: 'POST',
          mode: 'cors',
          cache: 'no-cache',
          credentials: 'omit',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          body: JSON.stringify({
            text: text,
            model_type: model
          }),
          signal: controller.signal
        };
        
        console.log('Sending fetch request...');
        
        // Send the request
        const response = await fetch(fullUrl, requestOptions);
        clearTimeout(timeoutId);
        
        // Update the API status in storage based on this response
        if (response.ok && apiStatus !== 'online') {
          chrome.storage.local.set({ 
            apiStatus: 'online',
            lastHealthCheckTime: Date.now() // Update the timestamp since we know API is working
          });
          console.log('Updated API status to online based on successful analyze request');
        } else if (!response.ok && apiStatus === 'online') {
          chrome.storage.local.set({ apiStatus: 'error' });
          console.log('Updated API status to error based on failed analyze request');
        }
        
        // Use our helper function to process the response
        const result = await processApiResponse(response, 'analyze');
        
        // Save to history
        saveAnalysisToHistory(text, result);
        
        // Convert the API response to our internal format
        return {
          sentiment: mapScoreToRange(result.score, result.category),
          category: result.category !== undefined ? result.category : 1,
          label: result.label || 'neutral',
          confidence: result.confidence || 0.5,
          emotions: result.emotions || {}
        };
        
      } catch (fetchError) {
        clearTimeout(timeoutId);
        console.error('Fetch error:', fetchError);
        
        // Update API status to offline or error if we get a network error
        if (fetchError.name === 'TypeError' && apiStatus === 'online') {
          chrome.storage.local.set({ apiStatus: 'offline' });
          console.log('Updated API status to offline due to network error');
        }
        
        if (fetchError.name === 'AbortError') {
          throw new Error('API request timed out after 10 seconds');
        }
        throw fetchError;
      }
    } catch (error) {
      console.error('Error in analyzeEmotions:', error);
      
      // Fall back to simple analysis on error
      return processSimpleAnalysis(text);
    }
  }
  
  // Helper function to map score to our -1 to 1 range
  function mapScoreToRange(score, category) {
    // If score is already in our expected range (-1 to 1)
    if (score !== undefined && score >= -1 && score <= 1) {
      return score;
    }
    
    // If we have category (0, 1, 2) but no score or score outside range
    if (category !== undefined) {
      // Map categories to scores in our expected range
      if (category === 0) return -0.7;      // Negative
      else if (category === 1) return 0;    // Neutral
      else if (category === 2) return 0.7;  // Positive
    }
    
    // If score exists but is not in our range
    if (score !== undefined) {
      // If score is in 0-2 range (some models use this)
      if (score >= 0 && score <= 2) {
        return score - 1; // Convert 0-2 to -1 to 1
      }
    }
    
    // Default to neutral if we couldn't determine
    return 0;
  }
  
  // Function to save analysis to history
  function saveAnalysisToHistory(text, result) {
    try {
      // Get existing history or initialize empty array
      chrome.storage.local.get(['analysisHistory'], function(data) {
        const history = data.analysisHistory || [];
        
        // Create new history entry
        const newEntry = {
          id: Date.now(), // Use timestamp as unique ID
          timestamp: new Date().toISOString(),
          text: text.length > 150 ? text.substring(0, 150) + '...' : text,
          result: result,
        };
        
        // Add to beginning of history array (most recent first)
        history.unshift(newEntry);
        
        // Keep only the most recent 50 entries
        const trimmedHistory = history.slice(0, 50);
        
        // Save back to storage
        chrome.storage.local.set({ analysisHistory: trimmedHistory }, function() {
          console.log('Analysis saved to history');
        });
      });
    } catch (error) {
      console.error('Error saving to history:', error);
    }
  }
  
  // Helper function to check if API is online - FIXED to handle URL construction correctly
  async function isApiOnline() {
    try {
      const apiUrl = await new Promise(resolve => {
        chrome.storage.local.get(['apiUrl'], function(data) {
          resolve(data.apiUrl || API_BASE_URL);
        });
      });
      
      // Construct URL properly
      let baseUrl = apiUrl.endsWith('/') ? apiUrl.slice(0, -1) : apiUrl;
      const healthUrl = `${baseUrl}/health`;
      
      console.log(`Checking if API is online at: ${healthUrl}`);
      
      // Add timeout to avoid hanging
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
      
      try {
        const response = await fetch(healthUrl, { 
          signal: controller.signal,
          method: 'GET',
          mode: 'cors',
          cache: 'no-cache',
          credentials: 'omit',
          headers: {
            'Accept': 'application/json',
          }
        });
        clearTimeout(timeoutId);
        console.log(`API health check response status: ${response.status}`);
        return response.ok;
      } catch (fetchError) {
        clearTimeout(timeoutId);
        console.error('Error in health check:', fetchError);
        return false;
      }
    } catch (error) {
      console.error('Error checking API status:', error);
      return false;
    }
  }
  
  // Simple offline text analysis
  function processSimpleAnalysis(text) {
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
  
  // Function to check backend health with improved error handling
  function checkBackendHealth() {
    return new Promise((resolve, reject) => {
      // Get stored API URL or use default
      chrome.storage.local.get(['apiUrl'], function(data) {
        const apiUrl = data.apiUrl || API_BASE_URL;
        
        // Ensure proper URL formatting
        const baseUrl = apiUrl.endsWith('/') ? apiUrl.slice(0, -1) : apiUrl;
        const healthUrl = `${baseUrl}/health`;
        
        console.log(`Checking backend health at: ${healthUrl}`);
        
        // Add timeout to avoid hanging
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
        
        fetch(healthUrl, { signal: controller.signal })
          .then(response => {
            clearTimeout(timeoutId);
            if (!response.ok) {
              throw new Error(`Health check failed with status ${response.status}`);
            }
            return response.json();
          })
          .then(data => {
            console.log('Health check response:', data);
            
            // Check different health response formats
            let status = 'online';
            
            if (data.status === 'error') {
              status = 'error';
            }
            
            resolve({
              status: status,
              data: data
            });
          })
          .catch(error => {
            clearTimeout(timeoutId);
            console.error('Health check error:', error);
            
            // Try fallback to root endpoint
            console.log('Trying fallback root endpoint');
            
            fetch(`${baseUrl}/`, { 
              signal: controller.signal,
              method: 'GET'
            })
              .then(response => {
                clearTimeout(timeoutId);
                if (response.ok) {
                  resolve({
                    status: 'partial',
                    message: 'Root endpoint available but health endpoint not responding'
                  });
                } else {
                  resolve({
                    status: 'offline',
                    message: 'Backend completely unavailable'
                  });
                }
              })
              .catch(error => {
                clearTimeout(timeoutId);
                console.error('Fallback check error:', error);
                resolve({
                  status: 'offline',
                  message: 'Backend completely unavailable'
                });
              });
          });
      });
    });
  }
  
  // Add direct fetch testing function for debugging
  function testFetchToEndpoint(endpoint, model = 'simple') {
    const message = document.createElement('div');
    message.className = 'api-message';
    message.style.margin = '10px 0';
    message.innerHTML = 'Testing direct connection to API...';
    
    // Get reference to the debug output
    const debugOutput = document.getElementById('debug-output');
    if (debugOutput) {
      const timestamp = new Date().toLocaleTimeString();
      const logEntry = document.createElement('div');
      logEntry.className = 'debug-entry test-entry';
      logEntry.innerHTML = `<span class="timestamp">[${timestamp}]</span> Starting direct API test to ${endpoint}...`;
      debugOutput.appendChild(logEntry);
      debugOutput.scrollTop = debugOutput.scrollHeight;
    }
    
    // Get API URL from the input or use default
    const apiUrlInput = document.getElementById('api-url-input');
    const apiUrl = apiUrlInput ? apiUrlInput.value.trim() : 'http://localhost:5000';
    const baseUrl = apiUrl.endsWith('/') ? apiUrl.slice(0, -1) : apiUrl;
    const url = `${baseUrl}/${endpoint}`.replace(/\/\//g, '/');
    
    console.log(`Testing direct fetch to ${url}`);
    
    // Make the test request
    const testData = {
      text: "This is a direct test of the API endpoint.",
      model_type: model
    };
    
    // First test with simple fetch to see if we get any response
    fetch(url, {
      method: endpoint === 'health' ? 'GET' : 'POST',
      mode: 'cors',
      cache: 'no-cache',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: endpoint === 'health' ? undefined : JSON.stringify(testData)
    })
    .then(response => {
      console.log(`Direct API test response status: ${response.status}`);
      if (debugOutput) {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = 'debug-entry';
        logEntry.innerHTML = `<span class="timestamp">[${timestamp}]</span> Response received: ${response.status} ${response.statusText}`;
        debugOutput.appendChild(logEntry);
      }
      
      return response.text();
    })
    .then(text => {
      console.log(`Response body: ${text.substring(0, 200)}...`);
      
      // Try to parse as JSON
      try {
        const json = JSON.parse(text);
        console.log('Parsed JSON response:', json);
        
        if (debugOutput) {
          const timestamp = new Date().toLocaleTimeString();
          const logEntry = document.createElement('div');
          logEntry.className = 'debug-entry success';
          logEntry.innerHTML = `<span class="timestamp">[${timestamp}]</span> Successfully parsed JSON response: ${JSON.stringify(json, null, 2).substring(0, 200)}...`;
          debugOutput.appendChild(logEntry);
        }
        
        message.className = 'api-message success';
        message.innerHTML = 'API test successful! Received valid JSON response.';
        
        // Add a further diagnostic for analyze responses
        if (endpoint === 'analyze' && json) {
          message.innerHTML += '<br>Response format:<br>';
          message.innerHTML += `<code>score: ${json.score !== undefined ? json.score : 'undefined'}, `;
          message.innerHTML += `category: ${json.category !== undefined ? json.category : 'undefined'}, `;
          message.innerHTML += `label: ${json.label !== undefined ? json.label : 'undefined'}</code>`;
          
          // Show how the score would be mapped by our function
          const mappedScore = mapScoreToRange(json.score, json.category);
          message.innerHTML += `<br>This maps to internal sentiment value: ${mappedScore.toFixed(2)}`;
        }
        
      } catch (e) {
        console.error('Error parsing JSON:', e);
        if (debugOutput) {
          const timestamp = new Date().toLocaleTimeString();
          const logEntry = document.createElement('div');
          logEntry.className = 'debug-entry error';
          logEntry.innerHTML = `<span class="timestamp">[${timestamp}]</span> Failed to parse response as JSON. Raw response: ${text.substring(0, 200)}...`;
          debugOutput.appendChild(logEntry);
        }
        
        message.className = 'api-message warning';
        message.innerHTML = 'API responded but returned invalid JSON. See debug console for details.';
      }
      
    })
    .catch(error => {
      console.error('Fetch error:', error);
      if (debugOutput) {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = 'debug-entry error';
        logEntry.innerHTML = `<span class="timestamp">[${timestamp}]</span> Error: ${error.message}`;
        debugOutput.appendChild(logEntry);
      }
      
      message.className = 'api-message error';
      message.innerHTML = `Direct API test failed: ${error.message}. See debug console for details.`;
    });
    
    // Add to the analyze container
    const analyzeContainer = document.querySelector('.analyze-container');
    if (analyzeContainer) {
      analyzeContainer.appendChild(message);
    }
    
    // Create and add a diagnostic button to the debug container
    const debugContainer = document.querySelector('.debug-container');
    if (debugContainer) {
      const diagnosticBtn = document.createElement('button');
      diagnosticBtn.textContent = 'Run API Diagnostics';
      diagnosticBtn.className = 'debug-btn';
      diagnosticBtn.onclick = function() {
        testFetchToEndpoint('health');
        setTimeout(() => testFetchToEndpoint('analyze', 'simple'), 1000);
      };
      
      // Only add if it doesn't exist yet
      if (!document.querySelector('.debug-btn')) {
        debugContainer.appendChild(diagnosticBtn);
      }
    }
  }
  
  // Function to directly test the /analyze endpoint with precise error reporting
  async function directAnalyzeTest() {
    // Get a reference to the debug console
    const debugOutput = document.getElementById('debug-output');
    
    // Helper function to log to debug console
    function logDebug(message, className = '') {
      console.log(message);
      if (debugOutput) {
        const timestamp = new Date().toLocaleTimeString();
        const entry = document.createElement('div');
        entry.className = `debug-entry ${className}`;
        entry.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;
        debugOutput.appendChild(entry);
        debugOutput.scrollTop = debugOutput.scrollHeight;
      }
    }
    
    try {
      // Get API URL from storage
      const apiUrl = await new Promise(resolve => {
        chrome.storage.local.get(['apiUrl'], function(data) {
          resolve(data.apiUrl || 'http://localhost:5000');
        });
      });
      
      // Format URL
      const baseUrl = apiUrl.endsWith('/') ? apiUrl.slice(0, -1) : apiUrl;
      const analyzeUrl = `${baseUrl}/analyze`;
      
      logDebug(`DIRECT TEST: Sending request to ${analyzeUrl}`, 'highlight');
      
      // Send a raw XHR request to see exact response
      const rawResponse = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', analyzeUrl, true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.setRequestHeader('Accept', 'application/json');
        
        // Log all events for debugging
        xhr.onreadystatechange = function() {
          logDebug(`XHR state changed: readyState=${xhr.readyState}, status=${xhr.status || 'N/A'}`);
        };
        
        xhr.onload = function() {
          logDebug(`XHR loaded: status=${xhr.status}, statusText=${xhr.statusText}`);
          logDebug(`Response headers: ${xhr.getAllResponseHeaders()}`, 'small');
          resolve({
            status: xhr.status,
            statusText: xhr.statusText,
            headers: xhr.getAllResponseHeaders(),
            body: xhr.responseText
          });
        };
        
        xhr.onerror = function(e) {
          logDebug(`XHR error: ${e}`, 'error');
          reject(new Error(`Network error: ${e}`));
        };
        
        xhr.ontimeout = function() {
          logDebug('XHR request timed out', 'error');
          reject(new Error('Request timed out'));
        };
        
        // Set timeout to 10 seconds
        xhr.timeout = 10000;
        
        // Send the request with a test payload
        const data = JSON.stringify({
          text: "This is a direct XHR test of the analyze endpoint.",
          model_type: "simple"
        });
        
        logDebug(`Request payload: ${data}`);
        xhr.send(data);
      });
      
      // Process the raw response
      logDebug(`Raw response status: ${rawResponse.status} ${rawResponse.statusText}`);
      logDebug(`Raw response body: ${rawResponse.body.substring(0, 200)}`);
      
      // Try to parse the JSON response
      try {
        const jsonResponse = safeParseJson(rawResponse.body);
        if (jsonResponse) {
          logDebug(`Parsed JSON response: ${JSON.stringify(jsonResponse, null, 2)}`, 'success');
          
          // Check if the response has the expected format
          if ('score' in jsonResponse && 'category' in jsonResponse && 'label' in jsonResponse) {
            logDebug('Response format is correct ✓', 'success highlight');
            
            // Show how the score would be mapped
            const mappedScore = mapScoreToRange(jsonResponse.score, jsonResponse.category);
            logDebug(`Score ${jsonResponse.score} with category ${jsonResponse.category} maps to: ${mappedScore.toFixed(2)}`, 'highlight');
            
            // Save this successful mapping to storage for reference
            chrome.storage.local.set({ 
              lastSuccessfulAnalysis: {
                timestamp: new Date().toISOString(),
                apiUrl: analyzeUrl,
                request: {
                  text: "This is a direct XHR test of the analyze endpoint.",
                  model_type: "simple"
                },
                response: jsonResponse,
                mappedScore: mappedScore
              }
            });
            
            return {
              success: true,
              data: jsonResponse,
              mappedScore: mappedScore
            };
          } else {
            logDebug('Response is missing expected fields (score, category, label)', 'error');
            logDebug(`Found properties: ${Object.keys(jsonResponse).join(', ')}`, 'error');
          }
        } else {
          logDebug('Failed to parse response as JSON', 'error');
        }
      } catch (parseError) {
        logDebug(`Error parsing JSON: ${parseError}`, 'error');
      }
      
    } catch (error) {
      logDebug(`Direct test error: ${error.message}`, 'error');
      logDebug(`Error stack: ${error.stack}`, 'small error');
    }
    
    // If we get here, the test failed
    return { success: false };
  }
  
  // Add button to launch test
  function addDirectTestButton() {
    const debugContainer = document.querySelector('.debug-container');
    if (!debugContainer) return;
    
    // Create direct test button
    const directTestBtn = document.createElement('button');
    directTestBtn.textContent = 'Run XHR Direct Test';
    directTestBtn.className = 'debug-btn xhr-test';
    directTestBtn.style.marginLeft = '8px';
    directTestBtn.onclick = async function() {
      const result = await directAnalyzeTest();
      console.log('Direct test result:', result);
    };
    
    // Add to container if not already added
    if (!document.querySelector('.xhr-test')) {
      debugContainer.appendChild(directTestBtn);
    }
  }
  
  // Add direct test functionality when debug pane is opened
  document.addEventListener('DOMContentLoaded', function() {
    const toggleDebugBtn = document.getElementById('toggle-debug');
    if (toggleDebugBtn) {
      toggleDebugBtn.addEventListener('click', function() {
        setTimeout(addDirectTestButton, 100);
      });
    }
  });

  // Cache API health status
  let lastHealthCheckTime = 0;
  const HEALTH_CHECK_INTERVAL = 60000; // Only check once per minute

  // Optimized health check function to reduce excessive API calls
  async function checkApiHealthWithCache() {
    // Check if we've done a health check recently
    const now = Date.now();
    const cachedStatus = await new Promise(resolve => {
      chrome.storage.local.get(['apiStatus', 'lastHealthCheckTime'], function(data) {
        resolve({
          status: data.apiStatus || 'unknown',
          timestamp: data.lastHealthCheckTime || 0
        });
      });
    });
    
    // If we've checked recently and API was online, use cached result
    if (cachedStatus.timestamp > now - HEALTH_CHECK_INTERVAL && cachedStatus.status === 'online') {
      console.log('Using cached API health status: online');
      return { status: 'online' };
    }
    
    // Otherwise, perform a real health check
    console.log('Performing fresh API health check');
    const result = await checkBackendHealth();
    
    // Save when we did this check
    chrome.storage.local.set({ lastHealthCheckTime: now });
    
    return result;
  }

  // Other global functions...

} catch (error) {
  console.error('Fatal error in Mood Map extension:', error);
  // Display error directly in popup
  window.addEventListener('DOMContentLoaded', () => {
    document.body.innerHTML = `<div style="color:red; padding:20px; border:2px solid red;">
      <h3>Extension Initialization Error</h3>
      <p>${error.message}</p>
      <p>Stack trace: ${error.stack}</p>
      <button onclick="location.reload()">Reload Extension</button>
    </div>`;
  });
}