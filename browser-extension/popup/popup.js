document.addEventListener('DOMContentLoaded', function() {
  // API configuration
  let apiBaseUrl = 'http://127.0.0.1:5000';
  let apiSecure = false;
  
  // UI state
  let currentTab = 'analyze';
  let sentimentHistory = [];
  let wordCloudInstance = null;
  let timelineChart = null;
  let interactiveChart = null;
  
  // Show content, hide loading
  setTimeout(() => {
    document.getElementById('content').style.display = 'block';
    document.getElementById('loading').style.display = 'none';
  }, 500);
  
  // Load API configuration from storage
  chrome.storage.local.get(['apiBaseUrl', 'apiSecure', 'darkMode'], function(result) {
    // API Settings
    if (result.apiBaseUrl) {
      apiBaseUrl = result.apiBaseUrl;
    }
    
    if (result.apiSecure !== undefined) {
      apiSecure = result.apiSecure;
    }
    
    updateApiDisplay();
    
    // Dark Mode
    const darkModeEnabled = result.darkMode === 'enabled';
    if (darkModeEnabled) {
      document.body.classList.add('dark-theme');
      const themeSelector = document.getElementById('theme-selector');
      if (themeSelector) {
        themeSelector.value = 'dark';
      }
    }
    
    // Test API connection on startup
    testApiConnection();
  });
  
  // Helper function to get API URL
  function getApiUrl(endpoint) {
    return `${apiSecure ? 'https://' : 'http://'}${apiBaseUrl}${endpoint}`;
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
  
  // Switch tab function
  function switchTab(tabId) {
    currentTab = tabId;
    
    // Update active tab button
    tabButtons.forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
    });
    
    // Show active tab content
    tabContents.forEach(content => {
      const isActive = content.id === `${tabId}-tab`;
      content.classList.toggle('active', isActive);
      
      if (isActive) {
        content.style.opacity = '0';
        setTimeout(() => {
          content.style.opacity = '1';
        }, 50);
      }
    });

    // Load history when switching to history tab
    if (tabId === 'history') {
      loadAndDisplayHistory();
    }
    
    // Load visualizations when switching to visualize tab
    if (tabId === 'visualize') {
      loadAndDisplayVisualizations();
    }
  }
  
  // Initialize theme selector
  const themeSelector = document.getElementById('theme-selector');
  if (themeSelector) {
    themeSelector.addEventListener('change', function() {
      const selectedTheme = this.value;
      
      if (selectedTheme === 'dark') {
        document.body.classList.add('dark-theme');
        chrome.storage.local.set({ darkMode: 'enabled' });
      } else if (selectedTheme === 'light') {
        document.body.classList.remove('dark-theme');
        chrome.storage.local.set({ darkMode: 'disabled' });
      } else if (selectedTheme === 'system') {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        document.body.classList.toggle('dark-theme', prefersDark);
        chrome.storage.local.set({ darkMode: prefersDark ? 'enabled' : 'disabled' });
      }
    });
  }

  // Initialize auto analyze toggle
  const autoAnalyzeToggle = document.getElementById('auto-analyze-toggle');
  if (autoAnalyzeToggle) {
    chrome.storage.local.get(['autoAnalyze'], function(result) {
      autoAnalyzeToggle.checked = result.autoAnalyze === 'enabled';
    });
    
    autoAnalyzeToggle.addEventListener('change', function() {
      chrome.storage.local.set({ autoAnalyze: this.checked ? 'enabled' : 'disabled' });
      logToDebug(`Auto analyze ${this.checked ? 'enabled' : 'disabled'}`);
      
      chrome.runtime.sendMessage({
        type: 'updateAutoAnalyze',
        enabled: this.checked
      });
    });
  }
  
  // Initialize model selector
  const modelSelector = document.getElementById('model-selector');
  if (modelSelector) {
    chrome.storage.local.get(['selectedModel'], function(result) {
      if (result.selectedModel) {
        modelSelector.value = result.selectedModel;
      }
    });
    
    modelSelector.addEventListener('change', function() {
      chrome.storage.local.set({ selectedModel: this.value });
      logToDebug(`Analysis mode changed to: ${this.value}`);
    });
  }
  
  // Initialize analyze button
  const analyzeButton = document.getElementById('analyze-button');
  const textInput = document.getElementById('text-input');
  
  if (analyzeButton && textInput) {
    analyzeButton.addEventListener('click', function() {
      const text = textInput.value.trim();
      if (text) {
        analyzeSentiment(text);
      } else {
        showToast('Please enter some text to analyze', 'error');
      }
    });
    
    textInput.addEventListener('keydown', function(e) {
      // Analyze on Ctrl+Enter
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        const text = textInput.value.trim();
        if (text) {
          analyzeSentiment(text);
        }
      }
    });
  }
  
  // Initialize debug toggle
  const toggleDebug = document.getElementById('toggle-debug');
  const debugContainer = document.querySelector('.debug-container');
  
  if (toggleDebug && debugContainer) {
    toggleDebug.addEventListener('click', function() {
      const isVisible = debugContainer.style.display !== 'none';
      debugContainer.style.display = isVisible ? 'none' : 'block';
      toggleDebug.textContent = isVisible ? 'Debug' : 'Hide Debug';
    });
  }
  
  // Initialize clear debug button
  const clearDebugBtn = document.getElementById('clear-debug-btn');
  const debugOutput = document.getElementById('debug-output');
  
  if (clearDebugBtn && debugOutput) {
    clearDebugBtn.addEventListener('click', function() {
      debugOutput.value = '';
      logToDebug('Debug console cleared');
    });
  }
  
  // Initialize clear history button
  const clearHistoryBtn = document.getElementById('clear-history-btn');
  
  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', function() {
      sentimentHistory = [];
      chrome.storage.local.set({ 'sentimentHistory': [] });
      displayHistory();
      showToast('History has been cleared', 'success');
    });
  }
  
  // Initialize API settings
  const updateApiUrlBtn = document.getElementById('update-api-url-btn');
  const apiProtocolSelect = document.getElementById('api-protocol-select');
  const apiUrlInput = document.getElementById('api-url-input');
  const testApiBtn = document.getElementById('test-api-btn');
  
  if (updateApiUrlBtn && apiUrlInput) {
    // Initialize input with current values
    apiUrlInput.value = apiBaseUrl;
    if (apiProtocolSelect) {
      apiProtocolSelect.value = apiSecure ? 'https' : 'http';
    }
    
    updateApiUrlBtn.addEventListener('click', function() {
      const newUrl = apiUrlInput.value.trim();
      if (!newUrl) {
        showToast('Please enter a valid API URL', 'error');
        return;
      }
      
      apiBaseUrl = newUrl.replace(/^https?:\/\//, ''); // Remove protocol if included
      apiSecure = apiProtocolSelect ? apiProtocolSelect.value === 'https' : false;
      
      chrome.storage.local.set({ 
        apiBaseUrl: apiBaseUrl,
        apiSecure: apiSecure
      });
      
      updateApiDisplay();
      testApiConnection();
      
      logToDebug(`API URL updated to: ${getApiUrl('')}`);
    });
  }
  
  if (testApiBtn) {
    testApiBtn.addEventListener('click', function() {
      testApiConnection();
    });
  }
  
  // Function to update the API URL display
  function updateApiDisplay() {
    const apiUrlDisplay = document.getElementById('api-url-display');
    if (apiUrlDisplay) {
      apiUrlDisplay.textContent = getApiUrl('');
    }
  }
  
  // Function to test API connection
  function testApiConnection() {
    const apiStatus = document.getElementById('api-status');
    const apiMessage = document.getElementById('api-message');
    
    if (apiStatus) {
      apiStatus.className = 'api-status unknown';
      apiStatus.textContent = 'Testing...';
    }
    
    if (apiMessage) {
      apiMessage.className = 'api-message';
      apiMessage.textContent = '';
    }
    
    logToDebug(`Testing API connection to ${getApiUrl('/health')}...`);
    
    fetch(getApiUrl('/health'))
      .then(response => {
        if (!response.ok) {
          throw new Error(`API returned ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        logToDebug(`API is online: ${JSON.stringify(data)}`);
        
        if (apiStatus) {
          apiStatus.className = 'api-status online';
          apiStatus.textContent = 'Online';
        }
        
        if (apiMessage) {
          apiMessage.className = 'api-message success';
          apiMessage.textContent = 'Connection successful! API is responding correctly.';
        }
      })
      .catch(error => {
        logToDebug(`API test failed: ${error.message}`);
        
        if (apiStatus) {
          apiStatus.className = 'api-status offline';
          apiStatus.textContent = 'Offline';
        }
        
        if (apiMessage) {
          apiMessage.className = 'api-message error';
          apiMessage.textContent = 'Connection failed. Check that the backend server is running.';
        }
        
        // Try alternative endpoint
        tryFallbackEndpoint();
      });
  }
  
  // Try fallback endpoint if health check fails
  function tryFallbackEndpoint() {
    fetch(getApiUrl('/'))
      .then(response => response.json())
      .then(data => {
        logToDebug(`API root endpoint is responding: ${JSON.stringify(data)}`);
        
        const apiStatus = document.getElementById('api-status');
        const apiMessage = document.getElementById('api-message');
        
        if (apiStatus) {
          apiStatus.className = 'api-status online';
          apiStatus.textContent = 'Online';
        }
        
        if (apiMessage) {
          apiMessage.className = 'api-message success';
          apiMessage.textContent = 'Connection successful via root endpoint!';
        }
      })
      .catch(() => {
        logToDebug('All connection attempts failed');
      });
  }
  
  // Toast notification function
  function showToast(message, type) {
    const toastId = `toast-${type}`;
    let toast = document.getElementById(toastId);
    
    if (!toast) {
      toast = document.createElement('div');
      toast.id = toastId;
      toast.className = `toast ${type}`;
      document.body.appendChild(toast);
    }
    
    toast.textContent = message;
    toast.classList.add('show');
    
    setTimeout(() => {
      toast.classList.remove('show');
    }, 3000);
  }
  
  // Log to debug console
  function logToDebug(message) {
    const debugOutput = document.getElementById('debug-output');
    if (debugOutput) {
      const timestamp = new Date().toLocaleTimeString();
      debugOutput.value += `[${timestamp}] ${message}\n`;
      debugOutput.scrollTop = debugOutput.scrollHeight;
    }
  }
  
  // Sentiment analysis function
  function analyzeSentiment(text) {
    const sentimentCircle = document.querySelector('.sentiment-circle');
    const sentimentLabel = document.querySelector('.sentiment-label');
    const sentimentScore = document.getElementById('sentiment-score');
    const summaryText = document.getElementById('summary-text');
    
    // Show loading state
    sentimentCircle.className = 'sentiment-circle analyzing';
    sentimentLabel.textContent = 'Analyzing...';
    sentimentScore.textContent = '...';
    
    if (summaryText) {
      summaryText.textContent = 'Generating summary...';
    }
    
    // Get selected model
    const selectedModel = modelSelector ? modelSelector.value : 'default';
    
    // Use basic offline analysis if selected
    if (selectedModel === 'basic') {
      const result = performBasicSentimentAnalysis(text);
      displaySentimentResult(result, text);
      addToHistory(text, result);
      return;
    }
    
    // Create a timeout promise
    const timeoutPromise = new Promise((_, reject) => 
      setTimeout(() => reject(new Error('API request timed out')), 5000)
    );
    
    logToDebug(`Analyzing text: "${text.substring(0, 30)}${text.length > 30 ? '...' : ''}"`);
    
    // API call to backend
    Promise.race([
      fetch(getApiUrl('/analyze'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text, model: selectedModel }),
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`API returned ${response.status}: ${response.statusText}`);
        }
        return response.json();
      }),
      timeoutPromise
    ])
    .then(data => {
      logToDebug('Analysis successful');
      
      // Process the API response
      const result = processApiResponse(data);
      
      // Display results and save to history
      displaySentimentResult(result, text);
      addToHistory(text, result);
    })
    .catch(error => {
      logToDebug(`Error analyzing sentiment: ${error.message}`);
      showToast('Failed to analyze text. Using offline analysis instead.', 'error');
      
      // Fallback to basic analysis
      const result = performBasicSentimentAnalysis(text);
      result.offline = true;
      
      displaySentimentResult(result, text);
      addToHistory(text, result);
    });
  }
  
  // Process API response to standardized format
  function processApiResponse(data) {
    // Extract or calculate score (-1 to 1)
    let score;
    if (data.score !== undefined) {
      score = data.score;
    } else if (data.sentiment_percentage !== undefined) {
      score = (data.sentiment_percentage / 50) - 1;
    } else if (data.prediction !== undefined) {
      score = mapPredictionToScore(data.prediction);
    } else {
      score = 0;
    }
    
    // Determine category
    const category = determineCategory(score);
    
    // Get summary text
    const summary = data.summary || `Text shows ${category.label.toLowerCase()} sentiment (${Math.round((score + 1) * 50)}% positive).`;
    
    return {
      score,
      category,
      summary
    };
  }
  
  // Map API prediction value to standardized score
  function mapPredictionToScore(prediction) {
    const predictionNum = parseInt(prediction);
    
    switch (predictionNum) {
      case 0: return -0.9;
      case 1: return -0.5;
      case 2: return 0.0;
      case 3: return 0.5;
      case 4: return 0.9;
      default: return 0.0;
    }
  }
  
  // Determine sentiment category based on score
  function determineCategory(score) {
    if (score < -0.6) {
      return {
        id: 'overwhelmingly-negative',
        label: 'Very Negative',
        emoji: '😢'
      };
    } else if (score < -0.2) {
      return {
        id: 'negative',
        label: 'Negative',
        emoji: '😞'
      };
    } else if (score < 0.2) {
      return {
        id: 'neutral',
        label: 'Neutral',
        emoji: '😐'
      };
    } else if (score < 0.6) {
      return {
        id: 'positive',
        label: 'Positive',
        emoji: '😊'
      };
    } else {
      return {
        id: 'overwhelmingly-positive',
        label: 'Very Positive',
        emoji: '😁'
      };
    }
  }
  
  // Basic offline sentiment analysis
  function performBasicSentimentAnalysis(text) {
    logToDebug('Using basic offline sentiment analysis');
    
    const positiveWords = ['good', 'great', 'excellent', 'happy', 'love', 'nice', 
                          'wonderful', 'awesome', 'fantastic', 'positive', 'best',
                          'amazing', 'brilliant', 'perfect', 'delighted'];
                          
    const negativeWords = ['bad', 'terrible', 'awful', 'sad', 'hate', 'poor', 
                          'negative', 'horrible', 'wrong', 'fail', 'worst',
                          'disappointed', 'frustrating', 'useless', 'annoying'];
    
    text = text.toLowerCase();
    let positiveCount = 0;
    let negativeCount = 0;
    
    // Count positive and negative words
    positiveWords.forEach(word => {
      const regex = new RegExp('\\b' + word + '\\b', 'gi');
      const matches = text.match(regex);
      if (matches) positiveCount += matches.length;
    });
    
    negativeWords.forEach(word => {
      const regex = new RegExp('\\b' + word + '\\b', 'gi');
      const matches = text.match(regex);
      if (matches) negativeCount += matches.length;
    });
    
    // Calculate simple score (-1 to 1)
    let score;
    if (positiveCount === 0 && negativeCount === 0) {
      score = 0;
    } else {
      score = (positiveCount - negativeCount) / Math.max(1, positiveCount + negativeCount);
    }
    
    // Clamp score between -1 and 1
    const clampedScore = Math.max(-1, Math.min(1, score));
    
    // Determine category
    const category = determineCategory(clampedScore);
    
    return {
      score: clampedScore,
      category,
      summary: `This text appears to have ${category.label.toLowerCase()} sentiment based on keyword analysis.`,
      positiveWordCount: positiveCount,
      negativeWordCount: negativeCount
    };
  }
  
  // Display sentiment analysis result
  function displaySentimentResult(result, text) {
    const sentimentCircle = document.querySelector('.sentiment-circle');
    const sentimentLabel = document.querySelector('.sentiment-label');
    const sentimentScore = document.getElementById('sentiment-score');
    const summaryText = document.getElementById('summary-text');
    
    // Clear previous classes and set new one
    sentimentCircle.className = 'sentiment-circle';
    setTimeout(() => {
      sentimentCircle.classList.add(result.category.id);
    }, 10);
    
    // Update with emoji instead of score
    sentimentScore.textContent = result.category.emoji;
    sentimentScore.setAttribute('data-score', result.score.toFixed(2));
    
    // Add scale animation
    sentimentScore.style.transform = 'scale(0.8)';
    setTimeout(() => {
      sentimentScore.style.transform = 'scale(1.3)';
      setTimeout(() => {
        sentimentScore.style.transform = '';
      }, 200);
    }, 100);
    
    // Update label with score in text
    const offlineTag = result.offline ? ' (Offline)' : '';
    sentimentLabel.textContent = `${result.category.label}${offlineTag} (${result.score.toFixed(2)})`;
    
    // Update summary
    if (summaryText) {
      summaryText.textContent = result.summary;
    }
    
    // Update visualization bars
    updateSentimentBars(result.score);
  }
  
  // Update sentiment visualization bars
  function updateSentimentBars(score) {
    // Calculate values for each bar based on score
    const negativeValue = Math.max(0, Math.min(100, (-score + 1) / 2 * 100));
    const neutralValue = Math.max(0, Math.min(100, (1 - Math.abs(score)) * 100));
    const positiveValue = Math.max(0, Math.min(100, (score + 1) / 2 * 100));
    
    // Update negative bar
    const negativeBar = document.getElementById('negative-bar');
    const negativeValue_el = document.getElementById('negative-value');
    if (negativeBar && negativeValue_el) {
      setTimeout(() => {
        negativeBar.style.width = `${negativeValue}%`;
        negativeValue_el.textContent = `${Math.round(negativeValue)}%`;
      }, 100);
    }
    
    // Update neutral bar
    const neutralBar = document.getElementById('neutral-bar');
    const neutralValue_el = document.getElementById('neutral-value');
    if (neutralBar && neutralValue_el) {
      setTimeout(() => {
        neutralBar.style.width = `${neutralValue}%`;
        neutralValue_el.textContent = `${Math.round(neutralValue)}%`;
      }, 200);
    }
    
    // Update positive bar
    const positiveBar = document.getElementById('positive-bar');
    const positiveValue_el = document.getElementById('positive-value');
    if (positiveBar && positiveValue_el) {
      setTimeout(() => {
        positiveBar.style.width = `${positiveValue}%`;
        positiveValue_el.textContent = `${Math.round(positiveValue)}%`;
      }, 300);
    }
  }
  
  // Add entry to history
  function addToHistory(text, result) {
    const entry = {
      text: text.length > 70 ? text.substring(0, 67) + '...' : text,
      fullText: text,
      score: result.score,
      category: result.category.id,
      timestamp: new Date().toISOString(),
      offline: result.offline || false
    };
    
    // Load existing history first
    chrome.storage.local.get(['sentimentHistory'], function(data) {
      let history = [];
      
      if (data.sentimentHistory) {
        try {
          history = data.sentimentHistory;
        } catch (e) {
          logToDebug('Error loading history, creating new one');
        }
      }
      
      // Add new entry at the beginning
      history.unshift(entry);
      
      // Limit to 50 entries
      if (history.length > 50) {
        history = history.slice(0, 50);
      }
      
      // Update global variable and save to storage
      sentimentHistory = history;
      chrome.storage.local.set({ 'sentimentHistory': history });
      
      // Update display if we're on history tab
      if (currentTab === 'history') {
        displayHistory();
      }
    });
  }
  
  // Load and display history
  function loadAndDisplayHistory() {
    chrome.storage.local.get(['sentimentHistory'], function(result) {
      if (result.sentimentHistory) {
        try {
          sentimentHistory = result.sentimentHistory;
          displayHistory();
        } catch (e) {
          logToDebug('Failed to load history');
          sentimentHistory = [];
        }
      }
    });
  }
  
  // Display history entries
  function displayHistory() {
    const historyList = document.getElementById('history-list');
    if (!historyList) return;
    
    // Clear current list
    historyList.innerHTML = '';
    
    // Show empty message if no history
    if (!sentimentHistory || sentimentHistory.length === 0) {
      historyList.innerHTML = '<div class="empty-history-message">No history items yet. Analyze some text to see it here.</div>';
      return;
    }
    
    // Filter based on selected option
    const historyFilter = document.getElementById('history-filter');
    const filterValue = historyFilter ? historyFilter.value : 'all';
    
    let filteredHistory = [...sentimentHistory];
    
    if (filterValue === 'today') {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      
      filteredHistory = filteredHistory.filter(entry => {
        const entryDate = new Date(entry.timestamp);
        return entryDate >= today;
      });
    } else if (filterValue === 'week') {
      const weekAgo = new Date();
      weekAgo.setDate(weekAgo.getDate() - 7);
      
      filteredHistory = filteredHistory.filter(entry => {
        const entryDate = new Date(entry.timestamp);
        return entryDate >= weekAgo;
      });
    }
    
    // Show message if no items after filtering
    if (filteredHistory.length === 0) {
      historyList.innerHTML = '<div class="empty-history-message">No history items for the selected time period.</div>';
      return;
    }
    
    // Create history items
    filteredHistory.forEach(entry => {
      const item = document.createElement('div');
      item.className = 'history-item';
      
      const timestamp = new Date(entry.timestamp).toLocaleString();
      const scoreClass = `text-${entry.category}`;
      
      item.innerHTML = `
        <div class="history-text">${entry.text}</div>
        <div class="history-score ${scoreClass}">${entry.score.toFixed(2)}</div>
        <div class="history-timestamp">${timestamp}</div>
      `;
      
      // Click to load text into input
      item.addEventListener('click', () => {
        const textInput = document.getElementById('text-input');
        if (textInput) {
          textInput.value = entry.fullText;
          switchTab('analyze');
        }
      });
      
      historyList.appendChild(item);
    });
  }
  
  // Load and display visualizations
  function loadAndDisplayVisualizations() {
    chrome.storage.local.get(['sentimentHistory'], function(result) {
      if (result.sentimentHistory) {
        try {
          sentimentHistory = result.sentimentHistory;
          
          // Filter based on selected option
          const visualizationFilter = document.getElementById('visualization-filter');
          const filterValue = visualizationFilter ? visualizationFilter.value : 'all';
          const filteredHistory = filterHistoryByTime(sentimentHistory, filterValue);
          
          if (filteredHistory.length > 0) {
            renderTimelineChart(filteredHistory);
            renderWordCloud(filteredHistory);
            renderInteractiveGraph(filteredHistory);
          } else {
            displayEmptyVisualization('timeline');
            displayEmptyVisualization('wordcloud');
            displayEmptyVisualization('interactive-graph');
          }
        } catch (e) {
          logToDebug('Failed to load visualizations: ' + e.message);
          displayEmptyVisualization('timeline');
          displayEmptyVisualization('wordcloud');
          displayEmptyVisualization('interactive-graph');
        }
      } else {
        displayEmptyVisualization('timeline');
        displayEmptyVisualization('wordcloud');
        displayEmptyVisualization('interactive-graph');
      }
    });
  }
  
  // Filter history by time period
  function filterHistoryByTime(history, filterType) {
    if (!history || !filterType || filterType === 'all') {
      return history;
    }
    
    let cutoffDate = new Date();
    
    if (filterType === 'today') {
      cutoffDate.setHours(0, 0, 0, 0);
    } else if (filterType === 'week') {
      cutoffDate.setDate(cutoffDate.getDate() - 7);
    } else if (filterType === 'month') {
      cutoffDate.setMonth(cutoffDate.getMonth() - 1);
    }
    
    return history.filter(entry => {
      const entryDate = new Date(entry.timestamp);
      return entryDate >= cutoffDate;
    });
  }
  
  // Display empty visualization message
  function displayEmptyVisualization(type) {
    if (type === 'timeline') {
      const timelineContainer = document.querySelector('.timeline-container');
      if (timelineContainer) {
        timelineContainer.innerHTML = '<div class="empty-visualization-message">No data available for the selected time period.</div>';
      }
    } else if (type === 'wordcloud') {
      const wordcloudContainer = document.getElementById('sentiment-wordcloud');
      if (wordcloudContainer) {
        wordcloudContainer.innerHTML = '<div class="empty-visualization-message">No data available for the selected time period.</div>';
      }
    } else if (type === 'interactive-graph') {
      const graphContainer = document.querySelector('.interactive-graph-container');
      if (graphContainer) {
        graphContainer.innerHTML = '<div class="empty-visualization-message">No data available for the selected time period.</div>';
      }
    }
  }
  
  // Timeline chart rendering
  function renderTimelineChart(history) {
    const timelineCanvas = document.getElementById('sentiment-timeline');
    if (!timelineCanvas) return;
    
    // Clear previous chart
    if (timelineChart) {
      timelineChart.destroy();
    }
    
    // Prepare data: most recent entries first in array, so reverse for chronological display
    // We'll limit to 30 entries maximum for readability
    const chartData = [...history].reverse().slice(0, 30);
    
    // Extract timestamps and scores
    const labels = chartData.map(entry => {
      const date = new Date(entry.timestamp);
      return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    });
    
    const scores = chartData.map(entry => entry.score);
    
    // Calculate gradient colors based on sentiment scores
    const pointBackgrounds = scores.map(score => {
      if (score < -0.6) return '#FF5252'; // Very negative
      if (score < -0.2) return '#FF9E80'; // Negative
      if (score < 0.2) return '#BDBDBD';  // Neutral
      if (score < 0.6) return '#A5D6A7';  // Positive
      return '#00C853';                   // Very positive
    });
    
    // Get colors based on theme
    const isDarkTheme = document.body.classList.contains('dark-theme');
    const gridColor = isDarkTheme ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    const textColor = isDarkTheme ? 'rgba(255, 255, 255, 0.7)' : 'rgba(0, 0, 0, 0.7)';
    
    // Create the chart
    timelineChart = new Chart(timelineCanvas, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Sentiment Score',
          data: scores,
          fill: false,
          borderColor: '#4285F4',
          tension: 0.2,
          pointBackgroundColor: pointBackgrounds,
          pointRadius: 5,
          pointHoverRadius: 7
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 1000,
          easing: 'easeOutQuart'
        },
        scales: {
          y: {
            min: -1,
            max: 1,
            grid: {
              color: gridColor
            },
            ticks: {
              color: textColor,
              callback: function(value) {
                if (value === -1) return 'Very Negative';
                if (value === -0.5) return 'Negative';
                if (value === 0) return 'Neutral';
                if (value === 0.5) return 'Positive';
                if (value === 1) return 'Very Positive';
                return '';
              }
            }
          },
          x: {
            grid: {
              color: gridColor
            },
            ticks: {
              color: textColor,
              maxRotation: 45,
              minRotation: 45
            }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              afterLabel: function(context) {
                const entry = chartData[context.dataIndex];
                return entry.text;
              }
            }
          },
          legend: {
            display: false
          }
        }
      }
    });
  }
  
  // Word cloud rendering
  function renderWordCloud(history) {
    const wordcloudContainer = document.getElementById('sentiment-wordcloud');
    if (!wordcloudContainer) return;
    
    // Clear previous word cloud
    wordcloudContainer.innerHTML = '';
    
    // Get text from all entries
    const allText = history.map(entry => entry.fullText || entry.text).join(' ');
    
    // Process text into word frequencies
    const words = processTextForWordCloud(allText);
    
    if (words.length === 0) {
      displayEmptyVisualization('wordcloud');
      return;
    }
    
    // Calculate container dimensions
    const width = wordcloudContainer.clientWidth;
    const height = wordcloudContainer.clientHeight;
    
    // Create SVG element
    const svg = d3.select(wordcloudContainer)
      .append('svg')
      .attr('width', width)
      .attr('height', height);
      
    const g = svg.append('g')
      .attr('transform', `translate(${width / 2}, ${height / 2})`);
    
    // Create cloud layout
    const layout = d3.layout.cloud()
      .size([width, height])
      .words(words)
      .padding(5)
      .rotate(() => (Math.random() < 0.5 ? 0 : 90 * (Math.round(Math.random()) * 2 - 1)))
      .font('Arial')
      .fontSize(d => Math.sqrt(d.value) * 5)
      .on('end', draw);
    
    // Start layout calculation
    layout.start();
    
    // Draw word cloud
    function draw(words) {
      g.selectAll('text')
        .data(words)
        .enter()
        .append('text')
        .attr('class', 'word-cloud-word')
        .style('fill', d => d.color)
        .style('font-family', 'Arial')
        .style('font-size', d => `${d.size}px`)
        .style('opacity', 0)
        .attr('text-anchor', 'middle')
        .attr('transform', d => `translate(${d.x}, ${d.y}) rotate(${d.rotate})`)
        .text(d => d.text)
        .transition()
        .delay((d, i) => i * 20)
        .style('opacity', 1)
        .on('end', function(d, i) {
          d3.select(this)
            .style('cursor', 'pointer')
            .append('title')
            .text(d => `${d.text}: ${d.count} occurrences (score: ${d.sentimentScore.toFixed(2)})`);
        });
    }
  }
  
  // Process text for word cloud
  function processTextForWordCloud(text) {
    // Stop words to filter out
    const stopWords = new Set([
      'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 
      'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
      'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 
      'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 
      'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'me', 'more', 'most', 'my', 'myself', 
      'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 
      'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 
      'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 
      'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 
      'who', 'whom', 'why', 'will', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves'
    ]);
    
    // Positive and negative word lists for sentiment scoring
    const positiveWords = new Set([
      'good', 'great', 'excellent', 'happy', 'love', 'nice', 'wonderful', 'awesome', 'fantastic', 
      'positive', 'best', 'amazing', 'brilliant', 'perfect', 'delighted', 'joy', 'success', 'beautiful', 
      'enjoy', 'liked', 'favorite', 'helpful', 'impressive', 'win', 'winning', 'praise', 'thank', 
      'thanks', 'pleased', 'pleasure', 'exciting', 'excited', 'hope', 'hopeful'
    ]);
    
    const negativeWords = new Set([
      'bad', 'terrible', 'awful', 'sad', 'hate', 'poor', 'negative', 'horrible', 'wrong', 'fail', 
      'worst', 'disappointed', 'frustrating', 'useless', 'annoying', 'dislike', 'problem', 'issue', 
      'difficult', 'trouble', 'worry', 'worrying', 'concerned', 'concern', 'unfortunately', 'ugly', 
      'broken', 'error', 'failed', 'boring', 'pain', 'painful', 'angry', 'upset'
    ]);
    
    // Clean and tokenize text
    const cleanText = text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')  // Replace punctuation with spaces
      .replace(/\s+/g, ' ')      // Replace multiple spaces with a single space
      .trim();
    
    const words = cleanText.split(' ');
    
    // Count words and filter out stop words and short words
    const wordCount = {};
    const wordSentiment = {};
    
    words.forEach(word => {
      if (word.length > 2 && !stopWords.has(word)) {
        // Increment count
        wordCount[word] = (wordCount[word] || 0) + 1;
        
        // Calculate sentiment score for the word
        let sentimentScore = 0;
        if (positiveWords.has(word)) {
          sentimentScore = 0.8;
        } else if (negativeWords.has(word)) {
          sentimentScore = -0.8;
        }
        
        // Accumulate sentiment (we'll average later)
        if (!wordSentiment[word]) {
          wordSentiment[word] = { total: sentimentScore, count: 1 };
        } else {
          wordSentiment[word].total += sentimentScore;
          wordSentiment[word].count++;
        }
      }
    });
    
    // Convert to array for d3-cloud and take top 100 words
    const wordArray = Object.keys(wordCount)
      .map(word => ({
        text: word,
        count: wordCount[word],
        value: wordCount[word],
        sentimentScore: wordSentiment[word].total / wordSentiment[word].count
      }))
      .filter(d => d.count > 1)  // Only include words that appear more than once
      .sort((a, b) => b.count - a.count)
      .slice(0, 100);
    
    // Assign colors based on sentiment
    wordArray.forEach(word => {
      const score = word.sentimentScore;
      if (score <= -0.5) {
        word.color = '#FF5252'; // Very negative - red
      } else if (score < 0) {
        word.color = '#FF9E80'; // Negative - light red
      } else if (score === 0) {
        word.color = '#BDBDBD'; // Neutral - gray
      } else if (score < 0.5) {
        word.color = '#A5D6A7'; // Positive - light green
      } else {
        word.color = '#00C853'; // Very positive - green
      }
    });
    
    return wordArray;
  }
  
  // Interactive graph rendering
  function renderInteractiveGraph(history) {
    // Get the current selected graph type
    let graphType = 'pie';
    const selectedRadio = document.querySelector('input[name="graph-type"]:checked');
    if (selectedRadio) {
      graphType = selectedRadio.value;
    }
    
    // Render the graph with the selected type
    updateInteractiveGraph(graphType, history);
  }
  
  // Update interactive graph based on selected type
  function updateInteractiveGraph(type, history) {
    if (!history) {
      chrome.storage.local.get(['sentimentHistory'], function(result) {
        if (result.sentimentHistory) {
          // Filter based on selected option
          const visualizationFilter = document.getElementById('visualization-filter');
          const filterValue = visualizationFilter ? visualizationFilter.value : 'all';
          const filteredHistory = filterHistoryByTime(result.sentimentHistory, filterValue);
          createInteractiveGraph(type, filteredHistory);
        }
      });
    } else {
      createInteractiveGraph(type, history);
    }
  }
  
  // Create interactive graph
  function createInteractiveGraph(type, history) {
    const graphCanvas = document.getElementById('interactive-sentiment-graph');
    if (!graphCanvas) return;
    
    // Clear previous chart
    if (interactiveChart) {
      interactiveChart.destroy();
    }
    
    if (history.length === 0) {
      displayEmptyVisualization('interactive-graph');
      return;
    }
    
    // Count sentiment categories
    const sentimentCounts = {
      'overwhelmingly-negative': 0,
      'negative': 0,
      'neutral': 0,
      'positive': 0,
      'overwhelmingly-positive': 0
    };
    
    history.forEach(entry => {
      if (entry.category) {
        sentimentCounts[entry.category] = (sentimentCounts[entry.category] || 0) + 1;
      }
    });
    
    // Prepare data
    const labels = [
      'Very Negative', 
      'Negative', 
      'Neutral', 
      'Positive', 
      'Very Positive'
    ];
    
    const data = [
      sentimentCounts['overwhelmingly-negative'], 
      sentimentCounts['negative'], 
      sentimentCounts['neutral'], 
      sentimentCounts['positive'], 
      sentimentCounts['overwhelmingly-positive']
    ];
    
    const colors = [
      '#FF5252', // Very negative - red
      '#FF9E80', // Negative - light red
      '#BDBDBD', // Neutral - gray
      '#A5D6A7', // Positive - light green
      '#00C853'  // Very positive - green
    ];
    
    // Get colors based on theme
    const isDarkTheme = document.body.classList.contains('dark-theme');
    const gridColor = isDarkTheme ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    const textColor = isDarkTheme ? 'rgba(255, 255, 255, 0.7)' : 'rgba(0, 0, 0, 0.7)';
    
    // Determine chart type and options
    let chartConfig = {
      type: type,
      data: {
        labels: labels,
        datasets: [{
          label: 'Sentiment Distribution',
          data: data,
          backgroundColor: colors,
          borderColor: isDarkTheme ? 'rgba(0, 0, 0, 0.2)' : 'white',
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 1000,
          easing: 'easeOutQuart'
        },
        plugins: {
          legend: {
            display: true,
            position: 'bottom',
            labels: {
              color: textColor,
              padding: 15
            }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const value = context.raw;
                const total = data.reduce((a, b) => a + b, 0);
                const percentage = Math.round((value / total) * 100);
                return `${value} entries (${percentage}%)`;
              }
            }
          }
        }
      }
    };
    
    // Chart-specific options
    if (type === 'radar') {
      chartConfig.options.scales = {
        r: {
          angleLines: {
            color: gridColor
          },
          grid: {
            color: gridColor
          },
          pointLabels: {
            color: textColor
          },
          ticks: {
            color: textColor,
            backdropColor: 'transparent'
          }
        }
      };
    }
    
    // Create the chart
    interactiveChart = new Chart(graphCanvas, chartConfig);
  }
  
  // Initialize visualization filter
  const visualizationFilter = document.getElementById('visualization-filter');
  if (visualizationFilter) {
    visualizationFilter.addEventListener('change', function() {
      loadAndDisplayVisualizations();
    });
  }
  
  // Initialize graph type radio buttons
  const graphTypeRadios = document.querySelectorAll('input[name="graph-type"]');
  if (graphTypeRadios.length) {
    graphTypeRadios.forEach(radio => {
      radio.addEventListener('change', function() {
        if (this.checked) {
          updateInteractiveGraph(this.value);
        }
      });
    });
  }
  
  // Start by showing the analyze tab
  switchTab('analyze');
  
  // Add to history
  function addToHistory(text, result) {
    const maxHistoryItems = 100;
    
    // Create history item
    const historyItem = {
      text: text,
      result: result,
      timestamp: new Date().toISOString()
    };
    
    // Add to in-memory history
    sentimentHistory.unshift(historyItem);
    
    // Trim array if needed
    if (sentimentHistory.length > maxHistoryItems) {
      sentimentHistory = sentimentHistory.slice(0, maxHistoryItems);
    }
    
    // Save to storage
    chrome.storage.local.set({
      'sentimentHistory': sentimentHistory
    });
  }
  
  // Load and display history
  function loadAndDisplayHistory() {
    chrome.storage.local.get(['sentimentHistory'], function(result) {
      if (result.sentimentHistory && Array.isArray(result.sentimentHistory)) {
        sentimentHistory = result.sentimentHistory;
        displayHistory();
      } else {
        sentimentHistory = [];
        displayHistory();
      }
    });
  }
  
  // Display history items
  function displayHistory() {
    const historyList = document.getElementById('history-list');
    if (!historyList) return;
    
    // Get selected filter
    const filter = document.getElementById('history-filter');
    const filterValue = filter ? filter.value : 'all';
    
    // Clear current list
    historyList.innerHTML = '';
    
    // Filter history items
    let filteredHistory = sentimentHistory;
    
    if (filterValue === 'today') {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      filteredHistory = sentimentHistory.filter(item => {
        const itemDate = new Date(item.timestamp);
        return itemDate >= today;
      });
    } else if (filterValue === 'week') {
      const weekAgo = new Date();
      weekAgo.setDate(weekAgo.getDate() - 7);
      filteredHistory = sentimentHistory.filter(item => {
        const itemDate = new Date(item.timestamp);
        return itemDate >= weekAgo;
      });
    }
    
    // Check if we have any history
    if (filteredHistory.length === 0) {
      const emptyMessage = document.createElement('div');
      emptyMessage.className = 'empty-history-message';
      emptyMessage.textContent = 'No history items for the selected filter.';
      historyList.appendChild(emptyMessage);
      return;
    }
    
    // Append history items
    filteredHistory.forEach((item, index) => {
      const historyItem = createHistoryItem(item, index);
      historyList.appendChild(historyItem);
    });
  }
  
  // Create a history item element
  function createHistoryItem(item, index) {
    const historyItem = document.createElement('div');
    historyItem.className = 'history-item';
    
    const timestamp = new Date(item.timestamp);
    const formattedDate = timestamp.toLocaleDateString();
    const formattedTime = timestamp.toLocaleTimeString();
    
    historyItem.innerHTML = `
      <div class="history-item-header">
        <span class="history-item-date">${formattedDate} ${formattedTime}</span>
        <span class="history-item-sentiment ${item.result.category.id}">${item.result.category.emoji} ${item.result.category.label}</span>
      </div>
      <div class="history-item-text">${item.text.length > 100 ? item.text.substring(0, 100) + '...' : item.text}</div>
      <div class="history-item-summary">${item.result.summary}</div>
    `;
    
    // Expand/collapse on click
    historyItem.addEventListener('click', function() {
      this.classList.toggle('expanded');
    });
    
    return historyItem;
  }
  
  // Load and display visualizations
  function loadAndDisplayVisualizations() {
    chrome.storage.local.get(['sentimentHistory'], function(result) {
      if (result.sentimentHistory && Array.isArray(result.sentimentHistory)) {
        sentimentHistory = result.sentimentHistory;
        
        // Display all visualizations
        displayTimelineVisualization();
        displayWordCloudVisualization();
        displayInteractiveSentimentGraph();
      } else {
        sentimentHistory = [];
        showVisualizationEmptyState();
      }
    });
  }
  
  // Show empty state message for visualizations
  function showVisualizationEmptyState() {
    const containers = [
      document.querySelector('.timeline-container'),
      document.querySelector('.wordcloud-container'),
      document.querySelector('.interactive-graph-container')
    ];
    
    containers.forEach(container => {
      if (container) {
        container.innerHTML = '<div class="empty-visualization-message">No data available. Analyze some text to generate visualizations.</div>';
      }
    });
  }
  
  // Timeline Visualization
  function displayTimelineVisualization() {
    const timelineCanvas = document.getElementById('sentiment-timeline');
    if (!timelineCanvas) return;
    
    // Get selected filter
    const filter = document.getElementById('visualization-filter');
    const filterValue = filter ? filter.value : 'all';
    
    // Filter history items
    let filteredHistory = [...sentimentHistory].reverse(); // Reverse for chronological order
    
    if (filterValue === 'today') {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      filteredHistory = filteredHistory.filter(item => {
        const itemDate = new Date(item.timestamp);
        return itemDate >= today;
      });
    } else if (filterValue === 'week') {
      const weekAgo = new Date();
      weekAgo.setDate(weekAgo.getDate() - 7);
      filteredHistory = filteredHistory.filter(item => {
        const itemDate = new Date(item.timestamp);
        return itemDate >= weekAgo;
      });
    } else if (filterValue === 'month') {
      const monthAgo = new Date();
      monthAgo.setMonth(monthAgo.getMonth() - 1);
      filteredHistory = filteredHistory.filter(item => {
        const itemDate = new Date(item.timestamp);
        return itemDate >= monthAgo;
      });
    }
    
    // Check if we have any history
    if (filteredHistory.length === 0) {
      timelineCanvas.parentElement.innerHTML = '<div class="empty-visualization-message">No data available for the selected time range.</div>';
      return;
    }
    
    // Prepare data for timeline chart
    const timeLabels = filteredHistory.map(item => {
      const date = new Date(item.timestamp);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    });
    
    const sentimentData = filteredHistory.map(item => item.result.score);
    
    // Get context and destroy previous chart if it exists
    const ctx = timelineCanvas.getContext('2d');
    if (timelineChart) {
      timelineChart.destroy();
    }
    
    // Create new chart
    timelineChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: timeLabels,
        datasets: [{
          label: 'Sentiment Score',
          data: sentimentData,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          borderWidth: 2,
          tension: 0.4,
          pointBackgroundColor: function(context) {
            const score = context.raw;
            if (score < -0.6) return '#d32f2f';
            if (score < -0.2) return '#ff5722';
            if (score < 0.2) return '#ffeb3b';
            if (score < 0.6) return '#8bc34a';
            return '#4caf50';
          },
          pointRadius: 5,
          pointHoverRadius: 7
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: -1,
            max: 1,
            title: {
              display: true,
              text: 'Sentiment Score'
            },
            ticks: {
              callback: function(value) {
                if (value === -1) return 'Very Negative';
                if (value === -0.5) return 'Negative';
                if (value === 0) return 'Neutral';
                if (value === 0.5) return 'Positive';
                if (value === 1) return 'Very Positive';
                return '';
              }
            }
          },
          x: {
            title: {
              display: true,
              text: 'Time'
            }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              afterLabel: function(context) {
                const item = filteredHistory[context.dataIndex];
                return item.text.length > 50 ? item.text.substring(0, 50) + '...' : item.text;
              }
            }
          },
          legend: {
            display: false
          }
        }
      }
    });
  }
  
  // Word Cloud Visualization
  function displayWordCloudVisualization() {
    const wordCloudContainer = document.getElementById('sentiment-wordcloud');
    if (!wordCloudContainer) return;
    
    // Get selected filter
    const filter = document.getElementById('visualization-filter');
    const filterValue = filter ? filter.value : 'all';
    
    // Filter history items
    let filteredHistory = [...sentimentHistory];
    
    if (filterValue === 'today') {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      filteredHistory = filteredHistory.filter(item => {
        const itemDate = new Date(item.timestamp);
        return itemDate >= today;
      });
    } else if (filterValue === 'week') {
      const weekAgo = new Date();
      weekAgo.setDate(weekAgo.getDate() - 7);
      filteredHistory = filteredHistory.filter(item => {
        const itemDate = new Date(item.timestamp);
        return itemDate >= weekAgo;
      });
    } else if (filterValue === 'month') {
      const monthAgo = new Date();
      monthAgo.setMonth(monthAgo.getMonth() - 1);
      filteredHistory = filteredHistory.filter(item => {
        const itemDate = new Date(item.timestamp);
        return itemDate >= monthAgo;
      });
    }
    
    // Check if we have any history
    if (filteredHistory.length === 0) {
      wordCloudContainer.innerHTML = '<div class="empty-visualization-message">No data available for the selected time range.</div>';
      return;
    }
    
    // Clear previous word cloud
    wordCloudContainer.innerHTML = '';
    
    // Extract words and associated sentiments
    const wordData = extractWordsFromHistory(filteredHistory);
    
    // Check if we have enough words
    if (wordData.length === 0) {
      wordCloudContainer.innerHTML = '<div class="empty-visualization-message">Not enough text data for word cloud.</div>';
      return;
    }
    
    // Set up word cloud dimensions
    const width = wordCloudContainer.clientWidth;
    const height = 300;
    
    // Create SVG container
    const svg = d3.select(wordCloudContainer)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('class', 'word-cloud-svg');
    
    // Create word cloud layout
    const layout = d3.layout.cloud()
      .size([width, height])
      .words(wordData)
      .padding(5)
      .rotate(() => 0) // No rotation for better readability
      .fontSize(d => Math.sqrt(d.frequency) * 10) // Scale font size by sqrt of frequency
      .on('end', drawWordCloud);
    
    // Generate the word cloud
    layout.start();
    
    // Draw the word cloud
    function drawWordCloud(words) {
      svg.append('g')
        .attr('transform', `translate(${width / 2},${height / 2})`)
        .selectAll('text')
        .data(words)
        .enter()
        .append('text')
        .style('font-size', d => `${d.size}px`)
        .style('font-family', 'Impact')
        .style('fill', d => {
          // Color based on sentiment
          if (d.sentiment < -0.6) return '#d32f2f'; // very negative
          if (d.sentiment < -0.2) return '#ff5722'; // negative
          if (d.sentiment < 0.2) return '#ffeb3b'; // neutral
          if (d.sentiment < 0.6) return '#8bc34a'; // positive
          return '#4caf50'; // very positive
        })
        .attr('text-anchor', 'middle')
        .attr('transform', d => `translate(${d.x},${d.y})`)
        .text(d => d.text);
    }
  }
  
  // Extract words from history
  function extractWordsFromHistory(history) {
    // Common stop words to exclude
    const stopWords = new Set([
      'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
      'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
      'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot', 'could', 'couldn\'t',
      'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during',
      'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t',
      'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here',
      'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i',
      'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it',
      'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my',
      'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
      'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t',
      'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some',
      'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves',
      'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re',
      'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
      'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were',
      'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which',
      'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would',
      'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours',
      'yourself', 'yourselves'
    ]);
    
    // Word frequency and sentiment mapping
    const wordMap = {};
    
    history.forEach(item => {
      const text = item.text.toLowerCase();
      const sentiment = item.result.score;
      
      // Simple word extraction (improve this for production)
      const words = text.split(/\W+/).filter(word => 
        word.length > 2 && !stopWords.has(word)
      );
      
      words.forEach(word => {
        if (!wordMap[word]) {
          wordMap[word] = { count: 0, sentimentTotal: 0 };
        }
        
        wordMap[word].count += 1;
        wordMap[word].sentimentTotal += sentiment;
      });
    });
    
    // Convert to format needed for d3-cloud
    const wordData = Object.keys(wordMap)
      .filter(word => wordMap[word].count > 1) // Only include words that appear more than once
      .map(word => ({
        text: word,
        frequency: wordMap[word].count,
        sentiment: wordMap[word].sentimentTotal / wordMap[word].count,
        size: 0 // This will be set by the layout
      }))
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, 100); // Limit to top 100 words
    
    return wordData;
  }
  
  // Interactive Sentiment Graph
  function displayInteractiveSentimentGraph() {
    const graphCanvas = document.getElementById('interactive-sentiment-graph');
    if (!graphCanvas) return;
    
    // Get selected filter
    const filter = document.getElementById('visualization-filter');
    const filterValue = filter ? filter.value : 'all';
    
    // Filter history items
    let filteredHistory = [...sentimentHistory];
    
    if (filterValue === 'today') {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      filteredHistory = filteredHistory.filter(item => {
        const itemDate = new Date(item.timestamp);
        return itemDate >= today;
      });
    } else if (filterValue === 'week') {
      const weekAgo = new Date();
      weekAgo.setDate(weekAgo.getDate() - 7);
      filteredHistory = filteredHistory.filter(item => {
        const itemDate = new Date(item.timestamp);
        return itemDate >= weekAgo;
      });
    } else if (filterValue === 'month') {
      const monthAgo = new Date();
      monthAgo.setMonth(monthAgo.getMonth() - 1);
      filteredHistory = filteredHistory.filter(item => {
        const itemDate = new Date(item.timestamp);
        return itemDate >= monthAgo;
      });
    }
    
    // Check if we have any history
    if (filteredHistory.length === 0) {
      graphCanvas.parentElement.innerHTML = '<div class="empty-visualization-message">No data available for the selected time range.</div>';
      return;
    }
    
    // Get selected graph type
    const graphType = document.querySelector('input[name="graph-type"]:checked').value;
    
    // Prepare data for sentiment distribution
    const sentimentCounts = {
      'veryNegative': 0,
      'negative': 0,
      'neutral': 0,
      'positive': 0,
      'veryPositive': 0
    };
    
    filteredHistory.forEach(item => {
      const score = item.result.score;
      
      if (score < -0.6) sentimentCounts.veryNegative++;
      else if (score < -0.2) sentimentCounts.negative++;
      else if (score < 0.2) sentimentCounts.neutral++;
      else if (score < 0.6) sentimentCounts.positive++;
      else sentimentCounts.veryPositive++;
    });
    
    // Get context and destroy previous chart if it exists
    const ctx = graphCanvas.getContext('2d');
    if (interactiveChart) {
      interactiveChart.destroy();
    }
    
    // Define chart data
    const chartData = {
      labels: ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'],
      datasets: [{
        data: [
          sentimentCounts.veryNegative,
          sentimentCounts.negative,
          sentimentCounts.neutral,
          sentimentCounts.positive,
          sentimentCounts.veryPositive
        ],
        backgroundColor: [
          '#d32f2f',
          '#ff5722',
          '#ffeb3b',
          '#8bc34a',
          '#4caf50'
        ],
        borderColor: '#ffffff',
        borderWidth: 1
      }]
    };
    
    // Create chart based on selected type
    interactiveChart = new Chart(ctx, {
      type: graphType === 'pie' ? 'pie' : 
            graphType === 'radar' ? 'radar' : 'polarArea',
      data: chartData,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom'
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const value = context.raw;
                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                const percentage = Math.round((value / total) * 100);
                return `${context.label}: ${value} (${percentage}%)`;
              }
            }
          }
        }
      }
    });
  }
  
  // Listen for visualization filter changes
  const visualizationFilter = document.getElementById('visualization-filter');
  if (visualizationFilter) {
    visualizationFilter.addEventListener('change', function() {
      loadAndDisplayVisualizations();
    });
  }
  
  // Listen for graph type changes
  const graphTypeRadios = document.querySelectorAll('input[name="graph-type"]');
  if (graphTypeRadios.length > 0) {
    graphTypeRadios.forEach(radio => {
      radio.addEventListener('change', function() {
        displayInteractiveSentimentGraph();
      });
    });
  }
  
  // Listen for history filter changes
  const historyFilter = document.getElementById('history-filter');
  if (historyFilter) {
    historyFilter.addEventListener('change', function() {
      displayHistory();
    });
  }
});