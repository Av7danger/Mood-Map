// MoodMap extension popup.js - Complete Implementation

document.addEventListener('DOMContentLoaded', function() {
  console.log('MoodMap extension initialized');
  
  // Initialize the UI components
  initializeTabs();
  initializeDebugConsole();
  initializeTheme();
  testApiConnection();
  
  // Set up event listeners
  setupEventListeners();
  
  // Auto-analyze current page content when popup opens
  fetchAndAnalyzeActiveTabContent();
  
  // Utility function to get elements by ID
  function getElement(id) {
    return document.getElementById(id);
  }
  
  // Function to log to debug console
  function logDebug(message) {
    console.log(message);
    const debugOutput = getElement('debug-output');
    if (debugOutput) {
      const timestamp = new Date().toLocaleTimeString();
      debugOutput.value += `[${timestamp}] ${message}\n`;
      // Auto-scroll to bottom
      debugOutput.scrollTop = debugOutput.scrollHeight;
    }
  }
  
  // Function to fetch and analyze content from active tab
  function fetchAndAnalyzeActiveTabContent() {
    logDebug('Fetching content from active tab');
    
    // Get the active tab in the current window
    chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
      if (!tabs || tabs.length === 0) {
        logDebug('No active tab found');
        return;
      }
      
      const activeTab = tabs[0];
      logDebug(`Found active tab: ${activeTab.title}`);
      
      // Execute a content script to extract text from the page
      chrome.scripting.executeScript({
        target: { tabId: activeTab.id },
        function: extractPageContent
      }, function(results) {
        if (chrome.runtime.lastError) {
          logDebug(`Error executing script: ${chrome.runtime.lastError.message}`);
          return;
        }
        
        if (!results || results.length === 0 || !results[0].result) {
          logDebug('No content extracted from page');
          return;
        }
        
        const extractedContent = results[0].result;
        logDebug(`Extracted content: ${extractedContent.substring(0, 100)}...`);
        
        // Fill the textarea with the extracted content
        const textInput = getElement('text-input');
        if (textInput) {
          textInput.value = extractedContent;
          
          // Use analyzeWithSummary endpoint for consistent results with content script
          analyzeExtractedContent(extractedContent);
        }
      });
    });
  }
  
  // Function to analyze content using the same approach as the content script
  function analyzeExtractedContent(text) {
    logDebug('Analyzing extracted content');
    
    // Show loading state
    resetResultDisplay();
    const sentimentLabel = getElement('sentiment-label');
    if (sentimentLabel) {
      sentimentLabel.textContent = 'Analyzing...';
      sentimentLabel.className = 'sentiment-label loading';
    }
    
    // Send the text to the background script for analysis with summary
    // This matches the approach used in content-scripts/mood_analyzer.js
    chrome.runtime.sendMessage({ 
      type: 'analyzeWithSummary', 
      text: text,
      options: {
        preferAdvancedModel: true, // Always use advanced model (BART) just like content script
        forceGenerateSummary: true // Always generate summary regardless of length
      }
    }, function(response) {
      logDebug('Received analysis response');
      
      if (response && !response.error) {
        // Update results display
        updateResultDisplay(response);
        
        // Save to history
        saveAnalysisToHistory(text, response);
      } else {
        logDebug(`Analysis error: ${response ? response.error : 'No response'}`);
        
        // Fallback to simple model on error (same as content script)
        const result = processLocalSentiment(text);
        result.summary = generateSimpleSummary(text, result);
        if (response && response.error) {
          result.error = response.error;
        }
        
        // Update results display
        updateResultDisplay(result);
        
        // Show error message
        const summaryText = getElement('summary-text');
        if (summaryText && response && response.error) {
          summaryText.innerHTML = `<div class="api-error">Error contacting API: ${response.error}. Using simple offline model instead.</div>` + summaryText.innerHTML;
        }
        
        // Save to history
        saveAnalysisToHistory(text, result);
      }
    });
  }
  
  // Function to extract content from the page - runs in the context of the webpage
  function extractPageContent() {
    // Try to find the main content based on common selectors
    // Twitter/X tweet
    if (window.location.hostname.includes('twitter.com') || window.location.hostname.includes('x.com')) {
      const tweetText = document.querySelector('[data-testid="tweetText"]');
      if (tweetText) {
        return tweetText.textContent.trim();
      }
    }
    
    // LinkedIn post - updated with specific class selector
    if (window.location.hostname.includes('linkedin.com')) {
      // Try the specific class for LinkedIn posts
      const linkedInPosts = document.querySelectorAll('.WqxMNMcWrLmrFTIJOknbFgIXZDwsOuTaLuwBw');
      
      if (linkedInPosts && linkedInPosts.length > 0) {
        console.log('Found LinkedIn posts with specific class', linkedInPosts.length);
        
        // Combine text from all posts
        let combinedText = '';
        linkedInPosts.forEach(post => {
          const postText = post.textContent.trim();
          if (postText) {
            combinedText += postText + '\n\n';
          }
        });
        
        if (combinedText) {
          return combinedText.trim();
        }
      }
      
      // Fallback to other LinkedIn selectors if specific class not found
      const postContent = document.querySelector('.feed-shared-update-v2__description');
      if (postContent) {
        return postContent.textContent.trim();
      }
    }
    
    // Facebook post
    if (window.location.hostname.includes('facebook.com')) {
      const postContent = document.querySelector('.x1iorvi4');
      if (postContent) {
        return postContent.textContent.trim();
      }
    }
    
    // Reddit post
    if (window.location.hostname.includes('reddit.com')) {
      const postContent = document.querySelector('.RichTextJSON-root');
      if (postContent) {
        return postContent.textContent.trim();
      }
      
      // If that doesn't work, try the post title and body
      const postTitle = document.querySelector('h1');
      const postBody = document.querySelector('.RichTextJSON-root, .md');
      
      if (postTitle && postBody) {
        return `${postTitle.textContent}\n\n${postBody.textContent}`;
      }
    }
    
    // Generic fallbacks if no specific platform content found
    // Try to find main article content
    const articleContent = document.querySelector('article, [role="main"], .main-content');
    if (articleContent) {
      return articleContent.textContent.trim();
    }
    
    // Try to extract heading and first paragraphs as a fallback
    const heading = document.querySelector('h1');
    const paragraphs = Array.from(document.querySelectorAll('p')).slice(0, 5);
    if (heading && paragraphs.length > 0) {
      return heading.textContent + '\n\n' +
          paragraphs
              .filter(p => p.textContent.trim().length > 0)
              .map(p => p.textContent)
              .join('\n\n');
    }
    
    // Last resort - return page title and meta description
    const metaDescription = document.querySelector('meta[name="description"]');
    return document.title + (metaDescription ? `\n\n${metaDescription.getAttribute('content')}` : '');
  }
  
  // Initialize tab switching
  function initializeTabs() {
    logDebug('Initializing tabs');
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Helper function to switch tabs
    function switchTab(targetTabId) {
      logDebug(`Switching to tab: ${targetTabId}`);
      
      // Hide all tab contents and remove active class from all buttons
      tabContents.forEach(tabContent => {
        tabContent.classList.remove('active');
      });
      
      tabButtons.forEach(btn => {
        btn.classList.remove('active');
      });
      
      // Show target tab content and mark target button as active
      const targetTab = document.getElementById(`${targetTabId}-tab`);
      const targetButton = document.querySelector(`.tab-btn[data-tab="${targetTabId}"]`);
      
      if (targetTab) {
        targetTab.classList.add('active');
        
        // Initialize specific tab content if needed
        if (targetTabId === 'history') {
          loadHistoryData();
        } else if (targetTabId === 'visualize') {
          loadVisualizationData();
        } else if (targetTabId === 'settings') {
          loadSettings();
        }
      } else {
        console.error(`Tab with ID ${targetTabId}-tab not found`);
      }
      
      if (targetButton) {
        targetButton.classList.add('active');
      } else {
        console.error(`Button for tab ${targetTabId} not found`);
      }
    }
    
    // Add click event listeners to all tab buttons
    tabButtons.forEach(button => {
      button.addEventListener('click', () => {
        const tabId = button.getAttribute('data-tab');
        if (tabId) {
          switchTab(tabId);
        }
      });
    });
    
    // Start with analyze tab active
    switchTab('analyze');
  }
  
  // Setup all event listeners
  function setupEventListeners() {
    // Analyze button
    const analyzeButton = getElement('analyze-button');
    if (analyzeButton) {
      analyzeButton.addEventListener('click', performAnalysis);
    }
    
    // Theme selector
    const themeSelector = getElement('theme-selector');
    if (themeSelector) {
      themeSelector.addEventListener('change', function() {
        const selectedTheme = themeSelector.value;
        saveToStorage({ 'theme': selectedTheme });
        applyTheme(selectedTheme);
      });
    }
    
    // Auto-analyze toggle
    const autoAnalyzeToggle = getElement('auto-analyze-toggle');
    if (autoAnalyzeToggle) {
      autoAnalyzeToggle.addEventListener('change', function() {
        saveToStorage({ 'autoAnalyze': autoAnalyzeToggle.checked });
      });
    }
    
    // API URL update button
    const updateApiUrlBtn = getElement('update-api-url-btn');
    if (updateApiUrlBtn) {
      updateApiUrlBtn.addEventListener('click', updateApiUrl);
    }
    
    // Test API button
    const testApiBtn = getElement('test-api-btn');
    if (testApiBtn) {
      testApiBtn.addEventListener('click', testApiConnection);
    }
    
    // Clear history button
    const clearHistoryBtn = getElement('clear-history-btn');
    if (clearHistoryBtn) {
      clearHistoryBtn.addEventListener('click', clearHistory);
    }
    
    // History filter
    const historyFilter = getElement('history-filter');
    if (historyFilter) {
      historyFilter.addEventListener('change', function() {
        loadHistoryData(historyFilter.value);
      });
    }
    
    // Visualization filter
    const visualizationFilter = getElement('visualization-filter');
    if (visualizationFilter) {
      visualizationFilter.addEventListener('change', function() {
        loadVisualizationData(visualizationFilter.value);
      });
    }
    
    // Graph type radio buttons
    const graphTypeRadios = document.querySelectorAll('input[name="graph-type"]');
    graphTypeRadios.forEach(radio => {
      radio.addEventListener('change', function() {
        if (this.checked) {
          updateDistributionChart(this.value);
        }
      });
    });
  }
  
  // Initialize debug console
  function initializeDebugConsole() {
    const toggleDebugBtn = getElement('toggle-debug');
    const debugContainer = getElement('debug-container');
    const clearDebugBtn = getElement('clear-debug-btn');
    const debugOutput = getElement('debug-output');
    
    if (toggleDebugBtn && debugContainer) {
      toggleDebugBtn.addEventListener('click', function() {
        debugContainer.style.display = debugContainer.style.display === 'none' ? 'block' : 'none';
      });
    }
    
    if (clearDebugBtn && debugOutput) {
      clearDebugBtn.addEventListener('click', function() {
        debugOutput.value = '';
      });
    }
    
    logDebug('Debug console initialized');
  }
  
  // Initialize theme
  function initializeTheme() {
    loadFromStorage({ 'theme': 'light' }, function(data) {
      const themeSelector = getElement('theme-selector');
      if (themeSelector) {
        themeSelector.value = data.theme;
      }
      applyTheme(data.theme);
    });
  }
  
  // Apply theme to the UI
  function applyTheme(theme) {
    logDebug(`Applying theme: ${theme}`);
    const body = document.body;
    body.classList.remove('light-theme', 'dark-theme');
    
    if (theme === 'dark') {
      body.classList.add('dark-theme');
    } else if (theme === 'system') {
      // Check system preference
      if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        body.classList.add('dark-theme');
      } else {
        body.classList.add('light-theme');
      }
    } else {
      body.classList.add('light-theme');
    }
  }
  
  // Load history data and populate history tab
  function loadHistoryData(filter = 'all') {
    logDebug(`Loading history data with filter: ${filter}`);
    const historyList = getElement('history-list');
    
    if (!historyList) {
      console.error('History list element not found');
      return;
    }
    
    // Clear previous history items
    historyList.innerHTML = '';
    
    // Load history from storage
    loadFromStorage({ 'analysisHistory': [] }, function(data) {
      let history = data.analysisHistory || [];
      
      // Apply filter
      if (filter === 'today') {
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        history = history.filter(item => new Date(item.timestamp) >= today);
      } else if (filter === 'week') {
        const weekAgo = new Date();
        weekAgo.setDate(weekAgo.getDate() - 7);
        history = history.filter(item => new Date(item.timestamp) >= weekAgo);
      }
      
      if (history.length === 0) {
        historyList.innerHTML = '<div class="empty-history-message">No history items yet. Analyze some text to see it here.</div>';
        return;
      }
      
      // Sort history by timestamp (newest first)
      history.sort((a, b) => b.timestamp - a.timestamp);
      
      // Add history items to the list
      history.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        
        const timestamp = new Date(item.timestamp);
        const formattedDate = timestamp.toLocaleDateString();
        const formattedTime = timestamp.toLocaleTimeString();
        
        // Determine sentiment class
        let sentimentClass = 'neutral';
        if (item.category === 0 || item.label === 'negative') {
          sentimentClass = 'negative';
        } else if (item.category === 2 || item.label === 'positive') {
          sentimentClass = 'positive';
        }
        
        // Truncate text if it's too long
        const truncatedText = item.text.length > 100 ? item.text.substring(0, 97) + '...' : item.text;
        
        historyItem.innerHTML = `
          <div class="history-item-header">
            <span class="history-timestamp">${formattedDate} ${formattedTime}</span>
            <span class="history-sentiment ${sentimentClass}">${item.label || 'Unknown'}</span>
          </div>
          <div class="history-item-text">${truncatedText}</div>
          <div class="history-item-footer">
            <span class="history-model">Model: ${item.model_used || 'Unknown'}</span>
            <button class="history-analyze-again" data-text="${item.text.replace(/"/g, '&quot;')}">Analyze Again</button>
          </div>
        `;
        
        historyList.appendChild(historyItem);
      });
      
      // Add event listeners to "Analyze Again" buttons
      const analyzeAgainButtons = document.querySelectorAll('.history-analyze-again');
      analyzeAgainButtons.forEach(button => {
        button.addEventListener('click', function() {
          const text = this.getAttribute('data-text');
          if (text) {
            // Switch to analyze tab
            const analyzeButton = document.querySelector('.tab-btn[data-tab="analyze"]');
            if (analyzeButton) {
              analyzeButton.click();
            }
            
            // Set the text and perform analysis
            const textInput = getElement('text-input');
            if (textInput) {
              textInput.value = text;
              performAnalysis();
            }
          }
        });
      });
    });
  }
  
  // Clear history
  function clearHistory() {
    if (confirm('Are you sure you want to clear all history? This cannot be undone.')) {
      saveToStorage({ 'analysisHistory': [] }, function(success) {
        if (success) {
          loadHistoryData(); // Refresh the history tab
          logDebug('History cleared');
        }
      });
    }
  }
  
  // Load and display visualization data
  function loadVisualizationData(filter = 'all') {
    logDebug(`Loading visualization data with filter: ${filter}`);
    
    // Load history from storage
    loadFromStorage({ 'analysisHistory': [] }, function(data) {
      let history = data.analysisHistory || [];
      
      // Apply filter
      if (filter === 'today') {
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        history = history.filter(item => new Date(item.timestamp) >= today);
      } else if (filter === 'week') {
        const weekAgo = new Date();
        weekAgo.setDate(weekAgo.getDate() - 7);
        history = history.filter(item => new Date(item.timestamp) >= weekAgo);
      } else if (filter === 'month') {
        const monthAgo = new Date();
        monthAgo.setMonth(monthAgo.getMonth() - 1);
        history = history.filter(item => new Date(item.timestamp) >= monthAgo);
      }
      
      if (history.length === 0) {
        const visualizationPanels = document.querySelectorAll('.visualization-panel');
        visualizationPanels.forEach(panel => {
          panel.innerHTML = '<div class="empty-visualization-message">No history data available for visualization. Analyze some text first.</div>';
        });
        return;
      }
      
      // Create timeline chart
      createTimelineChart(history);
      
      
      // Create distribution chart - get selected type
      const selectedChartType = document.querySelector('input[name="graph-type"]:checked')?.value || 'pie';
      createDistributionChart(history, selectedChartType);
    });
  }
  
  // Create timeline chart showing sentiment over time
  function createTimelineChart(history) {
    const container = getElement('sentiment-timeline');
    if (!container || history.length < 2) {
      if (container) {
        const panel = container.closest('.visualization-panel');
        if (panel) {
          panel.innerHTML = '<div class="empty-visualization-message">Not enough data for timeline visualization. Analyze more text to see a timeline.</div>';
        }
      }
      return;
    }
    
    // Sort history by timestamp (oldest first)
    const sortedHistory = [...history].sort((a, b) => a.timestamp - b.timestamp);
    
    // Prepare data for chart
    const labels = [];
    const sentimentScores = [];
    
    sortedHistory.forEach(item => {
      // Format date for label
      const date = new Date(item.timestamp);
      labels.push(date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}));
      
      // Get sentiment score
      let score = 0;
      if (item.score !== undefined) {
        score = item.score;
      } else if (item.category === 0) {
        score = -0.7; // Negative
      } else if (item.category === 2) {
        score = 0.7; // Positive
      }
      
      sentimentScores.push(score);
    });
    
    // Create chart
    try {
      // Check if there's an existing chart and destroy it
      if (window.timelineChart) {
        window.timelineChart.destroy();
      }
      
      // Create new chart
      window.timelineChart = new Chart(container, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [{
            label: 'Sentiment Score',
            data: sentimentScores,
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderWidth: 2,
            tension: 0.4,
            fill: true
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: false,
              min: -1,
              max: 1,
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
            }
          },
          plugins: {
            legend: {
              display: true,
              position: 'top'
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  const score = context.raw;
                  let sentiment = 'Neutral';
                  if (score <= -0.7) sentiment = 'Very Negative';
                  else if (score < 0) sentiment = 'Negative';
                  else if (score >= 0.7) sentiment = 'Very Positive';
                  else if (score > 0) sentiment = 'Positive';
                  return `Sentiment: ${sentiment} (${score.toFixed(2)})`;
                }
              }
            }
          }
        }
      });
      
      logDebug('Timeline chart created');
    } catch (error) {
      console.error('Error creating timeline chart:', error);
      logDebug(`Error creating timeline chart: ${error.message}`);
    }
  }
  
  // Create distribution chart (pie, radar, or polar area)
  function createDistributionChart(history, chartType = 'pie') {
    const container = getElement('interactive-sentiment-graph');
    if (!container || history.length === 0) {
      if (container) {
        const panel = container.closest('.visualization-panel');
        if (panel) {
          panel.innerHTML = '<div class="empty-visualization-message">No data available for sentiment distribution visualization.</div>';
        }
      }
      return;
    }
    
    // Count sentiment categories
    let negativeCount = 0;
    let neutralCount = 0;
    let positiveCount = 0;
    
    history.forEach(item => {
      if (item.category === 0 || item.label === 'negative') {
        negativeCount++;
      } else if (item.category === 2 || item.label === 'positive') {
        positiveCount++;
      } else {
        neutralCount++;
      }
    });
    
    // Create chart
    try {
      // Check if there's an existing chart and destroy it
      if (window.distributionChart) {
        window.distributionChart.destroy();
      }
      
      // Determine chart type
      const type = chartType === 'radar' ? 'radar' : (chartType === 'polar' ? 'polarArea' : 'doughnut');
      
      // Create new chart
      window.distributionChart = new Chart(container, {
        type: type,
        data: {
          labels: ['Negative', 'Neutral', 'Positive'],
          datasets: [{
            data: [negativeCount, neutralCount, positiveCount],
            backgroundColor: [
              'rgba(255, 99, 132, 0.7)',
              'rgba(255, 205, 86, 0.7)',
              'rgba(75, 192, 192, 0.7)'
            ],
            borderColor: [
              'rgb(255, 99, 132)',
              'rgb(255, 205, 86)',
              'rgb(75, 192, 192)'
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
                  const value = context.raw;
                  const total = negativeCount + neutralCount + positiveCount;
                  const percentage = ((value / total) * 100).toFixed(1);
                  return `${context.label}: ${value} (${percentage}%)`;
                }
              }
            }
          }
        }
      });
      
      logDebug(`Distribution chart (${chartType}) created`);
    } catch (error) {
      console.error(`Error creating ${chartType} chart:`, error);
      logDebug(`Error creating ${chartType} chart: ${error.message}`);
    }
  }
  
  // Update distribution chart type
  function updateDistributionChart(chartType) {
    logDebug(`Updating distribution chart type to: ${chartType}`);
    loadFromStorage({ 'analysisHistory': [] }, function(data) {
      const history = data.analysisHistory || [];
      createDistributionChart(history, chartType);
    });
  }
    
  // Load settings
  function loadSettings() {
    logDebug('Loading settings');
    
    // Load settings from storage
    loadFromStorage({
      'theme': 'light',
      'autoAnalyze': false,
      'apiUrl': 'http://localhost:5000'
    }, function(data) {
      // Apply theme
      const themeSelector = getElement('theme-selector');
      if (themeSelector) {
        themeSelector.value = data.theme;
      }
      
      // Set auto-analyze toggle
      const autoAnalyzeToggle = getElement('auto-analyze-toggle');
      if (autoAnalyzeToggle) {
        autoAnalyzeToggle.checked = data.autoAnalyze;
      }
      
      // Set API URL display and inputs
      const apiUrlDisplay = getElement('api-url-display');
      const apiProtocolSelect = getElement('api-protocol-select');
      const apiUrlInput = getElement('api-url-input');
      
      if (apiUrlDisplay) {
        apiUrlDisplay.textContent = data.apiUrl;
      }
      
      if (apiProtocolSelect && apiUrlInput && data.apiUrl) {
        try {
          const url = new URL(data.apiUrl);
          apiProtocolSelect.value = url.protocol.replace(':', '');
          apiUrlInput.value = url.host + url.pathname.replace(/\/$/, '');
        } catch (e) {
          console.error('Error parsing API URL:', e);
          apiProtocolSelect.value = 'http';
          apiUrlInput.value = 'localhost:5000';
        }
      }
    });
  }
  
  // Update API URL
  function updateApiUrl() {
    const apiProtocolSelect = getElement('api-protocol-select');
    const apiUrlInput = getElement('api-url-input');
    const apiUrlDisplay = getElement('api-url-display');
    const apiMessage = getElement('api-message');
    
    if (!apiProtocolSelect || !apiUrlInput) {
      return;
    }
    
    const protocol = apiProtocolSelect.value;
    const host = apiUrlInput.value.trim();
    
    if (!host) {
      if (apiMessage) {
        apiMessage.textContent = 'Please enter a valid API host';
        apiMessage.className = 'api-message error';
      }
      return;
    }
    
    // Construct the new API URL
    const apiUrl = `${protocol}://${host.replace(/^\/*/, '').replace(/\/*$/, '')}`;
    
    // Save to storage
    saveToStorage({ 'apiUrl': apiUrl }, function(success) {
      if (success) {
        if (apiUrlDisplay) {
          apiUrlDisplay.textContent = apiUrl;
        }
        
        if (apiMessage) {
          apiMessage.textContent = `API URL updated to: ${apiUrl}`;
          apiMessage.className = 'api-message success';
        }
        
        // Test the connection with the new URL
        testApiConnection();
      }
    });
    
    logDebug(`API URL updated to: ${apiUrl}`);
  }
  
  // Test API connection
  function testApiConnection() {
    const apiStatus = getElement('api-status');
    const apiMessage = getElement('api-message');
    
    if (apiStatus) {
      apiStatus.textContent = 'Checking...';
      apiStatus.className = 'api-status checking';
    }
    
    if (apiMessage) {
      apiMessage.textContent = 'Testing connection to the API...';
      apiMessage.className = 'api-message';
    }
    
    logDebug('Testing API connection...');
    
    // Get API URL from storage
    loadFromStorage({ 'apiUrl': 'http://localhost:5000' }, function(data) {
      const apiUrl = data.apiUrl;
      
      // Make a request to the health endpoint
      fetch(`${apiUrl}/health`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        logDebug('API health check response: ' + JSON.stringify(data));
        
        if (apiStatus) {
          apiStatus.textContent = 'Connected';
          apiStatus.className = 'api-status connected';
        }
        
        if (apiMessage) {
          apiMessage.textContent = `Successfully connected to API. ${data.models ? `Available models: ${Object.keys(data.models).join(', ')}` : ''}`;
          apiMessage.className = 'api-message success';
        }
        
        // Update model selector if we have model status
        if (data.models) {
          updateModelSelector(data.models);
        }
      })
      .catch(error => {
        console.error('API connection error:', error);
        logDebug(`API connection error: ${error.message}`);
        
        if (apiStatus) {
          apiStatus.textContent = 'Disconnected';
          apiStatus.className = 'api-status disconnected';
        }
        
        if (apiMessage) {
          apiMessage.textContent = `Could not connect to API: ${error.message}. You can still use the simple offline model.`;
          apiMessage.className = 'api-message error';
        }
        
        // Update model selector to only show offline model
        updateModelSelector({ simple: true });
      });
    });
  }
  
  // Update model selector based on available models
  function updateModelSelector(modelsStatus) {
    const modelSelector = getElement('model-selector');
    if (!modelSelector) {
      console.error('Model selector element not found');
      return;
    }
    
    // Save current selection
    const currentSelection = modelSelector.value;
    
    // Clear existing options
    modelSelector.innerHTML = '';
    
    // Add available models
    if (modelsStatus.ensemble) {
      addModelOption(modelSelector, 'ensemble', 'Ensemble Model (Best Overall)');
    }
    
    if (modelsStatus.attention) {
      addModelOption(modelSelector, 'attention', 'Attention Model (Detail Focused)');
    }
    
    if (modelsStatus.neutral) {
      addModelOption(modelSelector, 'neutral', 'Neutral Model (Balanced)');
    }
    
    if (modelsStatus.advanced) {
      addModelOption(modelSelector, 'advanced', 'Advanced Model (Most Accurate)');
    }
    
    // Always add simple option for offline use
    addModelOption(modelSelector, 'simple', 'Simple Model (Offline Mode - Always Available)');
    
    // Try to restore previous selection
    let selected = false;
    for (let i = 0; i < modelSelector.options.length; i++) {
      if (modelSelector.options[i].value === currentSelection) {
        modelSelector.selectedIndex = i;
        selected = true;
        break;
      }
    }
    
    // Default to best available model if previous selection not available
    if (!selected) {
      if (modelsStatus.ensemble) {
        modelSelector.value = 'ensemble';
      } else if (modelsStatus.attention) {
        modelSelector.value = 'attention';
      } else if (modelsStatus.advanced) {
        modelSelector.value = 'advanced';
      } else if (modelsStatus.neutral) {
        modelSelector.value = 'neutral';
      } else {
        modelSelector.value = 'simple';
      }
    }
    
    // Save the selected model
    saveToStorage({ selectedModel: modelSelector.value });
    logDebug(`Model set to: ${modelSelector.value}`);
  }
  
  // Helper to add option to model selector
  function addModelOption(selector, value, text) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = text;
    selector.appendChild(option);
  }
  
  // Perform sentiment analysis
  function performAnalysis() {
    const textInput = getElement('text-input');
    if (!textInput || !textInput.value.trim()) {
      alert('Please enter some text to analyze');
      return;
    }
    
    const text = textInput.value.trim();
    const modelSelector = getElement('model-selector');
    const modelType = modelSelector ? modelSelector.value : 'simple';
    
    logDebug(`Analyzing text with model: ${modelType}`);
    
    // Clear previous results
    resetResultDisplay();
    
    // Show loading state
    const sentimentLabel = getElement('sentiment-label');
    if (sentimentLabel) {
      sentimentLabel.textContent = 'Analyzing...';
      sentimentLabel.className = 'sentiment-label loading';
    }
    
    // If using simple model, process locally
    if (modelType === 'simple') {
      logDebug('Using simple offline model');
      // Local processing without API
      const result = processLocalSentiment(text);
      
      // Generate a simple summary
      result.summary = generateSimpleSummary(text, result);
      
      // Update results display
      updateResultDisplay(result);
      
      // Save to history
      saveAnalysisToHistory(text, result);
    } else {
      // Get API URL from storage
      loadFromStorage({ 'apiUrl': 'http://localhost:5000' }, function(data) {
        const apiUrl = data.apiUrl;
        
        logDebug(`Sending request to API at ${apiUrl}`);
        
        // Make API request
        fetch(`${apiUrl}/analyze`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify({
            text: text,
            model: modelType,
            include_summary: true
          })
        })
        .then(response => {
          if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
          }
          return response.json();
        })
        .then(result => {
          logDebug('API response received');
          
          // Update results display
          updateResultDisplay(result);
          
          // Save to history
          saveAnalysisToHistory(text, result);
        })
        .catch(error => {
          console.error('API error:', error);
          logDebug(`API error: ${error.message}`);
          
          // Fallback to simple model on error
          const result = processLocalSentiment(text);
          result.summary = generateSimpleSummary(text, result);
          result.error = error.message;
          
          // Update results display
          updateResultDisplay(result);
          
          // Show error message
          const summaryText = getElement('summary-text');
          if (summaryText) {
            summaryText.innerHTML = `<div class="api-error">Error contacting API: ${error.message}. Using simple offline model instead.</div>` + summaryText.innerHTML;
          }
          
          // Save to history
          saveAnalysisToHistory(text, result);
        });
      });
    }
  }
  
  // Reset result display
  function resetResultDisplay() {
    const sentimentLabel = getElement('sentiment-label');
    const summaryText = getElement('summary-text');
    const sentimentScore = getElement('sentiment-score');
    
    if (sentimentScore) {
      sentimentScore.textContent = '-';
    }
    
    if (sentimentLabel) {
      sentimentLabel.textContent = 'Analyzing...';
      sentimentLabel.className = 'sentiment-label';
    }
    
    if (summaryText) {
      summaryText.textContent = 'Generating summary...';
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
    logDebug('Updating display with result');
    
    const sentimentLabel = getElement('sentiment-label');
    const summaryText = getElement('summary-text');
    const sentimentScore = getElement('sentiment-score');
    const sentimentEmoji = getElement('sentiment-emoji');
    const sentimentCircle = document.querySelector('.sentiment-circle');
    
    // Normalize result format
    const normalizedResult = {
      score: result.score !== undefined ? result.score : 0,
      category: result.category !== undefined ? result.category : 1,
      label: result.label || 'neutral',
      confidence: result.confidence || 0.5,
      summary: result.summary || null,
      model_used: result.model_used || 'unknown',
      summarization_method: result.summarization_method
    };
    
    // Update sentiment circle
    if (sentimentCircle) {
      // Remove previous classes
      sentimentCircle.classList.remove('negative', 'neutral', 'positive');
      
      // Add appropriate class
      if (normalizedResult.category === 0 || normalizedResult.label === 'negative') {
        sentimentCircle.classList.add('negative');
      } else if (normalizedResult.category === 2 || normalizedResult.label === 'positive') {
        sentimentCircle.classList.add('positive');
      } else {
        sentimentCircle.classList.add('neutral');
      }
    }
    
    // Update sentiment score (hidden but still updated for reference)
    if (sentimentScore) {
      sentimentScore.textContent = normalizedResult.score.toFixed(2);
    }
    
    // Update sentiment emoji based on score/category
    if (sentimentEmoji) {
      let emoji = '😐'; // Default neutral emoji
      
      if (normalizedResult.category === 0 || normalizedResult.label === 'negative') {
        // Select negative emoji based on score intensity
        const score = normalizedResult.score;
        if (score < -0.7) {
          emoji = '😡'; // Very negative
        } else if (score < -0.4) {
          emoji = '😟'; // Moderately negative
        } else {
          emoji = '🙁'; // Slightly negative
        }
      } else if (normalizedResult.category === 2 || normalizedResult.label === 'positive') {
        // Select positive emoji based on score intensity
        const score = normalizedResult.score;
        if (score > 0.7) {
          emoji = '😄'; // Very positive
        } else if (score > 0.4) {
          emoji = '😊'; // Moderately positive
        } else {
          emoji = '🙂'; // Slightly positive
        }
      }
      
      sentimentEmoji.textContent = emoji;
    }
    
    // Update sentiment label
    if (sentimentLabel) {
      // Determine sentiment text
      let sentiment = 'Neutral';
      let labelClass = 'neutral';
      
      if (normalizedResult.category === 0 || normalizedResult.label === 'negative') {
        sentiment = 'Negative';
        labelClass = 'negative';
      } else if (normalizedResult.category === 2 || normalizedResult.label === 'positive') {
        sentiment = 'Positive';
        labelClass = 'positive';
      }
      
      sentimentLabel.textContent = sentiment;
      sentimentLabel.className = `sentiment-label ${labelClass}`;
    }
    
    // Update sentiment bars
    updateSentimentBars(normalizedResult);
    
    // Update summary text
    if (summaryText) {
      if (normalizedResult.summary) {
        summaryText.textContent = normalizedResult.summary;
      } else {
        summaryText.textContent = 'No summary available for this text.';
      }
      
      // Format the model information to show both sentiment and summarization clearly
      let modelInfo = normalizedResult.model_used;
      
      // Check if we have both sentiment model and BART summarizer
      if (normalizedResult.summarization_method === 'bart' && 
          !normalizedResult.model_used.includes('BART') && 
          !normalizedResult.model_used.includes('bart')) {
        modelInfo = `${normalizedResult.model_used} + BART summarizer`;
      }
      
      // Add model information
      const modelInfoElement = document.createElement('div');
      modelInfoElement.className = 'model-info';
      modelInfoElement.textContent = `Analyzed using: ${modelInfo}`;
      summaryText.appendChild(modelInfoElement);
      
      // Add error information if applicable
      if (result.error) {
        const errorInfo = document.createElement('div');
        errorInfo.className = 'error-info';
        errorInfo.textContent = `Note: ${result.error}`;
        summaryText.appendChild(errorInfo);
      }
      
      // If model fallback happened, add explanation
      if (result.fallback_to && result.fallback_to !== "none") {
        const fallbackInfo = document.createElement('div');
        fallbackInfo.className = 'fallback-info';
        fallbackInfo.textContent = `Note: Advanced model was not available, used ${result.fallback_to} model instead.`;
        summaryText.appendChild(fallbackInfo);
      }
    }
  }
  
  // Update sentiment bars
  function updateSentimentBars(result) {
    // Convert to bar percentages
    let negativeValue = 0;
    let neutralValue = 0;
    let positiveValue = 0;
    
    // Map category to bar values
    if (result.category === 0 || result.label === 'negative') {
      negativeValue = 70;
      neutralValue = 20;
      positiveValue = 10;
    } else if (result.category === 2 || result.label === 'positive') {
      negativeValue = 10;
      neutralValue = 20;
      positiveValue = 70;
    } else {
      negativeValue = 20;
      neutralValue = 60;
      positiveValue = 20;
    }
    
    // Fine-tune with confidence if available
    if (result.confidence) {
      const confidence = result.confidence;
      
      if (result.category === 0 || result.label === 'negative') {
        negativeValue = Math.min(90, negativeValue + (confidence * 20));
        neutralValue = Math.max(5, neutralValue - (confidence * 10));
        positiveValue = Math.max(5, positiveValue - (confidence * 10));
      } else if (result.category === 2 || result.label === 'positive') {
        positiveValue = Math.min(90, positiveValue + (confidence * 20));
        neutralValue = Math.max(5, neutralValue - (confidence * 10));
        negativeValue = Math.max(5, negativeValue - (confidence * 10));
      } else {
        neutralValue = Math.min(80, neutralValue + (confidence * 20));
        negativeValue = Math.max(10, negativeValue - (confidence * 10));
        positiveValue = Math.max(10, positiveValue - (confidence * 10));
      }
    }
    
    // Score-based adjustments
    if (result.score !== undefined) {
      const score = result.score;
      
      if (score < 0) {
        // Negative score - increase negative bar
        const adjustment = Math.abs(score) * 30;
        negativeValue += adjustment;
        neutralValue = Math.max(5, neutralValue - adjustment / 2);
        positiveValue = Math.max(5, positiveValue - adjustment / 2);
      } else if (score > 0) {
        // Positive score - increase positive bar
        const adjustment = score * 30;
        positiveValue += adjustment;
        neutralValue = Math.max(5, neutralValue - adjustment / 2);
        negativeValue = Math.max(5, negativeValue - adjustment / 2);
      }
      
      // Normalize percentages to add up to 100%
      const total = negativeValue + neutralValue + positiveValue;
      negativeValue = Math.round((negativeValue / total) * 100);
      neutralValue = Math.round((neutralValue / total) * 100);
      positiveValue = Math.round((positiveValue / total) * 100);
      
      // Ensure they add up to exactly 100%
      const diff = 100 - (negativeValue + neutralValue + positiveValue);
      neutralValue += diff;
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
    
    if (negativeValue_el) negativeValue_el.textContent = `${negativeValue}%`;
    if (neutralValue_el) neutralValue_el.textContent = `${neutralValue}%`;
    if (positiveValue_el) positiveValue_el.textContent = `${positiveValue}%`;
  }
  
  // Improved offline sentiment analysis with contextual awareness
  function processLocalSentiment(text) {
    if (!text || text.trim().length === 0) {
      return {
        category: "neutral",
        score: 0.5,
        confidence: 0,
        summary: "No text to analyze.",
        details: {
          sentiment_trend: "consistent",
          sentiment_shift: false,
          has_questions: false,
          section_sentiments: []
        }
      };
    }
    
    // Simple positive and negative word lists for basic sentiment detection
    const positiveWords = [
      "good", "great", "excellent", "awesome", "happy", "love", "like", "best", 
      "positive", "wonderful", "enjoy", "nice", "fantastic", "amazing", "perfect",
      "excited", "glad", "pleased", "grateful", "satisfied", "enthusiastic"
    ];
    
    const negativeWords = [
      "bad", "terrible", "awful", "hate", "dislike", "worst", "negative", "poor", 
      "horrible", "unfortunate", "sad", "angry", "upset", "disappoint", "frustrat",
      "annoyed", "worried", "sorry", "unhappy", "regret", "depressed", "concerned"
    ];
    
    // Split text into sections for trend analysis
    const sentences = text.match(/[^\.!\?]+[\.!\?]+/g) || [text];
    const sectionCount = Math.min(3, Math.ceil(sentences.length / 3));
    const sectionsLength = Math.ceil(sentences.length / sectionCount);
    
    const sections = [];
    for (let i = 0; i < sectionCount; i++) {
      const start = i * sectionsLength;
      const end = Math.min(start + sectionsLength, sentences.length);
      const sectionText = sentences.slice(start, end).join(" ");
      sections.push(sectionText);
    }
    
    // Analyze each section
    const sectionSentiments = sections.map(section => {
      return analyzeTextSentiment(section, positiveWords, negativeWords);
    });
    
    // Detect sentiment trend
    let sentimentTrend = "consistent";
    if (sectionSentiments.length > 1) {
      const firstScore = sectionSentiments[0].score;
      const lastScore = sectionSentiments[sectionSentiments.length - 1].score;
      
      if (lastScore - firstScore > 0.15) {
        sentimentTrend = "improving";
      } else if (firstScore - lastScore > 0.15) {
        sentimentTrend = "worsening";
      }
    }
    
    // Detect sentiment shifts
    let sentimentShift = false;
    for (let i = 1; i < sectionSentiments.length; i++) {
      const prevScore = sectionSentiments[i-1].score;
      const currScore = sectionSentiments[i].score;
      if (Math.abs(currScore - prevScore) > 0.25) {
        sentimentShift = true;
        break;
      }
    }
    
    // Check for questions that might indicate different context
    const hasQuestions = text.includes("?");
    
    // Calculate overall sentiment
    const overallSentiment = analyzeTextSentiment(text, positiveWords, negativeWords);
    
    // Generate confidence based on sentiment strength and text length
    const confidence = calculateConfidence(overallSentiment.score, text.length, sentimentShift);
    
    // Create the result object with enhanced contextual details
    const result = {
      category: overallSentiment.category,
      score: overallSentiment.score,
      confidence: confidence,
      details: {
        sentiment_trend: sentimentTrend,
        sentiment_shift: sentimentShift,
        has_questions: hasQuestions,
        section_sentiments: sectionSentiments.map(s => s.category)
      }
    };
    
    // Generate context-aware summary
    result.summary = generateSimpleSummary(text, result);
    
    return result;
  }
  
  // Helper function to analyze sentiment of a text segment
  function analyzeTextSentiment(text, positiveWords, negativeWords) {
    text = text.toLowerCase();
    let positiveCount = 0;
    let negativeCount = 0;
    
    // Count positive words with consideration for negations
    positiveWords.forEach(word => {
      const regex = new RegExp("\\b" + word + "\\b", "gi");
      const matches = text.match(regex);
      if (matches) {
        positiveCount += matches.length;
        
        // Check for negations before positive words
        matches.forEach(() => {
          const negationRegex = new RegExp(`\\b(not|no|never|don't|doesn't|isn't|aren't|wasn't|weren't)\\s+[\\w\\s]{0,10}\\b${word}\\b`, 'gi');
          const negatedMatches = text.match(negationRegex);
          if (negatedMatches) {
            positiveCount -= negatedMatches.length;
            negativeCount += negatedMatches.length;
          }
        });
      }
    });
    
    // Count negative words with consideration for negations
    negativeWords.forEach(word => {
      const regex = new RegExp("\\b" + word + "\\b", "gi");
      const matches = text.match(regex);
      if (matches) {
        negativeCount += matches.length;
        
        // Check for negations before negative words
        matches.forEach(() => {
          const negationRegex = new RegExp(`\\b(not|no|never|don't|doesn't|isn't|aren't|wasn't|weren't)\\s+[\\w\\s]{0,10}\\b${word}\\b`, 'gi');
          const negatedMatches = text.match(negationRegex);
          if (negatedMatches) {
            negativeCount -= negatedMatches.length;
            positiveCount += negatedMatches.length;
          }
        });
      }
    });
    
    // Calculate sentiment score (0 to 1, where 0 is very negative, 1 is very positive)
    let score = 0.5; // Neutral by default
    const total = positiveCount + negativeCount;
    
    if (total > 0) {
      score = 0.5 + (positiveCount - negativeCount) / (2 * total);
      // Ensure score is within bounds
      score = Math.max(0, Math.min(1, score));
    }
    
    // Determine sentiment category
    let category = "neutral";
    if (score > 0.6) category = "positive";
    if (score < 0.4) category = "negative";
    
    return { category, score };
  }
  
  // Helper function to calculate confidence level
  function calculateConfidence(score, textLength, hasSentimentShift) {
    // Base confidence on how far score is from neutral (0.5)
    let confidence = Math.abs(score - 0.5) * 2; // 0 to 1 scale
    
    // Adjust confidence based on text length
    if (textLength < 20) {
      confidence *= 0.7; // Shorter texts have lower confidence
    } else if (textLength > 200) {
      confidence *= 0.85; // Longer texts have moderate confidence
    }
    
    // Reduce confidence if there are sentiment shifts
    if (hasSentimentShift) {
      confidence *= 0.8;
    }
    
    return Math.min(0.95, confidence); // Cap at 0.95 for offline mode
  }
  
  // Enhanced contextual summary generation for offline mode
  function generateSimpleSummary(text, sentimentResult) {
    if (!text || text.trim().length === 0) {
      return "No text provided to analyze.";
    }
    
    // Get basic text statistics
    const wordCount = text.split(/\s+/).filter(word => word.length > 0).length;
    const sentences = text.match(/[^\.!\?]+[\.!\?]+/g) || [text];
    const sentenceCount = sentences.length;
    
    // Extract key phrases (simple method for offline mode)
    const keyPhrases = extractKeyPhrases(text);
    
    // Build contextual summary
    let summary = `Analyzed text containing ${wordCount} words in ${sentenceCount} sentences. `;
    
    // Add sentiment context based on detailed analysis
    const { sentiment_trend, sentiment_shift, has_questions, section_sentiments } = sentimentResult.details;
    
    // Add sentiment trend information
    if (sentiment_trend === "improving") {
      summary += "The sentiment appears to improve throughout the text. ";
    } else if (sentiment_trend === "worsening") {
      summary += "The sentiment tends to become more negative as the text progresses. ";
    } else {
      summary += "The sentiment remains relatively consistent throughout the text. ";
    }
    
    // Add information about sentiment shifts if present
    if (sentiment_shift) {
      summary += "There are notable shifts in sentiment within the text. ";
    }
    
    // Add context about questions if present
    if (has_questions) {
      summary += "The text contains questions, suggesting an inquisitive tone. ";
    }
    
    // Add key phrases from the text
    if (keyPhrases.length > 0) {
      summary += `Key phrases identified: ${keyPhrases.slice(0, 3).join(", ")}`;
      if (keyPhrases.length > 3) {
        summary += ", and others";
      }
      summary += ". ";
    }
    
    // Add sentence excerpts based on sentiment
    if (sentimentResult.category === "positive") {
      const positiveSentences = findMostPositiveSentences(sentences, 1);
      if (positiveSentences.length > 0) {
        summary += `Notable positive excerpt: "${positiveSentences[0]}" `;
      }
    } else if (sentimentResult.category === "negative") {
      const negativeSentences = findMostNegativeSentences(sentences, 1);
      if (negativeSentences.length > 0) {
        summary += `Notable negative excerpt: "${negativeSentences[0]}" `;
      }
    }
    
    // Add confidence context
    if (sentimentResult.confidence < 0.4) {
      summary += "Note: The sentiment analysis has low confidence due to mixed or ambiguous language.";
    }
    
    return summary;
  }
  
  // Helper function to extract key phrases (simple version for offline mode)
  function extractKeyPhrases(text) {
    const words = text.toLowerCase().split(/\s+/);
    const stopWords = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", 
                      "by", "about", "as", "of", "from", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall",
                      "should", "can", "could", "may", "might", "must", "i", "you", "he", "she", 
                      "it", "we", "they", "me", "him", "her", "us", "them", "this", "that", "these",
                      "those", "my", "your", "his", "her", "its", "our", "their"];
    
    // Remove stopwords and short words
    const filteredWords = words.filter(word => 
      !stopWords.includes(word) && 
      word.length > 3 &&
      !/^\d+$/.test(word) // Exclude numbers
    );
    
    // Count word frequencies
    const wordFrequency = {};
    filteredWords.forEach(word => {
      wordFrequency[word] = (wordFrequency[word] || 0) + 1;
    });
    
    // Extract 2-gram phrases
    const phrases = [];
    for (let i = 0; i < words.length - 1; i++) {
      if (!stopWords.includes(words[i]) && !stopWords.includes(words[i+1])) {
        const phrase = words[i] + " " + words[i+1];
        phrases.push(phrase);
      }
    }
    
    // Sort words by frequency
    const sortedWords = Object.entries(wordFrequency)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(entry => entry[0]);
    
    // Combine single words and phrases
    return [...sortedWords, ...phrases.slice(0, 2)].slice(0, 5);
  }
  
  // Helper function to find the most positive sentences
  function findMostPositiveSentences(sentences, count) {
    // Simple heuristic for finding positive sentences
    const positiveWords = [
      "good", "great", "excellent", "awesome", "happy", "love", "like", "best", 
      "positive", "wonderful", "enjoy", "nice", "fantastic", "amazing", "perfect"
    ];
    
    // Score sentences based on positive word count
    const scoredSentences = sentences.map(sentence => {
      const lowerSentence = sentence.toLowerCase();
      let score = 0;
      positiveWords.forEach(word => {
        const regex = new RegExp("\\b" + word + "\\b", "gi");
        const matches = lowerSentence.match(regex);
        if (matches) score += matches.length;
      });
      return { sentence, score };
    });
    
    // Sort by score and get the top sentences
    return scoredSentences
      .sort((a, b) => b.score - a.score)
      .slice(0, count)
      .map(item => item.sentence.trim());
  }
  
  // Helper function to find the most negative sentences
  function findMostNegativeSentences(sentences, count) {
    // Simple heuristic for finding negative sentences
    const negativeWords = [
      "bad", "terrible", "awful", "hate", "dislike", "worst", "negative", "poor", 
      "horrible", "unfortunate", "sad", "angry", "upset", "disappoint", "frustrat"
    ];
    
    // Score sentences based on negative word count
    const scoredSentences = sentences.map(sentence => {
      const lowerSentence = sentence.toLowerCase();
      let score = 0;
      negativeWords.forEach(word => {
        const regex = new RegExp("\\b" + word + "\\b", "gi");
        const matches = lowerSentence.match(regex);
        if (matches) score += matches.length;
      });
      return { sentence, score };
    });
    
    // Sort by score and get the top sentences
    return scoredSentences
      .sort((a, b) => b.score - a.score)
      .slice(0, count)
      .map(item => item.sentence.trim());
  }
  
  // Save analysis to history
  function saveAnalysisToHistory(text, result) {
    // Only save if we have actual text and a result
    if (!text || !result) {
      console.error('Cannot save to history - missing text or result');
      return;
    }
    
    // Get existing history
    loadFromStorage({ 'analysisHistory': [] }, (data) => {
      const history = data.analysisHistory || [];
      
      // Create history item
      const historyItem = {
        text: text,
        timestamp: Date.now(),
        score: result.score,
        category: result.category,
        label: result.label,
        confidence: result.confidence,
        model_used: result.model_used
      };
      
      // Add to history
      history.push(historyItem);
      
      // Limit history size to 100 items
      if (history.length > 100) {
        history.sort((a, b) => b.timestamp - a.timestamp);
        history.length = 100;
      }
      
      // Save updated history
      saveToStorage({ 'analysisHistory': history }, function(success) {
        if (success) {
          logDebug('Analysis saved to history');
        } else {
          logDebug('Failed to save analysis to history');
        }
      });
    });
  }
  
  // Storage helper functions
  function loadFromStorage(defaults, callback) {
    try {
      chrome.storage.local.get(defaults, (result) => {
        callback(result);
      });
    } catch (error) {
      console.error('Storage load error:', error);
      logDebug(`Storage load error: ${error.message}`);
      callback(defaults);
    }
  }
  
  function saveToStorage(items, callback) {
    try {
      chrome.storage.local.set(items, () => {
        if (chrome.runtime.lastError) {
          console.error('Storage save error:', chrome.runtime.lastError);
          logDebug(`Storage save error: ${chrome.runtime.lastError.message}`);
          if (callback) callback(false);
          return;
        }
        if (callback) callback(true);
      });
    } catch (error) {
      console.error('Storage save error:', error);
      logDebug(`Storage save error: ${error.message}`);
      if (callback) callback(false);
    }
  }
});

