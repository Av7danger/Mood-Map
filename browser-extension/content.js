// Content script for MoodMap extension
console.log("MoodMap content script loaded");

// Keep track of extension context validity
let extensionContextValid = true;
// Add flag to track if we've already analyzed a tweet on this page
let tweetAlreadyAnalyzed = false;

// Global error handler for extension context invalidation
self.onerror = function(message, source, lineno, colno, error) {
  console.error("Unhandled error in content script:", message, error);
  
  // If the error is about extension context, mark it as invalid
  if (message && typeof message === 'string' && message.includes("Extension context invalidated")) {
    extensionContextValid = false;
    console.warn("Extension context has been invalidated, will require reload");
  }
  
  return true;
};

// Helper function to safely send messages to the background script
function safeSendMessage(message, callback) {
  if (!extensionContextValid) {
    console.warn("Extension context invalidated, not sending message");
    if (typeof callback === 'function') {
      callback({ error: "Extension context invalidated" });
    }
    return;
  }
  
  try {
    chrome.runtime.sendMessage(message, function(response) {
      // Check for runtime errors
      const lastError = chrome.runtime.lastError;
      if (lastError) {
        console.error("Runtime error:", lastError.message);
        
        // Check for extension context invalidation
        if (lastError.message.includes("Extension context invalidated")) {
          extensionContextValid = false;
        }
        
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
    
    // Check for extension context invalidation
    if (error.message.includes("Extension context invalidated")) {
      extensionContextValid = false;
    }
    
    if (typeof callback === 'function') {
      callback({ error: error.message });
    }
  }
}

// Listen for messages from background script
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  try {
    console.log("Content script received message:", request);
    
    // Handle extension context invalidation preemptively
    if (!extensionContextValid) {
      console.warn("Extension context already invalidated, cannot process message");
      sendResponse({ error: "Extension context invalidated" });
      return false;
    }
    
    // Handle analyze selected text request
    if (request.type === "analyzeSelectedText") {
      console.log("Analyzing selected text:", request.text.substring(0, 50) + "...");
      
      // Create and show overlay with result
      analyzeAndShowOverlay(request.text);
      
      // Acknowledge receipt of message
      sendResponse({ status: "processing" });
      return true;
    }
    
    // Default response for unhandled message types
    sendResponse({ error: "Unhandled message type" });
    return true;
    
  } catch (error) {
    console.error("Error handling message:", error);
    
    // Check for extension context invalidation
    if (error.message.includes("Extension context invalidated")) {
      extensionContextValid = false;
    }
    
    // Try to send error response
    try {
      sendResponse({ error: error.message });
    } catch (e) {
      console.error("Could not send error response:", e);
    }
    
    return false;
  }
});

// Function to analyze text and show result in overlay
function analyzeAndShowOverlay(text, isTweet = false, summarize = false) {
  // Create overlay container if it doesn't exist
  let overlay = document.getElementById('mood-map-overlay');
  
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'mood-map-overlay';
    overlay.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 9999;
      background: white;
      border: 1px solid #ccc;
      border-radius: 8px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.2);
      padding: 15px;
      max-width: 350px;
      font-family: Arial, sans-serif;
      transition: opacity 0.3s ease;
    `;
    document.body.appendChild(overlay);
  }
  
  // Show loading state
  overlay.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <h3 style="margin:0;color:#333;">MoodMap Analysis</h3>
      <button id="close-mood-map-overlay" style="background:none;border:none;cursor:pointer;font-size:16px;">✕</button>
    </div>
    <div style="text-align:center;padding:20px;">
      <div class="loading-spinner" style="display:inline-block;width:30px;height:30px;border:3px solid #f3f3f3;border-top:3px solid #3498db;border-radius:50%;animation:spin 1s linear infinite;"></div>
      <p>Analyzing ${isTweet ? 'tweet' : 'text'}${summarize ? ' and generating summary' : ''}...</p>
    </div>
    <style>
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
    </style>
  `;
  
  // Add close button handler
  const closeButton = document.getElementById('close-mood-map-overlay');
  if (closeButton) {
    closeButton.addEventListener('click', function() {
      overlay.style.opacity = '0';
      setTimeout(() => {
        overlay.remove();
      }, 300);
    });
  }
  
  // Send text to background script for analysis
  safeSendMessage({ 
    type: 'analyzeSentiment', 
    text: text,
    options: {
      summarize: summarize,
      model: summarize ? 'advanced' : undefined  // Use advanced model for summarization
    }
  }, function(response) {
    if (response && response.error) {
      console.error("Error analyzing sentiment:", response.error);
      
      // Check if overlay still exists
      if (document.getElementById('mood-map-overlay')) {
        overlay.innerHTML = `
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <h3 style="margin:0;color:#333;">MoodMap Analysis</h3>
            <button id="close-mood-map-overlay" style="background:none;border:none;cursor:pointer;font-size:16px;">✕</button>
          </div>
          <div style="padding:10px;">
            <p style="color:red;">Error: ${response.error}</p>
            <p>Please try again or check that the extension is working properly.</p>
          </div>
        `;
        
        // Re-add close button handler
        const closeButton = document.getElementById('close-mood-map-overlay');
        if (closeButton) {
          closeButton.addEventListener('click', function() {
            overlay.style.opacity = '0';
            setTimeout(() => {
              overlay.remove();
            }, 300);
          });
        }
      }
      return;
    }
    
    // Check if overlay still exists
    if (!document.getElementById('mood-map-overlay')) {
      return;
    }
    
    // Determine emoji and color based on sentiment
    let emoji = '😐'; // Neutral
    let color = '#888888';
    let sentiment = 'neutral';
    
    if (response.category === 0 || response.sentiment < -0.3) {
      emoji = '😞'; // Negative
      color = '#e74c3c';
      sentiment = 'negative';
    } else if (response.category === 2 || response.sentiment > 0.3) {
      emoji = '😊'; // Positive
      color = '#2ecc71';
      sentiment = 'positive';
    }
    
    // Create sentiment bars
    const sentimentBars = `
      <div style="margin-top:10px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
          <span>Negative</span>
          <span id="negative-value">0%</span>
        </div>
        <div style="height:8px;background:#f1f1f1;border-radius:4px;overflow:hidden;margin-bottom:10px;">
          <div id="negative-bar" style="height:100%;background:#e74c3c;width:${response.category === 0 ? '80%' : '20%'};"></div>
        </div>
        
        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
          <span>Neutral</span>
          <span id="neutral-value">0%</span>
        </div>
        <div style="height:8px;background:#f1f1f1;border-radius:4px;overflow:hidden;margin-bottom:10px;">
          <div id="neutral-bar" style="height:100%;background:#888888;width:${response.category === 1 ? '80%' : '20%'};"></div>
        </div>
        
        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
          <span>Positive</span>
          <span id="positive-value">0%</span>
        </div>
        <div style="height:8px;background:#f1f1f1;border-radius:4px;overflow:hidden;margin-bottom:10px;">
          <div id="positive-bar" style="height:100%;background:#2ecc71;width:${response.category === 2 ? '80%' : '20%'};"></div>
        </div>
      </div>
    `;
    
    // Build summary section if available
    let summarySection = '';
    if (response.summary) {
      summarySection = `
        <div style="margin-top:15px;padding:10px;background:#f9f9f9;border-radius:5px;border-left:3px solid #3498db;">
          <h4 style="margin:0 0 5px 0;color:#3498db;">Summary</h4>
          <p style="margin:0;font-size:14px;">${response.summary}</p>
        </div>
      `;
    }
    
    // Display result
    overlay.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <h3 style="margin:0;color:#333;">MoodMap Analysis</h3>
        <button id="close-mood-map-overlay" style="background:none;border:none;cursor:pointer;font-size:16px;">✕</button>
      </div>
      <div style="text-align:center;margin-bottom:15px;">
        <div style="font-size:48px;margin-bottom:5px;">${emoji}</div>
        <div style="font-weight:bold;color:${color};font-size:18px;">${sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}</div>
        <div style="color:#666;margin-top:5px;font-size:14px;">Confidence: ${Math.round((response.confidence || 0.6) * 100)}%</div>
      </div>
      ${sentimentBars}
      ${summarySection}
      <div style="margin-top:15px;padding-top:10px;border-top:1px solid #eee;font-size:14px;color:#666;">
        <p style="margin:0 0 5px 0;">Analyzed ${isTweet ? 'tweet' : 'text'}: "${text.length > 100 ? text.substring(0, 100) + '...' : text}"</p>
        <p style="margin:0;font-size:12px;">Powered by MoodMap</p>
      </div>
    `;
    
    // Re-add close button handler
    const closeButton = document.getElementById('close-mood-map-overlay');
    if (closeButton) {
      closeButton.addEventListener('click', function() {
        overlay.style.opacity = '0';
        setTimeout(() => {
          overlay.remove();
        }, 300);
      });
    }
    
    // Auto-close after 30 seconds for automatic tweet analysis
    setTimeout(() => {
      const overlay = document.getElementById('mood-map-overlay');
      if (overlay) {
        overlay.style.opacity = '0';
        setTimeout(() => {
          if (overlay.parentNode) {
            overlay.remove();
          }
        }, 300);
      }
    }, 30000);
  });
}

// Check if extension context is still valid on page load
function checkExtensionContext() {
  try {
    // Try to send a ping message to background script
    safeSendMessage({ type: 'ping' }, (response) => {
      if (response && response.error) {
        if (response.error.includes("Extension context invalidated")) {
          extensionContextValid = false;
          console.warn("Extension context invalidated, content script needs reload");
        }
      } else if (response && response.type === 'pong') {
        console.log("Extension context is valid, background script responded");
      }
    });
  } catch (error) {
    console.error("Error checking extension context:", error);
    if (error.message.includes("Extension context invalidated")) {
      extensionContextValid = false;
    }
  }
}

// Check extension context on load
checkExtensionContext();

// Periodic check to ensure extension context is still valid
setInterval(checkExtensionContext, 60000); // Check every minute

console.log("MoodMap content script initialization complete");

// Check if we're on a social media site and set up the appropriate selectors
function setupSelectors() {
  let host = window.location.hostname;
  
  if (host.includes('twitter.com') || host.includes('x.com')) {
    return {
      posts: [
        '[data-testid="tweetText"]', 
        '.tweet-text', 
        '[data-testid="post-content"]',
        '[data-testid="tweet"]',
        '.timeline-Tweet-text'
      ].join(', '),
      platform: 'twitter'
    };
  } 
  else if (host.includes('facebook.com')) {
    return {
      posts: [
        '.userContent', 
        '[data-ad-preview="message"]', 
        '[data-testid="post_message"]'
      ].join(', '),
      platform: 'facebook'
    };
  } 
  else if (host.includes('linkedin.com')) {
    return {
      posts: [
        '.feed-shared-update-v2__description', 
        '.feed-shared-text', 
        '.comments-comment-item__main-content'
      ].join(', '),
      platform: 'linkedin'
    };
  } 
  else if (host.includes('reddit.com')) {
    return {
      posts: [
        '[data-testid="comment"]', 
        '.md', 
        '.Comment__body',
        '.PostHeader__post-title-line',
        '.RichTextJSON-root'
      ].join(', '),
      platform: 'reddit'
    };
  } 
  else {
    // Generic selectors for other sites
    return {
      posts: [
        'article div[lang]', 
        '[role="article"] div[lang]',
        '[aria-labelledby][aria-describedby]',
        '.post-content',
        '.message-body',
        '.comment-content'
      ].join(', '),
      platform: 'other'
    };
  }
}

// Function to extract and analyze tweets/posts on social media sites
function analyzePosts() {
  console.log("Looking for content to analyze...");
  
  // Get platform-specific selectors
  const platformData = setupSelectors();
  console.log(`Detected platform: ${platformData.platform}`);
  
  // Use platform-specific selectors to find posts
  const postElements = document.querySelectorAll(platformData.posts);
  
  if (postElements.length > 0) {
    console.log(`Found ${postElements.length} potential posts to analyze`);
    
    // Process each post
    postElements.forEach((post, index) => {
      // Skip if we already added an indicator
      if (post.querySelector('.mood-map-indicator')) {
        return;
      }
      
      // Skip if it's a parent of another post element we're already handling
      if (Array.from(postElements).some(el => el !== post && post.contains(el))) {
        return;
      }
      
      // Get the post text
      const postText = post.innerText.trim();
      
      if (postText && postText.length > 5) { // Ignore very short content
        console.log(`Analyzing post ${index + 1}:`, postText.substring(0, 50) + "...");
        
        // Send message to background script to analyze the text
        safeSendMessage({
          type: 'analyzeSentiment',
          text: postText
        }, response => {
          if (response && response.error) {
            console.error("Error analyzing sentiment:", response.error);
            return;
          }
          
          if (response) {
            console.log("Analysis response received:", response);
            addSentimentIndicator(post, response);
          } else {
            console.error("No response received from background script");
          }
        });
      }
    });
  } else {
    console.log("No posts found on this page");
  }
}

// Function to add sentiment indicator to analyzed content
function addSentimentIndicator(element, sentimentData) {
  // Check if we already added an indicator to this element
  if (element.querySelector('.mood-map-indicator')) {
    return;
  }
  
  // Create sentiment indicator element
  const indicator = document.createElement('div');
  indicator.className = 'mood-map-indicator';
  
  // Set indicator style based on sentiment
  let color, emoji;
  
  // Handle different response formats by normalizing the data
  let category = 1; // Default to neutral
  
  if (sentimentData.category !== undefined) {
    // Use category directly if available (0=negative, 1=neutral, 2=positive)
    category = sentimentData.category;
  } else if (sentimentData.prediction !== undefined) {
    // Use prediction if category is not available
    category = sentimentData.prediction;
  } else if (sentimentData.score !== undefined) {
    // Convert score to category if that's all we have
    if (sentimentData.score < -0.3) category = 0;
    else if (sentimentData.score > 0.3) category = 2;
    else category = 1;
  }
  
  // Map category to display properties
  switch(category) {
    case 0: // Negative
      color = '#ff4c4c';
      emoji = '😞';
      break;
    case 2: // Positive
      color = '#4caf50';
      emoji = '😊';
      break;
    default: // Neutral
      color = '#9e9e9e';
      emoji = '😐';
  }
  
  // Get the sentiment label
  let sentimentLabel = sentimentData.label || sentimentData.sentiment || 'Neutral';
  // Make first letter uppercase for consistency
  sentimentLabel = sentimentLabel.charAt(0).toUpperCase() + sentimentLabel.slice(1).toLowerCase();
  
  // Style the indicator
  indicator.style.cssText = `
    display: inline-flex;
    align-items: center;
    margin-left: 10px;
    padding: 2px 8px;
    border-radius: 12px;
    background-color: ${color}22;
    color: ${color};
    font-weight: bold;
    font-size: 12px;
    border: 1px solid ${color}55;
  `;
  
  // Add emoji to the indicator
  indicator.textContent = emoji;
  
  // Make indicator interactive - show more details on click
  indicator.addEventListener('click', function(e) {
    e.stopPropagation();
    showDetailedOverlay(element, sentimentData, {emoji, color, sentimentLabel});
  });
  
  // Add the indicator to the element
  element.appendChild(indicator);
  
  // Log the addition
  console.log(`Added sentiment indicator: ${sentimentLabel} (emoji: ${emoji})`);
}

// Function to show detailed sentiment overlay
function showDetailedOverlay(element, sentimentData, displayData) {
  // Remove any existing overlay first
  const existingOverlay = document.querySelector('.mood-map-selection-overlay');
  if (existingOverlay) {
    existingOverlay.remove();
  }
  
  // Create overlay container
  const overlay = document.createElement('div');
  overlay.className = 'mood-map-selection-overlay';
  
  // Use system dark mode preference if available
  if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    overlay.classList.add('dark-mode');
  }
  
  // Calculate position (near the element but visible)
  const rect = element.getBoundingClientRect();
  
  // Create header with title and close button
  const header = document.createElement('div');
  header.className = 'mood-map-overlay-header';
  header.innerHTML = `
    <div class="mood-map-overlay-title">Mood Map Analysis</div>
    <div class="mood-map-overlay-close">×</div>
  `;
  
  // Create results content
  const results = document.createElement('div');
  results.className = 'mood-map-selection-result';
  
  // Add sentiment info
  const sentimentRow = document.createElement('div');
  sentimentRow.className = 'mood-map-sentiment-row';
  sentimentRow.innerHTML = `
    <div class="mood-map-sentiment-label">Sentiment</div>
    <div class="mood-map-sentiment-value ${displayData.sentimentLabel.toLowerCase()}">${displayData.sentimentLabel}</div>
  `;
  
  // Create confidence meter
  const confidenceValue = sentimentData.confidence || 0.75;
  const confidenceMeter = document.createElement('div');
  confidenceMeter.innerHTML = `
    <div class="mood-map-confidence-meter">
      <div class="mood-map-confidence-value" style="width: ${confidenceValue * 100}%"></div>
    </div>
    <div class="mood-map-confidence-text">Confidence: ${Math.round(confidenceValue * 100)}%</div>
  `;
  
  // Add analyzed text preview
  const textPreview = document.createElement('div');
  textPreview.className = 'mood-map-selection-text';
  textPreview.textContent = element.innerText.trim().substring(0, 150) + (element.innerText.length > 150 ? '...' : '');
  
  // Assemble the overlay
  overlay.appendChild(header);
  results.appendChild(sentimentRow);
  results.appendChild(confidenceMeter);
  overlay.appendChild(results);
  overlay.appendChild(textPreview);
  
  // Position overlay
  overlay.style.position = 'absolute';
  overlay.style.top = `${rect.bottom + window.scrollY + 10}px`;
  overlay.style.left = `${rect.left + window.scrollX}px`;
  
  // Add overlay to document
  document.body.appendChild(overlay);
  
  // Handle close button click
  const closeButton = overlay.querySelector('.mood-map-overlay-close');
  closeButton.addEventListener('click', () => {
    overlay.remove();
  });
  
  // Close overlay when clicking outside
  document.addEventListener('click', function closeOverlay(e) {
    if (!overlay.contains(e.target) && e.target !== element) {
      overlay.remove();
      document.removeEventListener('click', closeOverlay);
    }
  });
}

// Function to handle selected text analysis (via context menu)
function handleSelectedTextAnalysis(request) {
  if (request.type === 'analyzeSelectedText' && request.text) {
    console.log("Analyzing selected text:", request.text.substring(0, 50) + "...");
    
    // Get current selection range
    const selection = window.getSelection();
    if (selection.rangeCount > 0) {
      const range = selection.getRangeAt(0);
      const span = document.createElement('span');
      span.className = 'mood-map-selection';
      
      // Wrap the selection in our span for highlighting
      try {
        range.surroundContents(span);
        
        // Send message to background script for analysis
        safeSendMessage({
          type: 'analyzeSentiment',
          text: request.text
        }, response => {
          if (response && response.error) {
            console.error("Error analyzing sentiment:", response.error);
            return;
          }
          
          if (response) {
            console.log("Selected text analysis result:", response);
            addSentimentIndicator(span, response);
          }
        });
      } catch (e) {
        console.error("Couldn't wrap selection:", e);
        
        // Alternative method if surroundContents fails
        const tempSpan = document.createElement('span');
        tempSpan.className = 'mood-map-selection';
        tempSpan.textContent = request.text;
        
        // Just show a floating analysis overlay instead
        document.body.appendChild(tempSpan);
        tempSpan.style.position = 'absolute';
        tempSpan.style.left = '-9999px';
        
        safeSendMessage({
          type: 'analyzeSentiment',
          text: request.text
        }, response => {
          if (response) {
            showFloatingAnalysis(request.text, response);
            tempSpan.remove();
          }
        });
      }
    }
  }
}

// Show floating analysis when we can't modify the page directly
function showFloatingAnalysis(text, result) {
  // Create a floating overlay for the result
  const overlay = document.createElement('div');
  overlay.className = 'mood-map-selection-overlay';
  overlay.style.position = 'fixed';
  overlay.style.top = '20%';
  overlay.style.right = '20px';
  overlay.style.zIndex = '10000';
  
  // Use system dark mode preference if available
  if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    overlay.classList.add('dark-mode');
  }
  
  // Create header with title and close button
  const header = document.createElement('div');
  header.className = 'mood-map-overlay-header';
  header.innerHTML = `
    <div class="mood-map-overlay-title">Selected Text Analysis</div>
    <div class="mood-map-overlay-close">×</div>
  `;
  
  // Get sentiment category
  let category = 1; // Default neutral
  let sentimentLabel = 'Neutral';
  let emoji = '😐';
  let color = '#9e9e9e';
  
  if (result.category !== undefined) {
    category = result.category;
  } else if (result.prediction !== undefined) {
    category = result.prediction;
  } else if (result.score !== undefined) {
    if (result.score < -0.3) category = 0;
    else if (result.score > 0.3) category = 2;
    else category = 1;
  }
  
  // Map category to display properties
  switch(category) {
    case 0: // Negative
      color = '#ff4c4c';
      emoji = '😞';
      sentimentLabel = 'Negative';
      break;
    case 2: // Positive
      color = '#4caf50';
      emoji = '😊';
      sentimentLabel = 'Positive';
      break;
    default: // Neutral
      color = '#9e9e9e';
      emoji = '😐';
      sentimentLabel = 'Neutral';
  }
  
  // Create results content
  const results = document.createElement('div');
  results.className = 'mood-map-selection-result';
  
  // Add sentiment info
  const sentimentRow = document.createElement('div');
  sentimentRow.className = 'mood-map-sentiment-row';
  sentimentRow.innerHTML = `
    <div class="mood-map-sentiment-label">Sentiment</div>
    <div class="mood-map-sentiment-value ${sentimentLabel.toLowerCase()}">${sentimentLabel}</div>
  `;
  
  // Create confidence meter
  const confidenceValue = result.confidence || 0.75;
  const confidenceMeter = document.createElement('div');
  confidenceMeter.innerHTML = `
    <div class="mood-map-confidence-meter">
      <div class="mood-map-confidence-value" style="width: ${confidenceValue * 100}%"></div>
    </div>
    <div class="mood-map-confidence-text">Confidence: ${Math.round(confidenceValue * 100)}%</div>
  `;
  
  // Add analyzed text preview
  const textPreview = document.createElement('div');
  textPreview.className = 'mood-map-selection-text';
  textPreview.textContent = text.substring(0, 150) + (text.length > 150 ? '...' : '');
  
  // Assemble the overlay
  overlay.appendChild(header);
  results.appendChild(sentimentRow);
  results.appendChild(confidenceMeter);
  overlay.appendChild(results);
  overlay.appendChild(textPreview);
  
  // Add overlay to document
  document.body.appendChild(overlay);
  
  // Handle close button click
  const closeButton = overlay.querySelector('.mood-map-overlay-close');
  closeButton.addEventListener('click', () => {
    overlay.remove();
  });
  
  // Close overlay when clicking outside
  document.addEventListener('click', function closeOverlay(e) {
    if (!overlay.contains(e.target)) {
      overlay.remove();
      document.removeEventListener('click', closeOverlay);
    }
  });
  
  // Auto-remove after 15 seconds
  setTimeout(() => {
    if (document.body.contains(overlay)) {
      overlay.remove();
    }
  }, 15000);
}

// Listen for messages from the background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log("Content script received message:", request.type);
  
  if (request.type === 'analyzeSelectedText') {
    handleSelectedTextAnalysis(request);
    return true;
  }
  
  return false;
});

// Initialize analysis after page load
window.addEventListener('load', () => {
  console.log("Page loaded, initializing content analysis");
  // Wait a moment for dynamic content to load
  setTimeout(analyzePosts, 1500);
});

// Re-analyze when content changes (for single-page applications)
// Use MutationObserver to detect when new posts are loaded
const observer = new MutationObserver((mutations) => {
  // Debounce the analysis to avoid excessive calls
  if (window.moodMapAnalysisTimeout) {
    clearTimeout(window.moodMapAnalysisTimeout);
  }
  
  window.moodMapAnalysisTimeout = setTimeout(() => {
    console.log("Content changed, re-analyzing posts");
    analyzePosts();
  }, 1000);
});

// Start observing the document body for changes
observer.observe(document.body, {
  childList: true, 
  subtree: true
});

// Initial analysis on script load
console.log("Running initial content analysis");
setTimeout(analyzePosts, 1000);

// Check if we're on a single tweet page
function isSingleTweetPage() {
  // Check URL patterns for Twitter/X single tweet page
  const url = window.location.href;
  return (
    (url.includes('twitter.com') || url.includes('x.com')) && 
    (url.includes('/status/') || url.match(/\/[^\/]+\/status\/\d+/))
  );
}

// Extract the main tweet text from a tweet page
function extractTweetText() {
  // Try different selectors that might contain the tweet text
  const possibleSelectors = [
    '[data-testid="tweetText"]',
    '.tweet-text',
    '[data-testid="tweet"] > div:nth-child(2)',
    '.css-901oao.r-18jsvk2.r-37j5jr.r-a023e6.r-16dba41.r-rjixqe.r-bcqeeo.r-qvutc0'
  ];
  
  for (const selector of possibleSelectors) {
    const elements = document.querySelectorAll(selector);
    if (elements && elements.length > 0) {
      // Use the first element that is visible and has text
      for (const element of elements) {
        const text = element.innerText.trim();
        if (text && text.length > 0 && isElementVisible(element)) {
          console.log(`Found tweet text using selector: ${selector}`);
          return text;
        }
      }
    }
  }
  
  // Fallback: try to find any element that looks like a tweet
  const articleElements = document.querySelectorAll('article');
  for (const article of articleElements) {
    // Look for elements with reasonable amount of text inside the article
    const textElements = article.querySelectorAll('div[dir="auto"], div[lang], span[dir="auto"]');
    for (const el of textElements) {
      const text = el.innerText.trim();
      if (text && text.length > 20 && text.length < 500 && isElementVisible(el)) {
        console.log('Found tweet text using fallback method');
        return text;
      }
    }
  }
  
  return null;
}

// Check if an element is visible
function isElementVisible(element) {
  const style = window.getComputedStyle(element);
  return style.display !== 'none' && 
         style.visibility !== 'hidden' && 
         element.offsetWidth > 0 && 
         element.offsetHeight > 0;
}

// Analyze a tweet page automatically
function analyzeTweetPage() {
  if (!isSingleTweetPage() || tweetAlreadyAnalyzed) {
    return;
  }
  
  console.log('Single tweet page detected, analyzing tweet automatically');
  
  // Extract tweet text
  const tweetText = extractTweetText();
  if (!tweetText) {
    console.log('Could not extract tweet text, will retry');
    // Retry after a short delay as content might still be loading
    setTimeout(analyzeTweetPage, 1500);
    return;
  }
  
  console.log('Extracted tweet text:', tweetText);
  tweetAlreadyAnalyzed = true;
  
  // Analyze the tweet and generate a summary
  analyzeAndShowOverlay(tweetText, true, true);
}

// Check for a tweet page and analyze on page load
window.addEventListener('load', () => {
  console.log("Page loaded, checking if this is a tweet page");
  // Wait a moment for dynamic content to load
  setTimeout(analyzeTweetPage, 1500);
});

// Re-analyze when navigation happens (for single-page applications)
let lastUrl = location.href;
new MutationObserver(() => {
  if (location.href !== lastUrl) {
    lastUrl = location.href;
    console.log('URL changed, checking if this is a tweet page');
    tweetAlreadyAnalyzed = false;
    setTimeout(analyzeTweetPage, 1500);
  }
}).observe(document, {subtree: true, childList: true});