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

// Function to analyze text and show result in overlay - DISABLED VERSION
function analyzeAndShowOverlay(text, isTweet = false, summarize = false) {
  // This function is now disabled to prevent popup windows
  console.log("MoodMap Analysis popup window has been disabled");
  
  // Still run the analysis but don't show the popup
  safeSendMessage({ 
    type: 'analyzeSentiment', 
    text: text,
    options: {
      summarize: summarize,
      model: summarize ? 'advanced' : undefined
    }
  }, function(response) {
    if (response && response.error) {
      console.error("Error analyzing sentiment:", response.error);
      return;
    }
    
    // Log the result but don't display the popup
    console.log("Analysis complete but popup disabled:", response);
  });
  
  // Return null instead of creating the overlay
  return null;
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
  const existingOverlay = document.querySelector('.mood-map-detailed-overlay');
  if (existingOverlay) {
    existingOverlay.remove();
  }
  
  // Create overlay container
  const overlay = document.createElement('div');
  overlay.className = 'mood-map-detailed-overlay';
  
  // Use system dark mode preference if available
  const isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (isDarkMode) {
    overlay.classList.add('dark-mode');
  }
  
  // Calculate position (near the element but visible)
  const rect = element.getBoundingClientRect();
  
  // Create header with title and close button
  const header = document.createElement('div');
  header.className = 'mood-map-overlay-header';
  header.innerHTML = `
    <div class="mood-map-overlay-title">MoodMap Analysis</div>
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
  const confidenceValue = sentimentData.confidence || Math.abs(sentimentData.score || 0.5);
  const confidenceMeter = document.createElement('div');
  confidenceMeter.innerHTML = `
    <div class="mood-map-confidence-meter">
      <div class="mood-map-confidence-value" style="width: ${confidenceValue * 100}%"></div>
    </div>
    <div class="mood-map-confidence-text">Confidence: ${Math.round(confidenceValue * 100)}%</div>
  `;
  
  // Add summary if available
  let summarySection = '';
  if (sentimentData.summary) {
    summarySection = `
      <div class="mood-map-summary-section">
        <div class="mood-map-summary-label">Summary</div>
        <div class="mood-map-summary-text">${sentimentData.summary}</div>
      </div>
    `;
  }
  
  // Add analyzed text preview
  const textPreview = document.createElement('div');
  textPreview.className = 'mood-map-selection-text';
  textPreview.textContent = element.innerText.trim().substring(0, 150) + (element.innerText.length > 150 ? '...' : '');
  
  // Add styles specific to this overlay
  const style = document.createElement('style');
  style.textContent = `
    .mood-map-detailed-overlay {
      position: absolute;
      z-index: 10000;
      background-color: white;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      padding: 12px;
      width: 320px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      color: #333;
      font-size: 14px;
    }
    
    .mood-map-detailed-overlay.dark-mode {
      background-color: #282c34;
      color: #e1e1e1;
      border: 1px solid #444;
    }
    
    .mood-map-overlay-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      border-bottom: 1px solid #eee;
      padding-bottom: 8px;
    }
    
    .dark-mode .mood-map-overlay-header {
      border-bottom-color: #444;
    }
    
    .mood-map-overlay-title {
      font-weight: bold;
      font-size: 16px;
    }
    
    .mood-map-overlay-close {
      cursor: pointer;
      font-size: 20px;
      line-height: 1;
      padding: 0 5px;
    }
    
    .mood-map-sentiment-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }
    
    .mood-map-sentiment-label {
      font-weight: 500;
    }
    
    .mood-map-sentiment-value {
      font-weight: bold;
      padding: 3px 8px;
      border-radius: 12px;
      color: white;
    }
    
    .mood-map-sentiment-value.positive {
      background-color: #4caf50;
    }
    
    .mood-map-sentiment-value.neutral {
      background-color: #9e9e9e;
    }
    
    .mood-map-sentiment-value.negative {
      background-color: #f44336;
    }
    
    .mood-map-confidence-meter {
      height: 8px;
      background-color: #f1f1f1;
      border-radius: 4px;
      margin-bottom: 5px;
      overflow: hidden;
    }
    
    .dark-mode .mood-map-confidence-meter {
      background-color: #3c3c3c;
    }
    
    .mood-map-confidence-value {
      height: 100%;
      background-color: #2196f3;
    }
    
    .mood-map-confidence-text {
      font-size: 12px;
      color: #666;
      text-align: right;
      margin-bottom: 12px;
    }
    
    .dark-mode .mood-map-confidence-text {
      color: #aaa;
    }
    
    .mood-map-summary-section {
      margin-bottom: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid #eee;
    }
    
    .dark-mode .mood-map-summary-section {
      border-bottom-color: #444;
    }
    
    .mood-map-summary-label {
      font-weight: 500;
      margin-bottom: 5px;
    }
    
    .mood-map-summary-text {
      font-size: 13px;
      line-height: 1.4;
    }
    
    .mood-map-selection-text {
      margin-top: 12px;
      font-size: 13px;
      line-height: 1.5;
      max-height: 100px;
      overflow-y: auto;
      opacity: 0.8;
    }
  `;
  
  // Assemble the overlay
  overlay.appendChild(style);
  overlay.appendChild(header);
  results.appendChild(sentimentRow);
  results.appendChild(confidenceMeter);
  if (summarySection) {
    const summaryDiv = document.createElement('div');
    summaryDiv.className = 'mood-map-summary-section';
    summaryDiv.innerHTML = summarySection;
    results.appendChild(summaryDiv);
  }
  overlay.appendChild(results);
  overlay.appendChild(textPreview);
  
  // Position overlay - making sure it's visible in the viewport
  overlay.style.position = 'absolute';
  
  // Calculate optimal position to avoid going off-screen
  const viewportHeight = window.innerHeight;
  const viewportWidth = window.innerWidth;
  const overlayHeight = 300; // Estimated height
  
  // Position vertically
  if (rect.bottom + overlayHeight + 20 < viewportHeight) {
    // Position below the element if there's room
    overlay.style.top = `${rect.bottom + window.scrollY + 10}px`;
  } else {
    // Position above the element if there's not enough room below
    overlay.style.top = `${rect.top + window.scrollY - overlayHeight - 10}px`;
  }
  
  // Position horizontally
  if (rect.left + 320 < viewportWidth) {
    // Align with left edge if possible
    overlay.style.left = `${rect.left + window.scrollX}px`;
  } else {
    // Align with right edge if not enough room
    overlay.style.left = `${viewportWidth - 340}px`;
  }
  
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
  
  // Log the detailed analysis
  console.log('Showing detailed analysis:', sentimentData);
  
  return overlay;
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

// Show floating analysis when we can't modify the page directly - DISABLED VERSION
function showFloatingAnalysis(text, result) {
  // This function is now disabled to prevent popup windows
  console.log("Floating analysis popup window has been disabled");
  console.log("Analysis result:", result);
  return null;
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

// Analyze a tweet page automatically and immediately
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
  
  // Immediately send the text to the analysis API
  safeSendMessage({ 
    type: 'analyzeSentiment', 
    text: tweetText,
    options: {
      summarize: true,
      model: 'ensemble'
    }
  }, function(response) {
    if (response && response.error) {
      console.error("Error analyzing sentiment:", response.error);
      return;
    }
    
    console.log("Analysis complete:", response);
    
    // Instead of displaying the result, just analyze the tweet for emoji-click functionality
    // Find the tweet element to add an invisible indicator that can be clicked
    const tweetArticle = document.querySelector('article[data-testid="tweet"]');
    if (tweetArticle) {
      // Find the tweet text element inside the article
      const tweetTextElement = tweetArticle.querySelector('[data-testid="tweetText"]');
      if (tweetTextElement) {
        // Add a subtle indicator that can be clicked but isn't as visible as the black bar
        addSentimentIndicator(tweetTextElement, response);
      }
    }
  });
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