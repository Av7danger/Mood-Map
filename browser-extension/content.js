// Content script for MoodMap extension
// Handles detection and analysis of text content from web pages, 
// particularly focusing on tweets and social media posts

console.log("MoodMap content script loaded");

// Function to extract and analyze tweets on Twitter
function analyzeTweets() {
  console.log("Looking for tweets to analyze...");
  
  // Updated selector to catch more potential tweet elements
  const tweetElements = document.querySelectorAll('[data-testid="tweetText"], .tweet-text, [data-testid="post-content"], .post-content, article div[lang], [role="article"] div[lang]');
  
  if (tweetElements.length > 0) {
    console.log(`Found ${tweetElements.length} potential tweets to analyze`);
    
    // Process each tweet
    tweetElements.forEach((tweet, index) => {
      // Get the tweet text
      const tweetText = tweet.innerText.trim();
      
      if (tweetText && tweetText.length > 5) { // Ignore very short content
        console.log(`Analyzing tweet ${index + 1}:`, tweetText.substring(0, 50) + "...");
        
        // Send message to background script to analyze the text
        chrome.runtime.sendMessage({
          type: 'analyzeSentiment',
          text: tweetText
        }, response => {
          if (chrome.runtime.lastError) {
            console.error("Error sending message:", chrome.runtime.lastError);
            return;
          }
          
          if (response) {
            console.log("Analysis response received:", response);
            addSentimentIndicator(tweet, response);
          } else {
            console.error("No response received from background script");
          }
        });
      }
    });
  } else {
    console.log("No tweets found on this page");
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
  switch(sentimentData.prediction) {
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
  
  // Calculate percentage for display
  const percentage = sentimentData.sentiment_percentage || 
                     (sentimentData.score ? Math.round((sentimentData.score + 1) / 2 * 100) : 50);
  
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
  
  // Add content to the indicator
  indicator.innerHTML = `${emoji} ${sentimentData.sentiment || sentimentData.label} (${percentage}%)`;
  
  // Add the indicator to the tweet
  element.appendChild(indicator);
  
  // Log the addition
  console.log(`Added sentiment indicator: ${sentimentData.sentiment || sentimentData.label} (${percentage}%)`);
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
        chrome.runtime.sendMessage({
          type: 'analyzeSentiment',
          text: request.text
        }, response => {
          if (chrome.runtime.lastError) {
            console.error("Error in selected text analysis:", chrome.runtime.lastError);
            return;
          }
          
          if (response) {
            console.log("Selected text analysis result:", response);
            addSentimentIndicator(span, response);
          }
        });
      } catch (e) {
        console.error("Couldn't wrap selection:", e);
      }
    }
  }
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
  console.log("Page loaded, initializing tweet analysis");
  // Wait a moment for dynamic content to load
  setTimeout(analyzeTweets, 1500);
});

// Re-analyze when content changes (for single-page applications)
// Use MutationObserver to detect when new tweets are loaded
const observer = new MutationObserver((mutations) => {
  // Debounce the analysis to avoid excessive calls
  if (window.moodMapAnalysisTimeout) {
    clearTimeout(window.moodMapAnalysisTimeout);
  }
  
  window.moodMapAnalysisTimeout = setTimeout(() => {
    console.log("Content changed, re-analyzing tweets");
    analyzeTweets();
  }, 1000);
});

// Start observing the document body for changes
observer.observe(document.body, {
  childList: true, 
  subtree: true
});

// Initial analysis on script load
console.log("Running initial tweet analysis");
setTimeout(analyzeTweets, 1000);