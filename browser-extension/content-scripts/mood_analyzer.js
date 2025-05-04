console.log("mood_analyzer.js loaded");

// Add visual indicator that extension is loaded
if (window.location.href.includes('twitter.com') || window.location.href.includes('x.com')) {
    console.log("Twitter/X detected - adding visual indicator");
    setTimeout(() => {
        const indicator = document.createElement('div');
        indicator.style.position = 'fixed';
        indicator.style.top = '10px';
        indicator.style.left = '10px';
        indicator.style.backgroundColor = 'rgba(0, 128, 255, 0.7)';
        indicator.style.color = 'white';
        indicator.style.padding = '5px 10px';
        indicator.style.borderRadius = '5px';
        indicator.style.zIndex = '9999';
        indicator.style.fontSize = '12px';
        indicator.style.fontWeight = 'bold';
        indicator.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        indicator.textContent = 'Mood Map Extension Active';
        document.body.appendChild(indicator);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            indicator.remove();
        }, 5000);
    }, 1000);
}

// Track the current URL to detect navigation to individual posts
let currentUrl = window.location.href;

// Function to detect if we're on an individual post page based on URL patterns
function isIndividualPostPage(url) {
    return (
        // Twitter/X.com individual tweet
        (url.includes('twitter.com') && url.match(/twitter\.com\/\w+\/status\/\d+/)) ||
        (url.includes('x.com') && url.match(/x\.com\/\w+\/status\/\d+/)) ||
        // Facebook post
        (url.includes('facebook.com') && (url.includes('/posts/') || url.includes('/permalink/'))) ||
        // Reddit post
        (url.includes('reddit.com') && url.includes('/comments/')) ||
        // LinkedIn post
        (url.includes('linkedin.com') && url.includes('/posts/')) ||
        // Instagram post
        (url.includes('instagram.com') && url.includes('/p/'))
    );
}

// Function to find the main post content when on an individual post page
function findMainPostContent() {
    const url = window.location.href;
    
    // Twitter/X.com specific post detection
    if (url.includes('twitter.com') || url.includes('x.com')) {
        console.log('Twitter/X post page detected, waiting for content to fully load...');
        
        // Allow more time for Twitter's dynamic content to load
        setTimeout(() => {
            // Get the main tweet content - the first tweet on a status page is the main tweet
            const tweetText = document.querySelector('article[data-testid="tweet"] [data-testid="tweetText"]');
            if (tweetText) {
                const postElement = tweetText.closest('article');
                const text = tweetText.innerText.trim();
                
                console.log('Found tweet content:', text.substring(0, 50) + '...');
                
                // Add a processing indicator within the tweet
                const processingIndicator = document.createElement('div');
                processingIndicator.className = 'mood-map-processing';
                processingIndicator.textContent = 'Analyzing sentiment...';
                processingIndicator.style.position = 'absolute';
                processingIndicator.style.top = '5px';
                processingIndicator.style.right = '5px';
                processingIndicator.style.backgroundColor = '#1DA1F2';
                processingIndicator.style.color = 'white';
                processingIndicator.style.padding = '3px 6px';
                processingIndicator.style.borderRadius = '3px';
                processingIndicator.style.fontSize = '12px';
                processingIndicator.style.zIndex = '10000';
                processingIndicator.style.boxShadow = '0 1px 3px rgba(0,0,0,0.3)';
                
                if (getComputedStyle(postElement).position === 'static') {
                    postElement.style.position = 'relative';
                }
                postElement.appendChild(processingIndicator);
                
                // Set a backup timeout to remove the indicator if analysis fails
                const processingTimeout = setTimeout(() => {
                    processingIndicator.remove();
                    
                    // Add offline fallback badge
                    const fallbackBadge = document.createElement('div');
                    fallbackBadge.className = 'mood-map-offline-badge';
                    fallbackBadge.textContent = 'Analysis unavailable';
                    fallbackBadge.style.position = 'absolute';
                    fallbackBadge.style.top = '5px';
                    fallbackBadge.style.right = '5px';
                    fallbackBadge.style.backgroundColor = '#aaa';
                    fallbackBadge.style.color = 'white';
                    fallbackBadge.style.padding = '3px 6px';
                    fallbackBadge.style.borderRadius = '3px';
                    fallbackBadge.style.fontSize = '12px';
                    fallbackBadge.style.zIndex = '10000';
                    postElement.appendChild(fallbackBadge);
                    
                    console.log('Sentiment analysis timed out, showing fallback badge');
                }, 10000); // 10 second timeout
                
                // Send message to background script for analysis
                chrome.runtime.sendMessage({ type: 'analyzeSentiment', text: text }, response => {
                    console.log('Received response from background script:', response);
                    clearTimeout(processingTimeout);
                    processingIndicator.remove();
                    
                    if (response && response.sentiment && !response.error) {
                        displaySentiment(postElement, response.sentiment, response.prediction, response.sentiment_percentage);
                        console.log('Analysis successful:', response.sentiment);
                    } else {
                        console.error('Failed to receive sentiment analysis response:', response);
                        
                        // Add detailed error information for debugging
                        let errorMsg = 'Analysis failed';
                        if (response && response.error) {
                            errorMsg += `: ${response.error}`;
                            console.error('Error details:', response.error);
                        } else if (!response) {
                            errorMsg += ': No response';
                            console.error('No response received from background script');
                        }
                        
                        // Add error badge
                        const errorBadge = document.createElement('div');
                        errorBadge.className = 'mood-map-error-badge';
                        errorBadge.textContent = errorMsg;
                        errorBadge.style.position = 'absolute';
                        errorBadge.style.top = '5px';
                        errorBadge.style.right = '5px';
                        errorBadge.style.backgroundColor = '#d32f2f';
                        errorBadge.style.color = 'white';
                        errorBadge.style.padding = '3px 6px';
                        errorBadge.style.borderRadius = '3px';
                        errorBadge.style.fontSize = '12px';
                        errorBadge.style.zIndex = '10000';
                        postElement.appendChild(errorBadge);
                    }
                });
                
                return {
                    text: text,
                    element: postElement
                };
            } else {
                console.log('Tweet text not found, might still be loading');
            }
        }, 2000); // Give Twitter some time to load the content
    }
    // Facebook specific post detection
    else if (url.includes('facebook.com')) {
        const postContent = document.querySelector('.x1iorvi4');
        if (postContent) {
            // Find the container element that wraps the post
            const postContainer = postContent.closest('div[role="article"]') || postContent.parentElement;
            return {
                text: postContent.innerText.trim(),
                element: postContainer || postContent
            };
        }
    }
    // Reddit specific post detection
    else if (url.includes('reddit.com')) {
        const postContent = document.querySelector('.RichTextJSON-root');
        if (postContent) {
            return {
                text: postContent.innerText.trim(),
                element: postContent.closest('.Post') || postContent
            };
        }
        
        // If that doesn't work, try the post title and body
        const postTitle = document.querySelector('h1');
        const postBody = document.querySelector('.RichTextJSON-root, .md');
        
        if (postTitle && postBody) {
            const text = `${postTitle.innerText}\n\n${postBody.innerText}`;
            return {
                text: text.trim(),
                element: postBody.closest('.Post') || postBody
            };
        }
    }
    // LinkedIn specific post detection
    else if (url.includes('linkedin.com')) {
        const postContent = document.querySelector('.feed-shared-update-v2__description');
        if (postContent) {
            return {
                text: postContent.innerText.trim(),
                element: postContent.closest('.feed-shared-update') || postContent
            };
        }
    }
    // Instagram specific post detection
    else if (url.includes('instagram.com')) {
        const postCaption = document.querySelector('._a9zs');
        if (postCaption) {
            return {
                text: postCaption.innerText.trim(),
                element: postCaption.closest('article') || postCaption
            };
        }
    }
    
    return null;
}

// Function to find fallback content if no post is found
function findFallbackContent() {
    // Try to get main heading and first few paragraphs
    const heading = document.querySelector('h1');
    const paragraphs = Array.from(document.querySelectorAll('p')).slice(0, 5);
    if (heading && paragraphs.length > 0) {
        return heading.innerText + '\n\n' +
            paragraphs
                .filter(p => p.innerText.trim().length > 0)
                .map(p => p.innerText)
                .join('\n\n');
    }
    // Last resort: just return the page title
    return document.title;
}

// Function to analyze the sentiment of a post
function analyzeSentiment(postData) {
    if (!postData || !postData.text || postData.text.length < 5) {
        console.log('No valid post data to analyze');
        return;
    }
    
    console.log('Analyzing sentiment for individual post:', postData.text.substring(0, 50) + '...');
    
    // Add a processing indicator
    const processingIndicator = document.createElement('div');
    processingIndicator.className = 'mood-map-processing';
    processingIndicator.textContent = 'Analyzing sentiment...';
    processingIndicator.style.position = 'absolute';
    processingIndicator.style.top = '5px';
    processingIndicator.style.right = '5px';
    processingIndicator.style.backgroundColor = '#f0f0f0';
    processingIndicator.style.color = '#666';
    processingIndicator.style.padding = '3px 6px';
    processingIndicator.style.borderRadius = '3px';
    processingIndicator.style.fontSize = '11px';
    processingIndicator.style.zIndex = '1000';
    
    // Ensure the parent element is positioned
    if (getComputedStyle(postData.element).position === 'static') {
        postData.element.style.position = 'relative';
    }
    postData.element.appendChild(processingIndicator);
    
    // Send message to background script for analysis
    chrome.runtime.sendMessage({ type: 'analyzeSentiment', text: postData.text }, response => {
        // Remove processing indicator
        processingIndicator.remove();
        
        if (response && response.sentiment) {
            displaySentiment(postData.element, response.sentiment, response.prediction, response.sentiment_percentage);
            
            // For longer posts, also get a summary
            if (postData.text.length > 200) {
                const summaryIndicator = document.createElement('div');
                summaryIndicator.className = 'mood-map-summary-processing';
                summaryIndicator.textContent = 'Generating summary...';
                summaryIndicator.style.position = 'absolute';
                summaryIndicator.style.top = '30px';
                summaryIndicator.style.right = '5px';
                summaryIndicator.style.backgroundColor = '#f0f0f0';
                summaryIndicator.style.color = '#666';
                summaryIndicator.style.padding = '3px 6px';
                summaryIndicator.style.borderRadius = '3px';
                summaryIndicator.style.fontSize = '11px';
                summaryIndicator.style.zIndex = '1000';
                postData.element.appendChild(summaryIndicator);
                
                chrome.runtime.sendMessage({ type: 'summarizeText', text: postData.text }, summaryResponse => {
                    summaryIndicator.remove();
                    if (summaryResponse && summaryResponse.summary) {
                        displaySummary(postData.element, summaryResponse.summary);
                    }
                });
            }
        } else {
            console.error('Failed to receive sentiment analysis response:', response);
        }
    });
}

// Function to display sentiment on the post
function displaySentiment(postElement, sentiment, sentimentCategory, sentimentPercentage) {
    // Remove any existing sentiment badge
    const existingBadge = postElement.querySelector('.mood-map-sentiment-badge');
    if (existingBadge) existingBadge.remove();
    
    const sentimentBadge = document.createElement('div');
    sentimentBadge.className = 'mood-map-sentiment-badge';
    
    // Include percentage in the badge text if available
    if (sentimentPercentage !== undefined) {
        sentimentBadge.innerText = `${sentiment} (${sentimentPercentage}%)`;
    } else {
        sentimentBadge.innerText = sentiment;
    }
    
    // Style based on sentiment category (0-2) - UPDATED FROM 5 CATEGORIES TO 3
    const colors = {
        0: '#E53935', // negative - red
        1: '#9E9E9E', // neutral - gray
        2: '#43A047'  // positive - green
    };
    
    sentimentBadge.style.position = 'absolute';
    sentimentBadge.style.top = '5px';
    sentimentBadge.style.right = '5px';
    sentimentBadge.style.backgroundColor = colors[sentimentCategory] || '#9e9e9e';
    sentimentBadge.style.color = 'white';
    sentimentBadge.style.padding = '4px 8px';
    sentimentBadge.style.borderRadius = '4px';
    sentimentBadge.style.fontSize = '12px';
    sentimentBadge.style.fontWeight = 'bold';
    sentimentBadge.style.zIndex = '1000';
    sentimentBadge.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
    
    postElement.appendChild(sentimentBadge);
}

// Function to display summary for longer posts
function displaySummary(postElement, summary) {
    // Remove any existing summary
    const existingSummary = postElement.querySelector('.mood-map-summary');
    if (existingSummary) existingSummary.remove();
    
    const summaryElement = document.createElement('div');
    summaryElement.className = 'mood-map-summary';
    
    // Create a header
    const summaryHeader = document.createElement('div');
    summaryHeader.className = 'mood-map-summary-header';
    summaryHeader.innerText = 'Summary ▶';
    summaryHeader.style.fontWeight = 'bold';
    summaryHeader.style.marginBottom = '4px';
    summaryHeader.style.borderBottom = '1px solid #e0e0e0';
    summaryHeader.style.paddingBottom = '2px';
    
    // Create the summary content
    const summaryContent = document.createElement('div');
    summaryContent.className = 'mood-map-summary-content';
    summaryContent.innerText = summary;
    
    // Style the summary container
    summaryElement.style.backgroundColor = '#f5f5f5';
    summaryElement.style.border = '1px solid #e0e0e0';
    summaryElement.style.borderRadius = '4px';
    summaryElement.style.padding = '8px';
    summaryElement.style.margin = '10px 0';
    summaryElement.style.fontSize = '14px';
    summaryElement.style.color = '#333';
    summaryElement.style.boxShadow = 'inset 0 1px 3px rgba(0,0,0,0.1)';
    
    // Add toggle functionality
    summaryElement.style.cursor = 'pointer';
    summaryContent.style.display = 'none'; // Initially hidden
    
    summaryHeader.onclick = function() {
        if (summaryContent.style.display === 'none') {
            summaryContent.style.display = 'block';
            summaryHeader.innerText = 'Summary ▼';
        } else {
            summaryContent.style.display = 'none';
            summaryHeader.innerText = 'Summary ▶';
        }
    };
    
    // Append elements
    summaryElement.appendChild(summaryHeader);
    summaryElement.appendChild(summaryContent);
    
    // Find a good insertion point - right after the post content
    postElement.appendChild(summaryElement);
}

// Function to check for URL changes (to detect navigation to individual posts)
function checkForUrlChange() {
    const newUrl = window.location.href;
    
    // If URL changed
    if (newUrl !== currentUrl) {
        console.log('URL changed from', currentUrl, 'to', newUrl);
        currentUrl = newUrl;
        
        // If we navigated to an individual post page
        if (isIndividualPostPage(newUrl)) {
            console.log('Detected navigation to individual post page');
            
            // Wait a moment for page content to load
            setTimeout(() => {
                // Find and analyze the main post
                const mainPost = findMainPostContent();
                if (mainPost) {
                    analyzeSentiment(mainPost);
                }
            }, 1000);
        }
    }
}

// Set up URL change detection using both methods for better coverage
// 1. History API monitoring
const pushState = history.pushState;
history.pushState = function() {
    pushState.apply(history, arguments);
    checkForUrlChange();
};

const replaceState = history.replaceState;
replaceState.apply(history, arguments);
    checkForUrlChange();
};

// 2. Regular interval checking as fallback
setInterval(checkForUrlChange, 1000);

// 3. Handle popstate events (back/forward navigation)
window.addEventListener('popstate', () => {
    checkForUrlChange();
});

// Initial check when script loads
if (isIndividualPostPage(currentUrl)) {
    console.log('Currently on individual post page, analyzing...');
    // Wait a moment for page content to load completely
    setTimeout(() => {
        let mainPost = findMainPostContent();
        if (!mainPost) {
            // Use fallback content if no post is found
            const fallbackText = findFallbackContent();
            if (fallbackText && fallbackText.trim().length > 0) {
                mainPost = { text: fallbackText, element: document.body };
            }
        }
        if (mainPost) {
            analyzeSentiment(mainPost);
        }
    }, 1000);
}

// Add dark mode toggle functionality
const toggleDarkMode = () => {
    const root = document.documentElement;
    const isDarkMode = root.style.getPropertyValue('--background-color') === '#121212';

    if (isDarkMode) {
        // Switch to light mode
        root.style.setProperty('--background-color', '#ffffff');
        root.style.setProperty('--text-color', '#000000');
        root.style.setProperty('--border-color', '#cccccc');
    } else {
        // Switch to dark mode
        root.style.setProperty('--background-color', '#121212');
        root.style.setProperty('--text-color', '#e0e0e0');
        root.style.setProperty('--border-color', '#333333');
    }
};

// Add event listener for dark mode toggle button
const darkModeButton = document.getElementById('dark-mode-toggle');
if (darkModeButton) {
    darkModeButton.addEventListener('click', toggleDarkMode);
}

// Add event listener for messages from background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'analyzeSelectedText') {
    // Get selected text and position
    const selection = window.getSelection();
    if (!selection || selection.toString().trim().length === 0) return;
    
    // Get the selection range and coordinates
    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    
    // Create and show overlay with loading state
    const overlay = createSelectionOverlay(rect.left + window.scrollX, 
                                          rect.bottom + window.scrollY);
    
    // Call the backend API to analyze the text
    chrome.runtime.sendMessage(
      { 
        type: 'analyzeSentiment', 
        text: selection.toString().trim() 
      }, 
      response => {
        // Check if we got a response at all
        if (!response) {
          handleAnalysisError(overlay, "No response from server", selection.toString().trim());
          return;
        }
        
        // Check for error in response
        if (response.error) {
          handleAnalysisError(overlay, response.error, selection.toString().trim());
          return;
        }
        
        // Update the overlay with the results
        updateSelectionOverlayWithResults(overlay, response, selection.toString().trim());
      }
    );
  }
});

// Function to handle analysis errors
function handleAnalysisError(overlay, errorMessage, selectedText) {
  const loadingMsg = overlay.querySelector('.mood-map-loading');
  if (loadingMsg) {
    // Replace loading spinner with error message
    loadingMsg.innerHTML = `
      <div style="color: #f44336; margin-bottom: 8px;">Analysis failed</div>
      <div>${errorMessage || 'The server may be busy or unreachable'}</div>
      <button class="mood-map-retry-button">Retry</button>
    `;
    
    // Add retry button functionality
    const retryButton = loadingMsg.querySelector('.mood-map-retry-button');
    if (retryButton) {
      retryButton.addEventListener('click', () => {
        // Show loading spinner again
        loadingMsg.innerHTML = '';
        const spinner = document.createElement('div');
        spinner.className = 'mood-map-loading-spinner';
        const loadingText = document.createElement('div');
        loadingText.textContent = 'Analyzing text...';
        loadingMsg.appendChild(spinner);
        loadingMsg.appendChild(loadingText);
        
        // Call the API again
        chrome.runtime.sendMessage(
          { 
            type: 'analyzeSentiment', 
            text: selectedText
          }, 
          response => {
            if (!response || response.error) {
              handleAnalysisError(overlay, response?.error || "Server unavailable", selectedText);
            } else {
              updateSelectionOverlayWithResults(overlay, response, selectedText);
            }
          }
        );
      });
    }
  }
}

// Function to create the selection analysis overlay - DISABLED VERSION
function createSelectionOverlay(x, y) {
  // This function is now disabled to prevent popup windows
  console.log("Popup analysis window has been disabled");
  return null;
}

// Function to update the overlay with analysis results - DISABLED VERSION
function updateSelectionOverlayWithResults(overlay, results, selectedText) {
  // This function is now disabled to prevent popup windows
  console.log("Popup analysis window has been disabled");
  return;
}

// Mood Map Content Script
// Handles analyzing text content directly on web pages

// Configuration
const MOOD_MAP_CLASS = 'mood-map-analyzed';
const MOOD_MAP_CONTAINER_CLASS = 'mood-map-container';
const BADGE_CLASS = 'mood-map-badge';
const MAX_TEXT_LENGTH = 1000;
const DEBOUNCE_DELAY = 500; // milliseconds
const MAX_RETRY_ATTEMPTS = 3;

// Stylesheet for sentiment badges
const SENTIMENT_STYLES = `
  .${MOOD_MAP_CONTAINER_CLASS} {
    position: relative;
    display: inline-block;
    margin-left: 8px;
    vertical-align: middle;
  }
  
  .${BADGE_CLASS} {
    display: flex;
    align-items: center;
    font-size: 12px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 10px;
    color: white;
    cursor: pointer;
    user-select: none;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  
  .${BADGE_CLASS}.positive {
    background-color: #4caf50;
  }
  
  .${BADGE_CLASS}.neutral {
    background-color: #9e9e9e;
  }
  
  .${BADGE_CLASS}.negative {
    background-color: #f44336;
  }
  
  .${BADGE_CLASS}.loading {
    background-color: #2196f3;
    animation: pulse 2s infinite;
  }
  
  .${BADGE_CLASS}.error {
    background-color: #ff9800;
  }
  
  .${BADGE_CLASS} .icon {
    margin-right: 4px;
    font-size: 13px;
  }
  
  @keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
  }
  
  .mood-map-tooltip {
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    margin-bottom: 8px;
    background: rgba(0, 0, 0, 0.8);
    color: white;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 12px;
    max-width: 250px;
    z-index: 10000;
    display: none;
    text-align: center;
    white-space: normal;
    line-height: 1.4;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
  }
  
  .${BADGE_CLASS}:hover .mood-map-tooltip {
    display: block;
  }
  
  .twitter-badge-container {
    display: inline-flex;
    margin-left: 8px;
    align-items: center;
  }
`;

// Add styles to page
function addStylesIfNeeded() {
  if (!document.getElementById('mood-map-styles')) {
    const styleEl = document.createElement('style');
    styleEl.id = 'mood-map-styles';
    styleEl.textContent = SENTIMENT_STYLES;
    document.head.appendChild(styleEl);
  }
}

// Generate sentiment badge element with click functionality to show detailed analysis
function createSentimentBadge(sentiment = null, score = null, isLoading = false, error = false, fullAnalysisData = null) {
  const container = document.createElement('div');
  container.className = MOOD_MAP_CONTAINER_CLASS;
  
  const badge = document.createElement('div');
  badge.className = BADGE_CLASS;
  
  if (isLoading) {
    badge.className += ' loading';
    badge.innerHTML = '<span class="icon">🔄</span> Analyzing...';
  } 
  else if (error) {
    badge.className += ' error';
    badge.innerHTML = '<span class="icon">⚠️</span> Analysis failed';
    
    // Add tooltip with error message
    const tooltip = document.createElement('div');
    tooltip.className = 'mood-map-tooltip';
    tooltip.textContent = 'Could not connect to sentiment API. Please check your connection.';
    badge.appendChild(tooltip);
  }
  else if (sentiment) {
    badge.className += ` ${sentiment.toLowerCase()}`;
    
    // Get emoji based on sentiment
    let emoji = '😐'; // Neutral default
    if (sentiment.toLowerCase() === 'positive') emoji = '😊';
    else if (sentiment.toLowerCase() === 'negative') emoji = '😞';
    
    badge.innerHTML = `<span class="icon">${emoji}</span> ${sentiment}`;
    
    // Add tooltip with score if available
    if (score !== null) {
      const tooltip = document.createElement('div');
      tooltip.className = 'mood-map-tooltip';
      tooltip.textContent = `Sentiment score: ${score.toFixed(2)}`;
      badge.appendChild(tooltip);
    }
    
    // Store the sentiment data for the click handler
    if (fullAnalysisData) {
      // Store the full analysis data as a data attribute
      badge.dataset.sentimentData = JSON.stringify(fullAnalysisData);
      
      // Add click event listener to show detailed overlay
      badge.addEventListener('click', function(e) {
        e.stopPropagation();
        console.log('Sentiment badge clicked, showing detailed analysis');
        
        // Create display data for the overlay
        const displayData = {
          emoji: emoji,
          color: sentiment.toLowerCase() === 'positive' ? '#4caf50' : 
                 sentiment.toLowerCase() === 'negative' ? '#f44336' : '#9e9e9e',
          sentimentLabel: sentiment
        };
        
        // Find the parent tweet element
        const tweetElement = badge.closest('article') || badge.closest('[data-testid="tweet"]') || 
                            container.parentElement.closest('article');
        
        // Show the detailed overlay
        showDetailedOverlay(tweetElement || container.parentElement, fullAnalysisData, displayData);
      });
      
      // Make it clear this is clickable
      badge.style.cursor = 'pointer';
      badge.title = 'Click for detailed analysis';
    }
  }
  
  container.appendChild(badge);
  return container;
}

// Process Twitter/X text content
function processTwitterContent() {
  // Twitter feed tweets selector
  const tweetSelectors = [
    '[data-testid="tweetText"]',
    // For detailed tweet view
    'article [data-testid="tweetText"]'
  ];

  // Find tweets that haven't been analyzed yet
  tweetSelectors.forEach(selector => {
    const tweets = document.querySelectorAll(`${selector}:not(.${MOOD_MAP_CLASS})`);
    
    tweets.forEach(tweetElement => {
      // Mark as processed to avoid duplicate analysis
      tweetElement.classList.add(MOOD_MAP_CLASS);
      
      // Get the tweet text content
      const tweetText = tweetElement.textContent.trim();
      
      if (tweetText && tweetText.length > 10) { // Minimum length to analyze
        // Find the right place to add the sentiment badge
        const tweetContainer = tweetElement.closest('article') || tweetElement.parentElement;
        
        // First check if there's already a badge
        if (tweetContainer.querySelector(`.${BADGE_CLASS}`)) {
          return;
        }
        
        // Create a loading badge first
        const loadingBadge = createSentimentBadge(null, null, true);
        
        // For Twitter/X, find a good spot to insert the badge
        // First check for the actions (reply/retweet) row
        let actionsRow = tweetContainer.querySelector('[role="group"]');
        
        if (actionsRow) {
          // Create a container for the badge that matches Twitter's style
          const badgeContainer = document.createElement('div');
          badgeContainer.className = 'twitter-badge-container';
          badgeContainer.appendChild(loadingBadge);
          actionsRow.appendChild(badgeContainer);
        } else {
          // If actions row not found, insert after the tweet text
          tweetElement.parentNode.insertBefore(loadingBadge, tweetElement.nextSibling);
        }
        
        // Now analyze the tweet text
        analyzeTweetText(tweetText, loadingBadge);
      }
    });
  });
}

// Function to analyze tweet text with throttling and retry logic
function analyzeTweetText(text, loadingBadgeContainer, attempt = 1) {
  // Trim text to maximum length
  const trimmedText = text.substring(0, MAX_TEXT_LENGTH);
  
  // Define callback to handle the response
  const handleResponse = response => {
    // Handle errors or missing response
    if (!response || chrome.runtime.lastError) {
      console.error('Mood Map analysis failed:', chrome.runtime.lastError);
      
      // Retry logic
      if (attempt < MAX_RETRY_ATTEMPTS) {
        console.log(`Retry attempt ${attempt + 1} for text: ${text.substring(0, 20)}...`);
        setTimeout(() => {
          analyzeTweetText(text, loadingBadgeContainer, attempt + 1);
        }, 1000 * attempt); // Exponential backoff
        return;
      }
      
      // When all retries fail, show error badge
      const errorBadge = createSentimentBadge(null, null, false, true);
      
      if (loadingBadgeContainer && loadingBadgeContainer.parentNode) {
        loadingBadgeContainer.parentNode.replaceChild(errorBadge, loadingBadgeContainer);
      }
      return;
    }
    
    // If response contains error
    if (response.error) {
      console.warn('API error detected, showing error badge:', response.error);
      
      // Show error badge
      const errorBadge = createSentimentBadge(null, null, false, true);
      
      if (loadingBadgeContainer && loadingBadgeContainer.parentNode) {
        loadingBadgeContainer.parentNode.replaceChild(errorBadge, loadingBadgeContainer);
      }
      return;
    }
    
    // Extract sentiment information
    let sentiment, score, category;
    
    if (response.label) {
      sentiment = response.label;
      score = response.score || 0;
      category = response.category !== undefined ? response.category : 1; // Default to neutral if category missing
    } else if (response.sentiment) {
      sentiment = response.sentiment;
      score = response.score || 0;
      category = response.prediction !== undefined ? response.prediction : 1; // Default to neutral if prediction missing
    } else {
      // Default to neutral if no sentiment info found
      sentiment = 'Neutral';
      score = 0;
      category = 1;
    }
    
    // Validate category is within range 0-2
    if (category < 0 || category > 2) {
      category = 1; // Default to neutral for invalid categories
    }
    
    console.log(`Sentiment analysis result: ${sentiment} (category: ${category}, score: ${score})`);
    
    // Add confidence value if not present
    if (!response.confidence && response.score !== undefined) {
      response.confidence = Math.abs(response.score);
    }
    
    // Create the sentiment badge with full analysis data for click handling
    const sentimentBadge = createSentimentBadge(
      sentiment,
      score, 
      false, 
      false, 
      response  // Pass the full response object for the detailed overlay
    );
    
    // Replace loading badge with sentiment badge
    if (loadingBadgeContainer && loadingBadgeContainer.parentNode) {
      loadingBadgeContainer.parentNode.replaceChild(sentimentBadge, loadingBadgeContainer);
    }
  };
  
  // Use our request queue system instead of direct API calls
  queueSentimentRequest(trimmedText, handleResponse);
}

// Request rate limiting configuration
const REQUEST_THROTTLING = {
  enabled: true,
  maxRequestsPerMinute: 30,
  requestQueue: [],
  requestTimestamps: [],
  processingQueue: false
};

// Function to check if we should throttle requests
function shouldThrottleRequest() {
  if (!REQUEST_THROTTLING.enabled) return false;
  
  // Clean old timestamps (older than 1 minute)
  const now = Date.now();
  REQUEST_THROTTLING.requestTimestamps = REQUEST_THROTTLING.requestTimestamps.filter(
    timestamp => now - timestamp < 60000
  );
  
  // If we're under the limit, allow the request
  return REQUEST_THROTTLING.requestTimestamps.length >= REQUEST_THROTTLING.maxRequestsPerMinute;
}

// Function to add a request to the queue
function queueSentimentRequest(text, callback, priority = false) {
  // Add to queue
  const request = { text, callback, timestamp: Date.now() };
  
  if (priority) {
    // Priority requests go to the front of the queue
    REQUEST_THROTTLING.requestQueue.unshift(request);
  } else {
    // Normal requests go to the back
    REQUEST_THROTTLING.requestQueue.push(request);
  }
  
  // Start processing the queue if not already processing
  if (!REQUEST_THROTTLING.processingQueue) {
    processRequestQueue();
  }
}

// Function to process the sentiment analysis request queue
function processRequestQueue() {
  if (REQUEST_THROTTLING.requestQueue.length === 0) {
    REQUEST_THROTTLING.processingQueue = false;
    return;
  }
  
  REQUEST_THROTTLING.processingQueue = true;
  
  // Check if we should throttle
  if (shouldThrottleRequest()) {
    console.log('Throttling sentiment API requests, waiting before processing next request...');
    setTimeout(processRequestQueue, 2000); // Wait 2 seconds before trying again
    return;
  }
  
  // Process next request in queue
  const request = REQUEST_THROTTLING.requestQueue.shift();
  
  // Record this request timestamp
  REQUEST_THROTTLING.requestTimestamps.push(Date.now());
  
  console.log(`Processing queued request (${REQUEST_THROTTLING.requestQueue.length} remaining in queue)`);
  
  // Send the actual request
  chrome.runtime.sendMessage(
    { type: 'analyzeSentiment', text: request.text },
    response => {
      // Call the callback with the response
      request.callback(response);
      
      // Process next request after a small delay to prevent bursts
      setTimeout(processRequestQueue, 500);
    }
  );
}

// Debounce function to limit how often we scan for new content
function debounce(func, wait) {
  let timeout;
  return function() {
    const context = this;
    const args = arguments;
    clearTimeout(timeout);
    timeout = setTimeout(() => {
      func.apply(context, args);
    }, wait);
  };
}

// The main processing function
const processPage = debounce(() => {
  // Only process Twitter for now
  if (window.location.hostname.includes('twitter.com') || 
      window.location.hostname.includes('x.com')) {
    processTwitterContent();
  }
}, DEBOUNCE_DELAY);

// Initial setup
function initialize() {
  addStylesIfNeeded();
  
  // Initial scan of the page
  processPage();
  
  // Watch for content changes to analyze new tweets as they appear
  const observer = new MutationObserver(mutations => {
    processPage();
  });
  
  // Start observing the document body for changes
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
  
  // Listen for messages from background script
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === 'analyzeSelectedText') {
      // This handles right-click context menu "Analyze with Mood Map"
      const text = request.text;
      alert(`Analyzing: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`);
      // Further implementation for popup analysis would go here
    }
  });
  
  console.log('Mood Map content script initialized');
}

// Add CSS to control MoodMap popup overlays - allowing only the emoji click popup
function addOverlayBlockingStyles() {
  const styleEl = document.createElement('style');
  styleEl.id = 'mood-map-overlay-blocker';
  styleEl.textContent = `
    /* Hide unwanted MoodMap popup overlays */
    #mood-map-overlay, 
    .mood-map-overlay,
    div[id*="mood-map"][id*="overlay"]:not(.mood-map-selection-overlay),
    div[class*="mood-map"][class*="overlay"]:not(.mood-map-selection-overlay),
    .mood-map-analysis {
      display: none !important;
      visibility: hidden !important;
      opacity: 0 !important;
      pointer-events: none !important;
      z-index: -9999 !important;
    }
    
    /* Ensure we're allowing the emoji click popup */
    .mood-map-selection-overlay {
      display: block !important;
      visibility: visible !important;
      opacity: 1 !important;
      pointer-events: auto !important;
      z-index: 10000 !important;
    }
    
    /* Ensure we're not hiding the small inline badges */
    .mood-map-badge,
    .mood-map-indicator,
    .${BADGE_CLASS},
    .${MOOD_MAP_CONTAINER_CLASS} {
      display: inline-flex !important;
      visibility: visible !important;
      opacity: 1 !important;
    }
  `;
  document.head.appendChild(styleEl);
  console.log("Added styles to selectively block MoodMap popups");
}

// Call this function early to ensure popups are blocked
addOverlayBlockingStyles();

// Also call it on window load and after a small delay to be extra safe
window.addEventListener('load', () => {
  addOverlayBlockingStyles();
  // Call again after a delay in case content loads dynamically
  setTimeout(addOverlayBlockingStyles, 1000);
  setTimeout(addOverlayBlockingStyles, 3000);
});

// Start the extension
initialize();

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
  
  // Add styles specific to this overlay
  const style = document.createElement('style');
  style.textContent = `
    .mood-map-selection-overlay {
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
    
    .mood-map-selection-overlay.dark-mode {
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
    }
    
    .dark-mode .mood-map-confidence-text {
      color: #aaa;
    }
    
    .mood-map-selection-text {
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid #eee;
      font-size: 13px;
      line-height: 1.5;
      max-height: 100px;
      overflow-y: auto;
    }
    
    .dark-mode .mood-map-selection-text {
      border-top-color: #444;
    }
  `;
  
  // Assemble the overlay
  overlay.appendChild(style);
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