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
        // Get the main tweet content - the first tweet on a status page is the main tweet
        const tweetText = document.querySelector('article[data-testid="tweet"] [data-testid="tweetText"]');
        if (tweetText) {
            return {
                text: tweetText.innerText.trim(),
                element: tweetText.closest('article')
            };
        }
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
    
    // Style based on sentiment category (0-4)
    const colors = {
        0: '#B71C1C', // overwhelmingly negative - dark red
        1: '#E53935', // negative - red
        2: '#9E9E9E', // neutral - gray
        3: '#43A047', // positive - green
        4: '#1B5E20'  // overwhelmingly positive - dark green
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
history.replaceState = function() {
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
        // Update the overlay with the results
        updateSelectionOverlayWithResults(overlay, response, selection.toString().trim());
      }
    );
  }
});

// Function to create the selection analysis overlay
function createSelectionOverlay(x, y) {
  // Remove any existing overlay
  const existingOverlay = document.querySelector('.mood-map-selection-overlay');
  if (existingOverlay) existingOverlay.remove();
  
  // Create overlay element
  const overlay = document.createElement('div');
  overlay.className = 'mood-map-selection-overlay';
  
  // Check if dark mode is preferred
  const prefersDarkMode = window.matchMedia && 
                          window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (prefersDarkMode) {
    overlay.classList.add('dark-mode');
  }
  
  // Adjust position to ensure visibility
  overlay.style.left = `${Math.max(5, Math.min(x, window.innerWidth - 330))}px`;
  
  // Position overlay below selection
  // If it would go below viewport bottom, position it above the selection instead
  if (y + 220 > window.innerHeight + window.scrollY) {
    const selectionHeight = 20; // Approximate height of selected text
    overlay.style.top = `${y - selectionHeight - 240}px`;
  } else {
    overlay.style.top = `${y + 10}px`;
  }
  
  // Add header
  const header = document.createElement('div');
  header.className = 'mood-map-overlay-header';
  
  const title = document.createElement('div');
  title.className = 'mood-map-overlay-title';
  title.textContent = 'Mood Map Analysis';
  
  const closeBtn = document.createElement('div');
  closeBtn.className = 'mood-map-overlay-close';
  closeBtn.textContent = '×';
  closeBtn.addEventListener('click', () => {
    overlay.style.opacity = '0';
    overlay.style.transform = 'translateY(10px)';
    setTimeout(() => overlay.remove(), 200);
  });
  
  header.appendChild(title);
  header.appendChild(closeBtn);
  
  // Add loading message with spinner
  const loadingMsg = document.createElement('div');
  loadingMsg.className = 'mood-map-loading';
  
  const spinner = document.createElement('div');
  spinner.className = 'mood-map-loading-spinner';
  
  const loadingText = document.createElement('div');
  loadingText.textContent = 'Analyzing text...';
  
  loadingMsg.appendChild(spinner);
  loadingMsg.appendChild(loadingText);
  
  // Append to overlay
  overlay.appendChild(header);
  overlay.appendChild(loadingMsg);
  
  // Add to page
  document.body.appendChild(overlay);
  
  return overlay;
}

// Function to update the overlay with analysis results
function updateSelectionOverlayWithResults(overlay, results, selectedText) {
  // Remove loading message
  const loadingMsg = overlay.querySelector('.mood-map-loading');
  if (loadingMsg) loadingMsg.remove();
  
  // Create results container
  const resultsContainer = document.createElement('div');
  resultsContainer.className = 'mood-map-selection-result';
  
  // Handle error case
  if (results.error) {
    const errorMsg = document.createElement('div');
    errorMsg.textContent = `Error: ${results.error}`;
    errorMsg.style.color = 'red';
    resultsContainer.appendChild(errorMsg);
    overlay.appendChild(resultsContainer);
    return;
  }
  
  // Add sentiment information
  const sentimentRow = document.createElement('div');
  sentimentRow.className = 'mood-map-sentiment-row';
  
  const sentimentLabel = document.createElement('div');
  sentimentLabel.className = 'mood-map-sentiment-label';
  sentimentLabel.textContent = 'Sentiment:';
  
  const sentimentValue = document.createElement('div');
  sentimentValue.className = 'mood-map-sentiment-value';
  sentimentValue.textContent = results.sentiment;
  
  // Add color class based on sentiment category
  if (results.prediction >= 3) {
    sentimentValue.classList.add('positive');
  } else if (results.prediction <= 1) {
    sentimentValue.classList.add('negative');
  } else {
    sentimentValue.classList.add('neutral');
  }
  
  sentimentRow.appendChild(sentimentLabel);
  sentimentRow.appendChild(sentimentValue);
  resultsContainer.appendChild(sentimentRow);
  
  // Add confidence/percentage if available
  if (results.sentiment_percentage !== undefined) {
    // Create confidence row
    const confidenceRow = document.createElement('div');
    confidenceRow.className = 'mood-map-sentiment-row';
    
    const confidenceLabel = document.createElement('div');
    confidenceLabel.className = 'mood-map-sentiment-label';
    confidenceLabel.textContent = 'Confidence:';
    
    confidenceRow.appendChild(confidenceLabel);
    resultsContainer.appendChild(confidenceRow);
    
    // Create confidence meter
    const confidenceMeter = document.createElement('div');
    confidenceMeter.className = 'mood-map-confidence-meter';
    
    const confidenceValue = document.createElement('div');
    confidenceValue.className = 'mood-map-confidence-value';
    confidenceValue.style.width = `${results.sentiment_percentage}%`;
    
    // Set color based on sentiment
    if (results.prediction >= 3) {
      confidenceValue.style.backgroundColor = '#4caf50';
    } else if (results.prediction <= 1) {
      confidenceValue.style.backgroundColor = '#f44336';
    } else {
      confidenceValue.style.backgroundColor = '#2196F3';
    }
    
    confidenceMeter.appendChild(confidenceValue);
    resultsContainer.appendChild(confidenceMeter);
    
    // Add text percentage
    const confidenceText = document.createElement('div');
    confidenceText.className = 'mood-map-confidence-text';
    confidenceText.textContent = `${results.sentiment_percentage}% confidence`;
    resultsContainer.appendChild(confidenceText);
  }
  
  // Add the selected text
  if (selectedText && selectedText.length > 0) {
    const textContainer = document.createElement('div');
    textContainer.className = 'mood-map-selection-text';
    textContainer.textContent = selectedText.length > 200 
      ? selectedText.substring(0, 197) + '...' 
      : selectedText;
    resultsContainer.appendChild(textContainer);
  }
  
  // Add the results to the overlay
  overlay.appendChild(resultsContainer);
  
  // Add keyboard shortcut to close
  const handleKeyDown = (e) => {
    if (e.key === 'Escape') {
      overlay.style.opacity = '0';
      overlay.style.transform = 'translateY(10px)';
      setTimeout(() => {
        overlay.remove();
        document.removeEventListener('keydown', handleKeyDown);
      }, 200);
    }
  };
  
  document.addEventListener('keydown', handleKeyDown);
}