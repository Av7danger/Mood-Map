document.addEventListener('DOMContentLoaded', () => {
    const sentimentScore = document.getElementById('sentiment-score');
    const sentimentLabel = document.getElementById('sentiment-label');
    const summaryText = document.getElementById('summary-text');
    const sentimentCircle = document.querySelector('.sentiment-circle');
    const summaryContainer = document.getElementById('summary-container');
    const analyzeButton = document.getElementById('analyze-button');
    const inputText = document.getElementById('input-text');
    
    // First check backend connectivity
    checkBackendConnectivity();
    
    // Check if backend is available
    function checkBackendConnectivity(retryCount = 0) {
        // Apply analyzing state initially
        sentimentScore.textContent = '...';
        sentimentLabel.textContent = 'Connecting...';
        sentimentLabel.classList.add('analyzing');
        sentimentCircle.classList.add('analyzing');

        chrome.runtime.sendMessage({ type: 'checkBackend' }, response => {
            if (response && response.isAvailable) {
                analyzePageContent();
            } else {
                if (retryCount < 3) {
                    // Retry after 1 second, up to 3 times
                    setTimeout(() => checkBackendConnectivity(retryCount + 1), 1000);
                } else {
                    displayError("Backend server not available. Make sure it's running on 127.0.0.1:5000");
                }
            }
        });
    }
    
    // Get the current active tab and analyze its content
    function analyzePageContent() {
        chrome.tabs.query({active: true, currentWindow: true}, tabs => {
            const activeTab = tabs[0];
            
            // Get the tab content
            chrome.scripting.executeScript({
                target: {tabId: activeTab.id},
                function: getPageContent
            }, results => {
                if (!results || chrome.runtime.lastError) {
                    displayError("Could not access page content");
                    return;
                }
                
                const pageContent = results[0]?.result;
                if (pageContent && pageContent.trim().length > 0) {
                    // Send the content for analysis
                    chrome.runtime.sendMessage({ 
                        type: 'analyzeSentiment', 
                        text: pageContent 
                    }, response => {
                        // Remove analyzing state
                        sentimentLabel.classList.remove('analyzing');
                        sentimentCircle.classList.remove('analyzing');
                        
                        if (response && response.sentiment) {
                            updateSentimentDisplay(response);
                            
                            // After sentiment analysis, request a summary with sentiment context
                            if (pageContent.length > 200) {
                                chrome.runtime.sendMessage({ 
                                    type: 'summarizeText', 
                                    text: pageContent,
                                    sentiment_category: response.prediction,
                                    sentiment_label: response.sentiment
                                }, summaryResponse => {
                                    console.log('[Debug] summaryResponse (pageContent):', summaryResponse); // Debug log
                                    if (summaryResponse && summaryResponse.summary) {
                                        updateSummaryWithSentimentReasoning(summaryResponse.summary, response.prediction);
                                    }
                                });
                            }
                        } else if (response && response.isBackendError) {
                            displayError("Backend server not available. Make sure it's running on 127.0.0.1:5000");
                        } else {
                            displayError("Error analyzing content");
                        }
                    });
                } else {
                    // If no analyzable content found, show a clear message
                    displayError("No analyzable content found on this page. Try entering text manually below or visit a supported social media post.");
                }
            });
        });
    }
    
    // Handle custom text analysis
    analyzeButton.addEventListener('click', () => {
        const text = inputText.value.trim();
        
        if (!text) {
            alert("Please enter some text to analyze");
            return;
        }
        
        // Show analyzing state
        sentimentScore.textContent = '...';
        sentimentLabel.textContent = 'Analyzing...';
        sentimentLabel.classList.add('analyzing');
        sentimentCircle.classList.add('analyzing');
        
        // Remove previous sentiment classes
        sentimentLabel.classList.remove(
            'sentiment-overwhelmingly-negative',
            'sentiment-negative',
            'sentiment-neutral',
            'sentiment-positive',
            'sentiment-overwhelmingly-positive'
        );
        sentimentCircle.classList.remove(
            'sentiment-circle-overwhelmingly-negative',
            'sentiment-circle-negative',
            'sentiment-circle-neutral',
            'sentiment-circle-positive',
            'sentiment-circle-overwhelmingly-positive'
        );
        
        chrome.runtime.sendMessage({ 
            type: 'analyzeSentiment', 
            text: text 
        }, response => {
            // Remove analyzing state
            sentimentLabel.classList.remove('analyzing');
            sentimentCircle.classList.remove('analyzing');
            
            // Clear the input text box after analysis
            inputText.value = '';
            
            if (response && response.sentiment) {
                updateSentimentDisplay(response);
                
                // Also get summary with sentiment reasoning
                if (text.length > 100) {
                    chrome.runtime.sendMessage({ 
                        type: 'summarizeText', 
                        text: text,
                        sentiment_category: response.prediction,
                        sentiment_label: response.sentiment
                    }, summaryResponse => {
                        console.log('[Debug] summaryResponse (custom text):', summaryResponse); // Debug log
                        if (summaryResponse && summaryResponse.summary) {
                            updateSummaryWithSentimentReasoning(summaryResponse.summary, response.prediction);
                        }
                    });
                }
            } else if (response && response.isBackendError) {
                displayError("Backend server not available. Make sure it's running on 127.0.0.1:5000");
            } else {
                displayError("Error analyzing text");
            }
        });
    });
    
    // Function to display sentiment with appropriate colors and styles
    function updateSentimentDisplay(response) {
        // Use backend-provided percentage and label directly
        let sentimentType;
        switch (response.prediction) {
            case 0:
                sentimentType = 'overwhelmingly-negative';
                break;
            case 1:
                sentimentType = 'negative';
                break;
            case 2:
                sentimentType = 'neutral';
                break;
            case 3:
                sentimentType = 'positive';
                break;
            case 4:
                sentimentType = 'overwhelmingly-positive';
                break;
            default:
                sentimentType = 'neutral';
        }

        // Use backend's percentage and label
        sentimentScore.textContent = (response.sentiment_percentage !== undefined)
            ? `${response.sentiment_percentage}%`
            : '...';
        sentimentLabel.textContent = response.sentiment || 'Unknown';

        // Remove any existing sentiment classes
        sentimentLabel.classList.remove(
            'sentiment-overwhelmingly-negative',
            'sentiment-negative',
            'sentiment-neutral',
            'sentiment-positive',
            'sentiment-overwhelmingly-positive'
        );
        
        sentimentCircle.classList.remove(
            'sentiment-circle-overwhelmingly-negative',
            'sentiment-circle-negative',
            'sentiment-circle-neutral',
            'sentiment-circle-positive',
            'sentiment-circle-overwhelmingly-positive'
        );
        
        // Apply appropriate classes for styling
        sentimentLabel.classList.add(`sentiment-${sentimentType}`);
        sentimentCircle.classList.add(`sentiment-circle-${sentimentType}`);
        
        // Update summary if available and apply matching sentiment styling
        if (response.summary) {
            summaryText.textContent = response.summary;
            summaryContainer.style.display = 'block';
            
            // Remove any existing summary text classes
            summaryText.classList.remove(
                'summary-text-overwhelmingly-negative',
                'summary-text-negative',
                'summary-text-neutral',
                'summary-text-positive',
                'summary-text-overwhelmingly-positive'
            );
            
            // Add appropriate summary styling class
            summaryText.classList.add(`summary-text-${sentimentType}`);
            
            // Add padding to summary text for better appearance with the border
            summaryText.style.padding = '8px 12px';
        } else {
            // Try to get summary for longer content
            if (response.text && response.text.length > 200) {
                chrome.runtime.sendMessage({ 
                    type: 'summarizeText', 
                    text: response.text 
                }, summaryResponse => {
                    if (summaryResponse && summaryResponse.summary) {
                        summaryText.textContent = summaryResponse.summary;
                        summaryContainer.style.display = 'block';
                        summaryText.classList.add(`summary-text-${sentimentType}`);
                        summaryText.style.padding = '8px 12px';
                    } else {
                        summaryContainer.style.display = 'none';
                    }
                });
            } else {
                summaryContainer.style.display = 'none';
            }
        }
    }
    
    // Function to update summary with sentiment reasoning
    function updateSummaryWithSentimentReasoning(summary, sentimentCategory) {
        console.log('[Debug] updateSummaryWithSentimentReasoning called with:', { summary, sentimentCategory }); // Debug log
        // Make sure summary container is visible
        summaryContainer.style.display = 'block';
        
        // Update summary text
        summaryText.textContent = summary;
        
        // Determine sentiment type for styling
        let sentimentType;
        switch (sentimentCategory) {
            case 0:
                sentimentType = 'overwhelmingly-negative';
                break;
            case 1:
                sentimentType = 'negative';
                break;
            case 2:
                sentimentType = 'neutral';
                break;
            case 3:
                sentimentType = 'positive';
                break;
            case 4:
                sentimentType = 'overwhelmingly-positive';
                break;
            default:
                sentimentType = 'neutral';
        }
        
        // Remove any existing summary text classes
        summaryText.classList.remove(
            'summary-text-overwhelmingly-negative',
            'summary-text-negative',
            'summary-text-neutral',
            'summary-text-positive',
            'summary-text-overwhelmingly-positive'
        );
        
        // Add appropriate summary styling class
        summaryText.classList.add(`summary-text-${sentimentType}`);
        summaryText.style.padding = '8px 12px';
    }
    
    function displayError(message) {
        sentimentLabel.classList.remove('analyzing');
        sentimentCircle.classList.remove('analyzing');
        
        sentimentScore.textContent = '!';
        sentimentLabel.textContent = message;
        sentimentLabel.style.fontSize = '14px';
        
        // Remove existing classes and add neutral
        sentimentLabel.classList.remove(
            'sentiment-overwhelmingly-negative',
            'sentiment-negative',
            'sentiment-positive',
            'sentiment-overwhelmingly-positive'
        );
        sentimentCircle.classList.remove(
            'sentiment-circle-overwhelmingly-negative',
            'sentiment-circle-negative',
            'sentiment-circle-positive',
            'sentiment-circle-overwhelmingly-positive'
        );
        
        sentimentLabel.classList.add('sentiment-neutral');
        sentimentCircle.classList.add('sentiment-circle-neutral');
        
        // Show error in summary section
        summaryText.textContent = "Could not analyze content. Try entering text manually below.";
        summaryContainer.style.display = 'block';
    }
    
    // Debug controls
    const testBackendBtn = document.getElementById('test-backend-btn');
    const reloadExtensionBtn = document.getElementById('reload-extension-btn');
    const debugOutput = document.getElementById('debug-output');
    
    // Test backend connection directly
    testBackendBtn.addEventListener('click', async () => {
        debugLog('Testing direct connection to backend...');
        try {
            const response = await fetch('http://127.0.0.1:5000/', {
                method: 'GET',
                headers: { 'X-Debug': 'true' }
            });
            
            if (response.ok) {
                const text = await response.text();
                debugLog(`✅ Backend connection successful: ${text}`);
                
                // Try a simple sentiment analysis test
                debugLog('Testing sentiment analysis endpoint...');
                const analyzeResponse = await fetch('http://127.0.0.1:5000/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: 'This is a test message from the extension.' })
                });
                
                if (analyzeResponse.ok) {
                    const result = await analyzeResponse.json();
                    debugLog(`✅ Sentiment analysis successful: ${result.sentiment}`);
                    
                    // Update UI with the test result
                    updateSentimentDisplay(result);
                } else {
                    debugLog(`❌ Sentiment analysis failed: ${analyzeResponse.status} ${analyzeResponse.statusText}`);
                }
            } else {
                debugLog(`❌ Backend connection failed: ${response.status} ${response.statusText}`);
            }
        } catch (error) {
            debugLog(`❌ Connection error: ${error.message}`);
        }
    });
    
    // Reload extension - useful for testing
    reloadExtensionBtn.addEventListener('click', () => {
        debugLog('Requesting extension reload...');
        chrome.runtime.reload();
    });
    
    // Helper function to log debug messages
    function debugLog(message) {
        const timestamp = new Date().toLocaleTimeString();
        debugOutput.innerHTML += `<div>[${timestamp}] ${message}</div>`;
        debugOutput.scrollTop = debugOutput.scrollHeight;
        console.log(`[Debug] ${message}`);
    }
});

// Function to get page content - will be injected into the active tab
function getPageContent() {
    // Determine which platform we're on based on URL
    const url = window.location.href;
    let mainPostContent = null;
    
    // Twitter/X.com specific post detection
    if (url.includes('twitter.com') || url.includes('x.com')) {
        // Check if we're on a specific tweet page
        if (url.match(/twitter\.com\/\w+\/status\/\d+/) || url.match(/x\.com\/\w+\/status\/\d+/)) {
            // Get the main tweet content - the first tweet on a status page is the main tweet
            const tweetText = document.querySelector('article[data-testid="tweet"] [data-testid="tweetText"]');
            if (tweetText) {
                mainPostContent = tweetText.innerText;
            }
        }
    }
    // Facebook specific post detection
    else if (url.includes('facebook.com')) {
        // Check if this is a specific post page
        if (url.includes('/posts/') || url.includes('/permalink/')) {
            // For Facebook posts
            const postContent = document.querySelector('.x1iorvi4');
            if (postContent) {
                mainPostContent = postContent.innerText;
            }
        }
    }
    // Reddit specific post detection
    else if (url.includes('reddit.com')) {
        // Check if we're on a specific post page
        if (url.includes('/comments/')) {
            // For Reddit posts
            const postContent = document.querySelector('.RichTextJSON-root');
            if (postContent) {
                mainPostContent = postContent.innerText;
            }
            
            // If that doesn't work, try the post title and body
            if (!mainPostContent) {
                const postTitle = document.querySelector('h1');
                const postBody = document.querySelector('.RichTextJSON-root, .md');
                if (postTitle && postBody) {
                    mainPostContent = `${postTitle.innerText}\n\n${postBody.innerText}`;
                }
            }
        }
    }
    // LinkedIn specific post detection
    else if (url.includes('linkedin.com')) {
        // Check if we're on a specific post page
        if (url.includes('/posts/')) {
            // For LinkedIn posts
            const postContent = document.querySelector('.feed-shared-update-v2__description');
            if (postContent) {
                mainPostContent = postContent.innerText;
            }
        }
    }
    // Instagram specific post detection
    else if (url.includes('instagram.com')) {
        // Check if we're on a specific post page
        if (url.includes('/p/')) {
            // For Instagram posts
            const postCaption = document.querySelector('._a9zs');
            if (postCaption) {
                mainPostContent = postCaption.innerText;
            }
        }
    }
    
    // If we detected a specific post, return it
    if (mainPostContent && mainPostContent.trim().length > 0) {
        return mainPostContent;
    }
    
    // If not a specific post page, or couldn't find main post content,
    // try to identify the main article content
    
    // Look for article elements which typically contain the main content
    const articleSelectors = [
        'article',
        '[role="article"]',
        '.article-content',
        '.post-content',
        '.entry-content',
        '.main-content',
        '#article-body',
        '.story-body'
    ];
    
    for (const selector of articleSelectors) {
        const articleElements = document.querySelectorAll(selector);
        if (articleElements.length > 0) {
            // We'll choose the largest article element as it's likely the main content
            let largestArticle = null;
            let maxLength = 0;
            
            articleElements.forEach(article => {
                if (article.innerText.length > maxLength) {
                    maxLength = article.innerText.length;
                    largestArticle = article;
                }
            });
            
            if (largestArticle && maxLength > 100) {
                return largestArticle.innerText;
            }
        }
    }
    
    // If all else fails, look for the page's main heading + first few paragraphs
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
