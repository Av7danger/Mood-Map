// Background script for Mood Map Extension

// Define backend API URL
const BACKEND_URL = 'http://127.0.0.1:5000';

// Listen for messages from the popup or content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('Background script received message:', message);
    
    // Handle different message types
    switch (message.type) {
        case 'analyzeSentiment':
            analyzeSentiment(message.text, sendResponse);
            break;
        case 'summarizeText':
            summarizeText(message.text, message.sentiment_category, message.sentiment_label, sendResponse);
            break;
        case 'checkBackend':
            checkBackendConnectivity(sendResponse);
            break;
        default:
            console.log('Unknown message type:', message.type);
            sendResponse({ error: 'Unknown message type' });
    }
    
    // Return true to indicate we'll send a response asynchronously
    return true;
});

// Check if backend is available
async function checkBackendConnectivity(sendResponse) {
    try {
        const response = await fetch(`${BACKEND_URL}/`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (response.ok) {
            console.log('Backend is available');
            sendResponse({ isAvailable: true });
        } else {
            console.error('Backend returned error:', response.status);
            sendResponse({ isAvailable: false, error: `HTTP Error: ${response.status}` });
        }
    } catch (error) {
        console.error('Error connecting to backend:', error);
        sendResponse({ isAvailable: false, error: error.message });
    }
}

// Send text to backend API for sentiment analysis
async function analyzeSentiment(text, sendResponse) {
    if (!text || text.trim().length === 0) {
        sendResponse({ error: 'Empty text provided' });
        return;
    }
    
    try {
        const response = await fetch(`${BACKEND_URL}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        
        if (response.ok) {
            const result = await response.json();
            sendResponse(result);
        } else {
            console.error('API error:', response.status);
            sendResponse({ error: `API error: ${response.status}`, isBackendError: true });
        }
    } catch (error) {
        console.error('Error calling API:', error);
        sendResponse({ error: error.message, isBackendError: true });
    }
}

// Send text to backend API for summarization
async function summarizeText(text, sentiment_category, sentiment_label, sendResponse) {
    if (!text || text.trim().length === 0) {
        sendResponse({ error: 'Empty text provided' });
        return;
    }
    
    try {
        const requestData = { text };
        
        // Include sentiment information if available
        if (sentiment_category !== undefined) {
            requestData.sentiment_category = sentiment_category;
        }
        
        if (sentiment_label) {
            requestData.sentiment_label = sentiment_label;
        }
        
        const response = await fetch(`${BACKEND_URL}/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        
        if (response.ok) {
            const result = await response.json();
            sendResponse(result);
        } else {
            console.error('Summarization API error:', response.status);
            sendResponse({ error: `API error: ${response.status}`, isBackendError: true });
        }
    } catch (error) {
        console.error('Error calling summarization API:', error);
        sendResponse({ error: error.message, isBackendError: true });
    }
}