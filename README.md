# mood-map
Perfect — let’s go all in! 🔥  
Here’s the full setup including **Tech Stack** and **How It Works** sections that would fit beautifully into your README:

---

# Mood Map

**Mood Map** is a Chrome extension that analyzes the sentiment of social media posts in real-time using a custom-trained AI model. It features a Flask-powered backend that handles data processing and inference, ensuring fast, reliable, and secure sentiment prediction.

The extension dynamically captures post content from supported platforms, securely transmits it to the backend via RESTful API endpoints, and seamlessly overlays the sentiment feedback (e.g., positive, neutral, negative) onto the user’s feed. Mood Map is designed with scalability, low latency, and user privacy in mind.


## 🚀 Tech Stack

- **Frontend:**  
  - Chrome Extension (HTML, JavaScript, CSS)  
  - Content Scripts for DOM interaction
  - Popup Interface for quick controls and settings

- **Backend:**  
  - Flask (Python) for API development
    
- **Machine Learning Model:**  
  - Custom-trained NLP model  (BERT)  
  - Scikit-learn

- **Deployment & Tools:**  
  - RESTful APIs for communication between extension and server  
  - JSON for data serialization  
  - Docker for containerization  
  - Git for version control


## ⚙️ How It Works

1. **User Browses Social Media:**  
   The Chrome extension passively listens to the web page, capturing visible post content in real-time via DOM scraping techniques.

2. **Content Extraction:**  
   Extracted text data is preprocessed and packaged into a secure POST request.

3. **API Call to Backend:**  
   The extension sends the post content to the Flask backend through a REST API endpoint (`/analyze`).

4. **Sentiment Prediction:**  
   The Flask server feeds the text into the AI model, which returns a sentiment label (e.g., positive, neutral, negative) along with a confidence score in terms of percentage and we graphically represent it.

5. **Display Sentiment:**  
   The Chrome extension receives the response and dynamically injects a sentiment badge or color-coded marker directly onto the corresponding social media post.

6. **Continuous Updates:**  
   As the user scrolls, the extension keeps monitoring new content, ensuring real-time sentiment feedback without reloading the page.
