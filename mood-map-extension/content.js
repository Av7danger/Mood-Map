console.log("content.js loaded");

// Function to handle content extraction and communication with the backend
function analyzeSentiment() {
  console.log("Analyzing sentiment of social media posts...");

  // Example: Extract visible text content from posts
  const posts = document.querySelectorAll(".post-content, .tweet, .post-text");
  const postContents = Array.from(posts).map(post => post.innerText.trim());

  // Send extracted content to the backend for sentiment analysis
  postContents.forEach(content => {
    if (content) {
      fetch("http://localhost:5000/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: content })
      })
        .then(response => response.json())
        .then(data => {
          console.log("Sentiment analysis result:", data);
          // Example: Add sentiment badge to the post
          const sentimentBadge = document.createElement("span");
          sentimentBadge.textContent = `${data.sentiment} (${data.confidence}%)`;
          sentimentBadge.style.color = data.sentiment === "positive" ? "green" : data.sentiment === "negative" ? "red" : "gray";
          post.appendChild(sentimentBadge);
        })
        .catch(error => console.error("Error analyzing sentiment:", error));
    }
  });
}

// Run the sentiment analysis function when the page loads
window.addEventListener("load", analyzeSentiment);