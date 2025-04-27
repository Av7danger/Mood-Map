console.log("content-script.js loaded");

function scrapePostDetails() {
  const hostname = window.location.hostname;

  // Mapping of hostnames to their respective query selectors
  const siteSelectors = {
    "facebook.com": "h1",
    "twitter.com": "h1",
    "instagram.com": "h1",
    "linkedin.com": ".post-header-name",
    "reddit.com": "#postTitle",
    "tiktok.com": ".VU-ZEz",
    "pinterest.com": ".tw-text-600",
  };

  // Find the appropriate selector for the current hostname
  const selector = Object.keys(siteSelectors).find((key) =>
    hostname.includes(key)
  );

  if (selector) {
    const postContent =
      document.querySelector(siteSelectors[selector])?.innerText.trim() ||
      "Post content not found.";
    return { name: postContent, link: `https://${selector}` };
  } else {
    console.warn(`No selector found for hostname: ${hostname}`);
    return { name: "No post content found.", link: "" };
  }
}