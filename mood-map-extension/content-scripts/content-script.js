console.log("content-script.js loaded");

let iconMap = {};

function scrapeProductDetails() {
  const hostname = window.location.hostname;

  switch (hostname) {
    case "www.bigbasket.com":
      return {
        name:
          document.querySelector("h1")?.innerText.trim() ||
          "Product name not found in h1 tag.",
      };

    case "www.zeptonow.com":
      return {
        name:
          document.querySelector("h1")?.innerText.trim() ||
          "Product name not found in h1 tag.",
      };

    case "www.swiggy.com":
      return {
        name:
          document.querySelector("h1")?.innerText.trim() ||
          "Product name not found in h1 tag.",
      };

    case "www.jiomart.com":
      return {
        name:
          document.querySelector(".product-header-name")?.innerText.trim() ||
          "Product name not found in h1 tag.",
      };

    case "www.amazon.in":
      return {
        name:
          document.querySelector("#productTitle")?.innerText.trim() ||
          "Product name not found in h1 tag.",
      };

    case "www.flipkart.com":
      return {
        name:
          document.querySelector(".VU-ZEz").innerText ||
          "Product name not found in .VU-ZEz class.",
      };

    case "blinkit.com":
      return {
        name:
          document.querySelector(".tw-text-600").innerText ||
          "Product name not found in .tw-text-600 class.",
      };

    default:
      // If the hostname doesn't match any cases, return a default value
      return {
        name: "No product name found.",
      };
  }
}

