# test.ipynb Documentation

## Overview
This notebook demonstrates an automated sports article generation system that scrapes news headlines from a website and uses OpenAI's GPT-4o-mini model to generate a humorous sports article based on those headlines.

## Purpose
The notebook creates a pipeline that:
1. Fetches links and titles from a news website (AP News sports section)
2. Prepares the extracted data for AI processing
3. Uses OpenAI's chat API to generate a funny, entertaining article by a fictional sports journalist
4. Displays the generated article in a formatted markdown output

---

## Code Breakdown

### Step 1: Setup & Imports
**Cell 1 - Library Imports**
- `BeautifulSoup`: HTML parsing library to extract content from web pages
- `requests`: HTTP library to fetch web pages
- `OpenAI`: Official OpenAI Python client for accessing GPT models
- `dotenv`: Loads environment variables (including OpenAI API key) from `.env` file
- `json`: JSON serialization library
- `IPython.display`: Tools to render formatted output in Jupyter notebooks

**Cell 2 - HTTP Headers**
- Defines a User-Agent header that identifies the request as coming from a standard web browser
- Prevents websites from blocking the request as a bot

### Step 2: Web Scraping Function
**Cell 3 - `parse_url_links(url)` Function**
Purpose: Extract all links and their titles from a website

Steps:
1. Sends an HTTP GET request to the provided URL with the User-Agent header
2. Parses the HTML response using BeautifulSoup
3. Finds the `<main>` section of the page
4. Extracts all links (`<a>` tags) that have both a href and visible text
5. Creates a list of dictionaries containing:
   - `url`: The link's href attribute
   - `title`: The link's visible text
6. Returns the list as a JSON string

**Cell 4 - Testing Web Scraping**
- Tests `parse_url_links()` on the AP News sports page
- Prints the extracted links and titles in JSON format

### Step 3: AI Prompt Setup
**Cell 5 - System Prompt**
- Defines the character: An experienced, funny sports journalist
- Sets the tone for the AI to be humorous
- This system prompt guides how the AI should respond

**Cell 5 (continued) - `parse_links_user_prompt(url)` Function**
Purpose: Create the user prompt for the AI

Steps:
1. Initializes a user prompt instructing the AI to write a funny article
2. Calls `parse_url_links(url)` to get the scraped links
3. Concatenates the extracted titles to the prompt
4. Returns the complete user prompt

**Cell 6 - Testing User Prompt**
- Displays the generated prompt to verify it contains the scraped data

### Step 4: AI Article Generation
**Cell 7 - `the_article(url)` Function**
Purpose: Generate the funny article using OpenAI's API

Steps:
1. Calls OpenAI's `chat.completions.create()` method with:
   - `model`: Uses "gpt-4o-mini" (efficient, cost-effective model)
   - `messages`: Array containing:
     - System message: The journalist character prompt
     - User message: The prompt with scraped article titles
2. Extracts the generated text from the response
3. Returns the complete article as a string

**Cell 8 - Final Execution & Display**
1. Calls `the_article()` on the AP News sports page URL
2. Uses `display(Markdown())` to render the article with proper formatting
3. The article is displayed as readable Markdown in the notebook

---

## Data Flow Diagram
```
Website (apnews.com/sports)
         ↓
    parse_url_links()
    (BeautifulSoup scraping)
         ↓
    JSON list of [url, title]
         ↓
  parse_links_user_prompt()
  (Create AI prompt)
         ↓
  User + System Prompt
         ↓
  OpenAI Chat API (GPT-4o-mini)
         ↓
  Generated Funny Sports Article
         ↓
  display(Markdown())
  (Render in notebook)
```

---

## Key Technologies Used
| Technology | Purpose |
|-----------|---------|
| **BeautifulSoup** | HTML parsing and link extraction |
| **Requests** | HTTP requests to fetch web pages |
| **OpenAI API** | GPT-4o-mini model for article generation |
| **Python dotenv** | Environment variable management |
| **Jupyter/IPython** | Interactive notebook execution and display |

---

## Configuration Requirements
Before running this notebook you need:
1. `.env` file with `OPENAI_API_KEY` variable set
2. Active internet connection
3. Access to AP News website (or the target website in the code)

---

## Output
The notebook produces a humorous sports article that:
- References all the scraped article titles
- Uses witty, entertaining language
- Is written from the perspective of a funny sports journalist
- Is rendered as formatted Markdown in the notebook output

---

## Potential Modifications
- **Change the website**: Replace URL in Step 2 and Step 4 to scrape different news sources
- **Change the tone**: Modify the `system_prompt` to make the journalist more serious, sarcastic, etc.
- **Change the model**: Replace "gpt-4o-mini" with other OpenAI models (gpt-4, gpt-3.5-turbo, etc.)
- **Filter links**: Modify `parse_url_links()` to exclude certain links or sections
