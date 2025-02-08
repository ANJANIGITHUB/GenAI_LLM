import pdfkit
import traceback
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

def extract_confluence_details(confluence_url):
    try:
        # Extract base URL up to 'wiki'
        if "/wiki" in confluence_url:
            base_url = confluence_url.split("/wiki")[0] + "/wiki/"
        else:
            return "Invalid URL: '/wiki' not found", None

        # Extract the Page ID
        # Page ID appears after '/pages/' in the URL
        page_id = None
        if "/pages/" in confluence_url:
            parts = confluence_url.split("/pages/")
            if len(parts) > 1:
                page_id = parts[1].split("/")[0]  # Get the part after 'pages/' up to the next '/'
        
        return base_url, page_id
    except Exception as e:
        return f"Error: {e}", None

# Output results
#print("Base URL up to 'wiki':", baseurl)
#print("Page ID:", pageid)


def get_confluence_content(confluence_url):

    # Example Confluence URL
    
    # Get base URL and Page ID
    baseurl, pageid = extract_confluence_details(confluence_url)

    # Replace with your values
    confluence_base_url =baseurl# 
    page_id = pageid#"183189373829"  # Replace with the Confluence page ID
    
    #api_token = "Your API Token" # Your API token
    api_token= os.getenv("api_token")
    email    = os.getenv("email")

    # Confluence REST API URL to fetch content
    url = f"{confluence_base_url}/rest/api/content/{page_id}?expand=body.storage"

    # Authentication
    auth = HTTPBasicAuth(email, api_token)

    # Make the request
    response = requests.get(url, auth=auth)
    
    try:
        # Check response status
        if response.status_code == 200:
            # Parse JSON response
            data = response.json()
            # Extract page content (HTML format)
            content = data["body"]["storage"]["value"]
            #print("Page Content (HTML):")
            #print(content)
    
            plain_text = BeautifulSoup(content, "html.parser").get_text()
            #print("Page Content (Plain Text):")
            return plain_text
        else:
            print(f"Failed to fetch page. Status code: {response.status_code}")
            print(response.text)
    except Exception as e:
        return f"Error: {e}", None


if __name__ == "__main__":
    load_dotenv()
    get_confluence_content(confluence_url)



