import os
import json
import requests
import wikipedia
import streamlit as st
from typing import List, Dict, Any, Optional
from googleapiclient.discovery import build
from bs4 import BeautifulSoup

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

class WebSearchTool:
    """Tool for performing web searches and Wikipedia lookups"""
    
    def __init__(self, google_api_key: str, google_cse_id: str):
        """
        Initialize the web search tool
        
        Args:
            google_api_key: Google API key for Custom Search JSON API
            google_cse_id: Google Custom Search Engine ID
        """
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        
        # Initialize the Google Custom Search API service
        try:
            self.google_service = build(
                "customsearch", "v1", 
                developerKey=self.google_api_key
            )
        except Exception as e:
            st.warning(f"Failed to initialize Google Search API: {str(e)}")
            self.google_service = None
    
    def google_search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Perform a Google search
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            List of dictionaries containing search results
        """
        if not self.google_service:
            st.error(f"Google Search API not initialized. API Key: {self.google_api_key[:5]}..., CSE ID: {self.google_cse_id[:5]}...")
            return [{"title": "Search Error", "link": "", "snippet": "Google Search API not initialized."}]
        
        try:
            # Execute the search
            st.info(f"Searching for: {query}")
            result = self.google_service.cse().list(
                q=query,
                cx=self.google_cse_id,
                num=num_results
            ).execute()
            
            # Extract the search results
            search_results = []
            if "items" in result:
                for item in result["items"]:
                    search_results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })
                st.success(f"Found {len(search_results)} results")
                return search_results
            else:
                st.warning("No search results found")
                return [{"title": "No Results", "link": "", "snippet": "No search results found."}]
        
        except Exception as e:
            st.error(f"Google search error: {str(e)}")
            return [{"title": "Search Error", "link": "", "snippet": f"Error: {str(e)}"}]
    
    def wikipedia_search(self, query: str, sentences: int = 3) -> Dict[str, Any]:
        """
        Search Wikipedia for information
        
        Args:
            query: The search query
            sentences: Number of sentences to return
            
        Returns:
            Dictionary containing Wikipedia search results
        """
        try:
            # Search for Wikipedia pages
            search_results = wikipedia.search(query)
            
            if not search_results:
                return {
                    "title": "",
                    "summary": "No Wikipedia results found.",
                    "url": "",
                    "success": False
                }
            
            # Get the most relevant page
            page_title = search_results[0]
            page = wikipedia.page(page_title)
            
            # Get the summary
            summary = wikipedia.summary(page_title, sentences=sentences)
            
            return {
                "title": page.title,
                "summary": summary,
                "url": page.url,
                "success": True
            }
        
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation pages
            if e.options:
                try:
                    # Try to get the first option
                    page_title = e.options[0]
                    page = wikipedia.page(page_title)
                    summary = wikipedia.summary(page_title, sentences=sentences)
                    
                    return {
                        "title": page.title,
                        "summary": summary,
                        "url": page.url,
                        "success": True,
                        "disambiguation": e.options[:5]  # Include top 5 options
                    }
                except Exception:
                    pass
            
            return {
                "title": query,
                "summary": f"Multiple Wikipedia entries found: {', '.join(e.options[:5])}...",
                "url": "",
                "success": False,
                "disambiguation": e.options[:5]
            }
        
        except Exception as e:
            return {
                "title": query,
                "summary": f"Error searching Wikipedia: {str(e)}",
                "url": "",
                "success": False
            }
    
    def extract_content_from_url(self, url: str) -> str:
        """
        Extract the main content from a URL
        
        Args:
            url: The URL to extract content from
            
        Returns:
            Extracted text content
        """
        try:
            st.info(f"Attempting to extract content from: {url}")
            # Send a request to the URL
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('Content-Type', '').lower()
            st.info(f"Content type: {content_type}")
            
            # Handle different content types
            if 'text/html' in content_type:
                # Parse the HTML content
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Remove script and style elements
                for script in soup(["script", "style", "header", "footer", "nav"]):
                    script.extract()
                
                # Get the text content
                text = soup.get_text(separator="\n")
                
                # Clean up the text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = "\n".join(chunk for chunk in chunks if chunk)
                
                st.success(f"Successfully extracted {len(text)} characters of content")
                # Limit the text length
                return text[:10000]
            elif 'application/pdf' in content_type:
                st.warning("PDF content detected - PDF extraction not implemented")
                return "This appears to be a PDF document. PDF content extraction is not supported."
            else:
                st.warning(f"Unsupported content type: {content_type}")
                return f"Unable to extract content from this URL. Unsupported content type: {content_type}"
        
        except Exception as e:
            st.error(f"Error extracting content: {str(e)}")
            return f"Error extracting content: {str(e)}"
    
    def search(self, query: str) -> Dict[str, Any]:
        """
        Perform a comprehensive search using Google and Wikipedia
        
        Args:
            query: The search query
            
        Returns:
            Dictionary containing search results
        """
        results = {
            "query": query,
            "google_results": [],
            "wikipedia_result": {},
            "content_extracts": []
        }
        
        # Perform Google search
        google_results = self.google_search(query)
        results["google_results"] = google_results
        
        # Perform Wikipedia search
        wiki_result = self.wikipedia_search(query)
        results["wikipedia_result"] = wiki_result
        
        # Extract content from top 2 Google results
        content_extracts = []
        for result in google_results[:2]:
            if result["link"]:
                content = self.extract_content_from_url(result["link"])
                content_extracts.append({
                    "title": result["title"],
                    "link": result["link"],
                    "content": content
                })
        
        results["content_extracts"] = content_extracts
        
        return results 