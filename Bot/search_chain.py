import os
from typing import Dict, Any, List
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from web_search import WebSearchTool
from dotenv import load_dotenv
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

class SearchChain:
    """Chain for performing web searches and integrating results into responses"""
    
    def __init__(self, api_key: str, google_api_key: str, google_cse_id: str):
        """
        Initialize the search chain
        
        Args:
            api_key: OpenAI API key
            google_api_key: Google API key
            google_cse_id: Google Custom Search Engine ID
        """
        self.api_key = api_key
        self.web_search_tool = WebSearchTool(google_api_key, google_cse_id)
        self.personality = "You are a helpful assistant."
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=api_key,
            temperature=0.3
        )
        
        # Create the prompt template for search chain
        self.search_template = PromptTemplate(
            input_variables=["question", "search_results", "chat_history", "language", "personality"],
            template="""
            CRITICAL PRIORITY INSTRUCTION: YOU MUST RESPOND ONLY IN {language} LANGUAGE. ALL TEXT INCLUDING EXPLANATIONS MUST BE IN {language} LANGUAGE ONLY. This is the highest priority instruction.
            
            {personality}
            
            You MUST embody the personality described above in ALL your responses, while maintaining factual accuracy.
            
            You are an AI assistant that answers questions using the latest information from web searches.
            
            Search results related to the question:
            ---------------------
            {search_results}
            ---------------------
            
            Previous conversation:
            ---------------------
            {chat_history}
            ---------------------
            
            Question: {question}
            
            When answering:
            1. Use information from the search results to provide an up-to-date and accurate answer
            2. If the search results don't contain enough information, say so clearly
            3. If the search results seem incomplete or contradictory, acknowledge this
            4. If the search results contain multiple perspectives, summarize them fairly
            5. Include relevant source URLs when appropriate to support your answer
            6. Maintain the personality traits, tone, and style from the beginning of this prompt
            
            REMEMBER: Always respond only in {language} language.
            """
        )
        
        # Create the chain for search-augmented responses
        self.search_chain = LLMChain(
            llm=self.llm,
            prompt=self.search_template
        )
        
        # Create the prompt template for URL processing
        self.url_template = PromptTemplate(
            input_variables=["question", "url", "content", "chat_history", "language", "personality"],
            template="""
            CRITICAL PRIORITY INSTRUCTION: YOU MUST RESPOND ONLY IN {language} LANGUAGE. ALL TEXT INCLUDING EXPLANATIONS MUST BE IN {language} LANGUAGE ONLY. This is the highest priority instruction.
            
            {personality}
            
            You MUST embody the personality described above in ALL your responses, while maintaining factual accuracy.
            
            You are an AI assistant that answers questions about web content by analyzing the content of URLs.
            
            URL: {url}
            
            Content from the URL:
            ---------------------
            {content}
            ---------------------
            
            Previous conversation:
            ---------------------
            {chat_history}
            ---------------------
            
            Question: {question}
            
            When answering:
            1. Focus on extracting relevant information from the URL content to answer the question
            2. If the content doesn't fully answer the question, acknowledge the limitations
            3. Quote or paraphrase from the content when appropriate
            4. Stay true to what the content actually says - don't make up information
            5. If necessary, describe the general structure or purpose of the web page
            6. Maintain the personality traits, tone, and style from the beginning of this prompt
            
            REMEMBER: Always respond only in {language} language.
            """
        )
    
    def needs_search(self, query: str) -> bool:
        """
        Determine if a query needs a web search
        
        Args:
            query: The user's query
            
        Returns:
            bool: True if the query needs a web search, False otherwise
        """
        import re
        
        # If force_search is enabled in session state, always use web search
        if st.session_state.get("force_search", False):
            return True
            
        query_lower = query.lower()
        
        # Categories of queries that benefit from web search
        time_sensitive_patterns = [
            # Current information
            r"current", r"latest", r"recent", r"today", r"now", r"present",
            
            # People in changing positions 
            r"richest", r"poorest", r"president", r"ceo", r"leader", r"minister",
            
            # Changing metrics
            r"price", r"cost", r"rate", r"value", r"stock", r"market cap",
            
            # Rankings that change
            r"top \d+", r"best", r"worst", r"highest", r"lowest", r"ranked",
            
            # Time-relative references
            r"this (year|month|week)"
        ]
        
        # Check if query contains time-sensitive patterns
        for pattern in time_sensitive_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # For facts that need verification or might be context-dependent
        factual_patterns = [
            r"how many", r"when (did|was|will)", r"where is", 
            r"what is the (population|size|area|distance)",
            r"who (is|are) (the |currently |now )"
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, query_lower):
                return True
                
        # Check for news-related or current events queries
        news_keywords = ["news", "happened", "announced", "released", "launched", 
                        "election", "war", "disaster", "event", "update"]
                        
        if any(keyword in query_lower for keyword in news_keywords):
            return True
            
        # Check for explicit year mentions (likely needs current info)
        year_pattern = r"\b20\d{2}\b"
        if re.search(year_pattern, query_lower):
            return True
            
        # Check for corrections or contradictions to previous information
        correction_patterns = [
            r"actually", r"no,", r"incorrect", r"wrong", r"not true", 
            r"that's not", r"that is not", r"isn't", r"is not"
        ]
        
        for pattern in correction_patterns:
            if re.search(pattern, query_lower):
                return True
                
        # List of search keywords from the original implementation that are still valid
        search_keywords = [
            "search", "find", "look up", "google", "internet", "web", "online",
            "weather", "stock", "worth", "where can", "how to find", "where to buy"
        ]
        
        # Check for search keywords
        for keyword in search_keywords:
            if keyword in query_lower:
                return True
        
        # Default to not using search for simple queries
        return False
    
    def format_search_results(self, results: Dict[str, Any]) -> str:
        """
        Format search results for inclusion in the prompt
        
        Args:
            results: The search results
            
        Returns:
            str: Formatted search results
        """
        formatted_results = f"SEARCH QUERY: {results['query']}\n\n"
        
        # Add Wikipedia results if available
        wiki_result = results["wikipedia_result"]
        if wiki_result.get("success", False):
            formatted_results += f"WIKIPEDIA: {wiki_result['title']}\n"
            formatted_results += f"{wiki_result['summary']}\n"
            formatted_results += f"Source: {wiki_result['url']}\n\n"
        
        # Add Google search results
        formatted_results += "GOOGLE SEARCH RESULTS:\n"
        for i, result in enumerate(results["google_results"]):
            formatted_results += f"{i+1}. {result['title']}\n"
            formatted_results += f"   {result['snippet']}\n"
            formatted_results += f"   URL: {result['link']}\n\n"
        
        # Add content extracts
        formatted_results += "CONTENT EXTRACTS:\n"
        for i, extract in enumerate(results["content_extracts"]):
            formatted_results += f"EXTRACT {i+1}: {extract['title']}\n"
            formatted_results += f"Source: {extract['link']}\n"
            formatted_results += f"{extract['content'][:1000]}...\n\n"
        
        return formatted_results
    
    def answer_with_search(self, question: str, chat_history: List[Dict[str, str]], language: str = "English") -> Dict[str, Any]:
        """
        Answer a question using web search results or process a URL
        
        Args:
            question: The user's question
            chat_history: List of previous chat messages
            language: Language to respond in
            
        Returns:
            Dict[str, Any]: A dictionary containing the answer and search results or URL content
        """
        # Check if the question contains a URL
        is_url_question = self.is_url(question)
        st.info(f"URL detection: {'URL detected' if is_url_question else 'No URL detected'}")
        
        if is_url_question:
            try:
                url = self.extract_url(question)
                st.info(f"Processing URL: {url}")
                
                # Extract content from the URL
                content = self.web_search_tool.extract_content_from_url(url)
                
                if not content or content.startswith("Error extracting content"):
                    st.warning(f"Failed to extract content from URL: {url}")
                    # Fall back to regular search if content extraction fails
                    return self._perform_regular_search(question, chat_history, language)
                
                return self.process_url(url, question, chat_history, language)
            except Exception as e:
                st.error(f"Error processing URL: {str(e)}")
                # Fall back to regular search if URL processing fails
                return self._perform_regular_search(question, chat_history, language)
        
        # If not a URL, perform a regular search
        return self._perform_regular_search(question, chat_history, language)

    def _perform_regular_search(self, question: str, chat_history: List[Dict[str, str]], language: str) -> Dict[str, Any]:
        """Helper method to perform a regular search"""
        st.info("Performing web search")
        # Perform the search
        search_results = self.web_search_tool.search(question)
        
        # Format the search results
        formatted_results = self.format_search_results(search_results)
        
        # Format chat history for context
        formatted_history = ""
        for msg in chat_history[-7:]:  # Use last 5 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['message']}\n"
        
        # Generate the answer
        response = self.search_chain.invoke({
            "question": question,
            "search_results": formatted_results,
            "chat_history": formatted_history,
            "language": language,
            "personality": self.personality
        })
        
        return {
            "answer": response["text"],
            "search_results": search_results,
            "is_url_processing": False
        }

    def is_url(self, text: str) -> bool:
        """
        Check if the text contains a URL
        
        Args:
            text: The text to check
            
        Returns:
            bool: True if the text contains a URL, False otherwise
        """
        # More robust URL detection regex
        url_pattern = re.compile(
            r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*(?:\?[\w=&]*)?'
        )
        
        # Check if the text contains a URL
        match = url_pattern.search(text)
        
        # If there's a match, consider it a URL request
        if match:
            return True
        
        return False

    def extract_url(self, text: str) -> str:
        """
        Extract URL from text
        
        Args:
            text: The text containing a URL
            
        Returns:
            str: The extracted URL
        """
        url_pattern = re.compile(
            r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*(?:\?[\w=&]*)?'
        )
        match = url_pattern.search(text)
        if match:
            return match.group(0)
        return ""

    def process_url(self, url: str, question: str, chat_history: List[Dict[str, Any]], language: str = "English") -> Dict[str, Any]:
        """
        Process a URL and generate a response
        
        Args:
            url: The URL to process
            question: The user's question
            chat_history: List of previous chat messages
            language: Language to respond in
            
        Returns:
            Dict[str, Any]: A dictionary containing the answer and URL content
        """
        try:
            # Extract content from the URL
            content = self.web_search_tool.extract_content_from_url(url)
            
            if not content:
                return {
                    "answer": f"I couldn't extract content from the URL: {url}. The website might be blocking automated access or the URL might be invalid.",
                    "url": url,
                    "content": "",
                    "is_url_processing": True
                }
            
            # Create a chain for URL processing
            url_chain = LLMChain(
                llm=self.llm,
                prompt=self.url_template
            )
            
            # Format chat history for context
            formatted_history = ""
            for msg in chat_history[-5:]:  # Use last 5 messages for context
                role = "User" if msg["role"] == "user" else "Assistant"
                formatted_history += f"{role}: {msg['message']}\n"
            
            # Generate the response
            response = url_chain.invoke({
                "url": url,
                "content": content,
                "question": question,
                "chat_history": formatted_history,
                "language": language,
                "personality": self.personality
            })
            
            return {
                "answer": response["text"],
                "url": url,
                "content": content[:500] + "..." if len(content) > 500 else content,  # Truncate for display
                "is_url_processing": True  # Add this flag to indicate it's URL processing
            }
        except Exception as e:
            st.error(f"Error processing URL: {str(e)}")
            return {
                "answer": f"I encountered an error while processing the URL: {str(e)}. Let me try to answer based on what I know.",
                "url": url,
                "content": "",
                "is_url_processing": True
            }

    def enhance_search_query(self, original_query: str) -> str:
        """
        Enhance a search query with appropriate time context
        
        Args:
            original_query: The original user query
            
        Returns:
            str: Enhanced query optimized for web search
        """
        query_lower = original_query.lower()
        
        # Get current date information
        now = datetime.now()
        current_year = now.year
        current_month = now.month
        
        # First check if the query already contains a year
        year_pattern = r"\b20\d{2}\b"
        has_year = re.search(year_pattern, query_lower)
        
        # Check if this is a query that would benefit from time context
        time_sensitive_keywords = [
            "current", "latest", "recent", "now", "present",
            "richest", "poorest", "president", "ceo", "leader",
            "price", "cost", "rate", "value", "worth",
            "top", "best", "worst", "highest", "lowest", "ranked",
            "winner", "champion", "record"
        ]
        
        needs_time_context = any(keyword in query_lower for keyword in time_sensitive_keywords)
        
        # If the query needs time context and doesn't already have a year, add the current year
        if needs_time_context and not has_year:
            enhanced_query = f"{original_query} {current_year} current"
        else:
            enhanced_query = original_query
            
        # Add search optimization terms for certain query types
        if re.search(r"who (is|are) (the )?richest", query_lower) or "top richest" in query_lower:
            enhanced_query += " forbes billionaires list net worth"
            
        elif re.search(r"who (is|are) (the )?(president|prime minister|leader)", query_lower):
            enhanced_query += " current official"
            
        elif re.search(r"(price|cost|value) of", query_lower):
            enhanced_query += " current market price"
            
        # Return the enhanced query
        return enhanced_query
        
    def search_with_web(self, query: str, chat_history: List[Dict[str, Any]], language: str = "English") -> Dict[str, Any]:
        """
        Search the web and generate a response with full conversation context
        
        Args:
            query: The user's query
            chat_history: List of previous chat messages
            language: Language to respond in
            
        Returns:
            Dict[str, Any]: A dictionary containing the answer and search results
        """
        try:
            # First, generate a better search query based on the conversation context
            improved_query = self.enhance_search_query(query)
            
            # Log the original and improved queries for debugging
            st.info(f"Original query: {query}")
            st.info(f"Improved query: {improved_query}")
            
            # Perform the web search with the improved query
            search_results = self.web_search_tool.search(improved_query)
            
            # Format chat history for context
            formatted_history = ""
            for msg in chat_history[-5:]:  # Use last 5 messages for context
                role = "User" if msg["role"] == "user" else "Assistant"
                formatted_history += f"{role}: {msg['message']}\n"
            
            # Generate the answer
            response = self.search_chain.invoke({
                "question": query,  # Use the original question for response generation
                "search_results": self.format_search_results(search_results),
                "chat_history": formatted_history,
                "language": language,
                "personality": self.personality
            })
            
            return {
                "answer": response["text"],
                "search_results": search_results,
                "is_web_search": True,
                "improved_query": improved_query  # Include the improved query in the response
            }
        except Exception as e:
            st.error(f"Error in web search: {str(e)}")
            return {
                "answer": f"I encountered an error while searching the web: {str(e)}. Let me try to answer based on what I know.",
                "search_results": None,
                "is_web_search": False
            }