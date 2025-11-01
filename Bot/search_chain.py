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

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

class SearchChain:
    """Chain for performing web searches and integrating results into responses"""
    
    def __init__(self, api_key: str, google_api_key: str, google_cse_id: str):
        """
        Initialize the search chain
        
        Args:
            api_key: OpenAI API key (from the user) - used for ALL operations including content generation and search determination
            google_api_key: Google API key (from the site owner)
            google_cse_id: Google Custom Search Engine ID (from the site owner)
        """
        self.api_key = api_key
        self.web_search_tool = WebSearchTool(google_api_key, google_cse_id)
        self.personality = "You are a helpful assistant."
        
        # Initialize the language model with the user's API key
        self.llm = ChatOpenAI(
            model="gpt-4",
            openai_api_key=api_key,
            temperature=0.3
        )
        
        # Initialize the internal LLM with the user's API key for search decisions
        self.internal_llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,  # Use user's API key instead of environment API key
            temperature=0.2
        )
        
        # Create the prompt template for search-augmented responses
        self.search_prompt_template = PromptTemplate(
            input_variables=["question", "search_results", "chat_history", "language", "personality"],
            template="""
            {personality}
            
            CRITICAL INSTRUCTION: You must respond in {language}. All text must be in {language}.
            
            You MUST embody the {personality} personality described above in ALL your responses, regardless of the web search content.
            
            You are an AI assistant with the ability to search the web for up-to-date information.
            
            User's question: {question}
            
            Search results:
            ---------------------
            {search_results}
            ---------------------
            
            Previous conversation:
            ---------------------
            {chat_history}
            ---------------------
            
            Based on the search results and the conversation history, provide a comprehensive answer to the user's question.
            
            Guidelines:
            1. Cite your sources by mentioning the websites or Wikipedia articles you're referencing
            2. If the search results don't provide relevant information, say so and provide your best answer based on your knowledge
            3. Be concise but thorough
            4. If there are multiple perspectives on a topic, present them fairly
            
            CRITICAL: Your response MUST maintain the personality traits, tone, and style described at the beginning. 
            The personality should affect HOW you respond, not WHAT information you provide.
            
            Remember to respond in {language}.
            """
        )
        
        # Create the chain for search-augmented responses
        self.search_chain = LLMChain(
            llm=self.llm,
            prompt=self.search_prompt_template
        )
    
    def needs_search(self, query: str) -> bool:
        """
        Determine if a query needs a web search
        
        Args:
            query: The user's query
            
        Returns:
            bool: True if the query needs a web search, False otherwise
        """
        # If force_search is enabled in session state, always use web search
        if st.session_state.get("force_search", False):
            return True
        
        # List of keywords that strongly indicate a need for web search
        search_keywords = [
            # General search terms
            "search", "find", "look up", "google", "internet", "web", "online",
            "latest", "recent", "news", "current", "today", "yesterday",
            "weather", "stock", "price", "cost", "worth", "value",
            "where is", "where can", "how to find", "where to buy",
            "who is", "what is happening", "what happened",
            
            # Time-sensitive query keywords
            "last", "latest", "most recent", "now", "current", "2024", "2023",
            "this year", "this month", "this week", "today", "yesterday",
            "upcoming", "next", "schedule", "when will", "when is",
            
            # Sports, competition and entertainment terms
            "won", "winner", "champion", "championship", "tournament", "competition",
            "olympia", "bodybuilding", "sport", "match", "game", "sports",
            "athlete", "player", "team", "league", "season", "standings",
            "ranked", "ranking", "medal", "score", "record", "title",
            
            # Personal information and social media terms
            "social media", "profile", "account", "facebook", "twitter", "instagram",
            "linkedin", "github", "youtube", "tiktok", "snapchat", "reddit",
            "available on", "find me", "my profile", "my account", "my social",
            
            # News and current events
            "news", "breaking", "headlines", "report", "announced", "released",
            "launched", "published", "confirmed", "revealed", "discovered",
            
            # Location and travel
            "location", "address", "directions", "map", "near", "close to",
            "distance", "travel", "trip", "visit", "tourist", "attraction",
            
            # Shopping and products
            "buy", "purchase", "price", "cost", "available", "in stock",
            "shipping", "delivery", "order", "product", "item", "store",
            
            # Technical and software
            "download", "install", "update", "version", "release", "bug",
            "feature", "update", "patch", "fix", "issue", "problem"
        ]
        
        # List of question types that can usually be answered without search
        general_knowledge_patterns = [
            "what is", "how do", "how does", "why is", "why do", "why does",
            "can you explain", "tell me about", "describe", "define",
            "tips", "advice", "help me", "guide", "steps", "how can I",
            "best practices", "recommendations", "suggest", "ideas for"
        ]
        
        query_lower = query.lower()
        
        # Check for search keywords
        for keyword in search_keywords:
            if keyword in query_lower:
               
                return True
        
        # Check if it's a general knowledge question
        for pattern in general_knowledge_patterns:
            if pattern in query_lower:
                
                return False
                
        # Special handling for "who" questions about people - these often need search
        if query_lower.startswith("who "):
            
            return True
            
        # Special handling for questions about events and competitions
        if any(term in query_lower for term in ["mr.", "mr ", "miss", "ms.", "ms ", "competition", "contest"]):
           
            return True
        
        # Check for specific date/time references that would need current info
        date_patterns = ["2023", "2024", "this year", "this month", "this week", "today"]
        for date in date_patterns:
            if date in query_lower:
                
                return True
        
        # Default to not using search for simple queries
        # Count the number of words - longer queries are more likely to need search
        word_count = len(query_lower.split())
        
        # If it's a very short query (1-3 words), it's likely a simple command or greeting
        if word_count <= 3:
            
            return False
        
        # If it's a very long query (>15 words), it's likely a complex question that might need search
        if word_count > 15:
           
            return True
        
        # For medium-length queries, be conservative and don't use search by default
        
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
        try:
            # Perform the search
            search_results = self.web_search_tool.search(question)
            
            # Format the search results
            formatted_results = self.format_search_results(search_results)
            
            # Format chat history for context
            formatted_history = ""
            for msg in chat_history[-5:]:  # Use last 5 messages for context
                role = "User" if msg["role"] == "user" else "Assistant"
                formatted_history += f"{role}: {msg['message']}\n"
            
            # Strongly emphasize language requirement in the system prompt
            strong_language_personality = self.personality + f"\n\nMANDATORY REQUIREMENT: You MUST respond in {language} ONLY. ALL text in your response MUST be in {language}."
            
            # Generate the answer using the user's OpenAI API key
            try:
                response = self.search_chain.invoke({
                    "question": question,
                    "search_results": formatted_results,
                    "chat_history": formatted_history,
                    "language": language,
                    "personality": strong_language_personality
                })
                
                return {
                    "answer": response["text"],
                    "search_results": search_results,
                    "is_url_processing": False,
                    "language": language  # Pass along the language preference
                }
            except Exception as e:
                error_message = str(e)
                if "quota" in error_message.lower() or "rate limit" in error_message.lower():
                    st.error(f"OpenAI API quota exceeded. Please check your API key limits or try again later.")
                    return {
                        "answer": "I found some information through web search, but I couldn't process it due to API usage limits with your OpenAI API key. I can still show you the search results I found.",
                        "search_results": search_results,
                        "is_url_processing": False,
                        "raw_search_results": formatted_results[:1000] + "..." if len(formatted_results) > 1000 else formatted_results,
                        "language": language  # Pass along the language preference
                    }
                else:
                    st.error(f"Error generating search response: {error_message}")
                    return {
                        "answer": f"I found some information through web search, but encountered an error while processing it: {error_message}. Here are the raw search results I found.",
                        "search_results": search_results,
                        "is_url_processing": False,
                        "raw_search_results": formatted_results[:1000] + "..." if len(formatted_results) > 1000 else formatted_results,
                        "language": language  # Pass along the language preference
                    }
        except Exception as e:
            st.error(f"Error performing web search: {str(e)}")
            return {
                "answer": f"I encountered an error while searching the web: {str(e)}. Let me try to answer based on what I know.",
                "search_results": None,
                "is_url_processing": False,
                "language": language  # Pass along the language preference
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
                    "is_url_processing": True,
                    "language": language
                }
            
            # Format chat history for context
            formatted_history = ""
            for msg in chat_history[-5:]:  # Use last 5 messages for context
                role = "User" if msg["role"] == "user" else "Assistant"
                formatted_history += f"{role}: {msg['message']}\n"
            
            # Create the messages list for streaming
            messages = [
                {"role": "system", "content": self.personality + f"\n\nMANDATORY REQUIREMENT: You MUST respond in {language} ONLY. ALL text in your response MUST be in {language}."},
                {"role": "user", "content": f"""
                URL: {url}
                
                Content from the URL:
                {content}
                
                User's question: {question}
                
                Previous conversation:
                {formatted_history}
                
                Based on the content from the URL and the previous conversation, provide a helpful response to the user's request.
                If the user asked to summarize, provide a concise summary of the main points.
                If the user asked a specific question about the content, answer it based on the information provided.
                If the content is not relevant to the request, explain why and provide the best response you can.
                """}
            ]
            
            # Generate streaming response
            stream = self.llm.stream(messages)
            full_response = ""
            
            # Process the stream and return the final response
            for chunk in stream:
                if chunk.content is not None:
                    full_response += chunk.content
            
            return {
                "answer": full_response,
                "url": url,
                "content": content[:500] + "..." if len(content) > 500 else content,
                "is_url_processing": True,
                "language": language
            }
            
        except Exception as e:
            st.error(f"Error processing URL: {str(e)}")
            return {
                "answer": f"I encountered an error while processing the URL: {str(e)}. Let me try to answer based on what I know.",
                "url": url,
                "content": "",
                "is_url_processing": True,
                "language": language
            }

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
            # Use the LLM to determine if search is needed and get an optimized query
            search_determination = self.determine_search_need_with_llm(query, chat_history, language)
            
            # If search is not needed, return a response indicating that
            if not search_determination["search_needed"]:
                return {
                    "answer": None,  # Signal to use the main model without search
                    "search_results": None,
                    "is_web_search": False,
                    "no_search_needed": True,
                    "reasoning": search_determination["reasoning"],
                    "language": language
                }
            
            # Use the reformulated query from the LLM
            improved_query = search_determination["reformulated_query"]
            
            # Format chat history for context
            formatted_history = ""
            for msg in chat_history[-5:]:  # Use last 5 messages for context
                role = "User" if msg["role"] == "user" else "Assistant"
                formatted_history += f"{role}: {msg['message']}\n"
            
            # Perform the web search with the improved query
            try:
                search_results = self.web_search_tool.search(improved_query)
            except Exception as e:
                st.error(f"Error performing web search: {str(e)}")
                return {
                    "answer": f"I determined that web search would be helpful for this query, but encountered an error: {str(e)}. Let me try to answer based on what I know.",
                    "search_results": None,
                    "is_web_search": False,
                    "error": str(e),
                    "language": language
                }
            
            # Strongly emphasize language requirement in the system prompt
            strong_language_personality = self.personality + f"\n\nMANDATORY REQUIREMENT: You MUST respond in {language} ONLY. ALL text in your response MUST be in {language}."
            
            # Generate the streaming response
            try:
                # Create the messages list for streaming
                messages = [
                    {"role": "system", "content": strong_language_personality},
                    {"role": "user", "content": f"""
                    Question: {query}
                    
                    Search results:
                    {self.format_search_results(search_results)}
                    
                    Previous conversation:
                    {formatted_history}
                    
                    Based on the search results and conversation history, provide a comprehensive answer.
                    Guidelines:
                    1. Cite your sources by mentioning the websites or Wikipedia articles you're referencing
                    2. If the search results don't provide relevant information, say so and provide your best answer
                    3. Be concise but thorough
                    4. If there are multiple perspectives on a topic, present them fairly
                    """}
                ]
                
                # Generate streaming response
                stream = self.llm.stream(messages)
                full_response = ""
                
                # Process the stream and return the final response
                for chunk in stream:
                    if chunk.content is not None:
                        full_response += chunk.content
                
                return {
                    "answer": full_response,
                    "search_results": search_results,
                    "is_web_search": True,
                    "improved_query": improved_query,
                    "reasoning": search_determination["reasoning"],
                    "language": language
                }
                
            except Exception as e:
                error_message = str(e)
                if "quota" in error_message.lower() or "rate limit" in error_message.lower():
                    st.error(f"OpenAI API quota exceeded. Please check your API key limits or try again later.")
                    return {
                        "answer": "I found some information through web search, but I couldn't process it due to API usage limits with your OpenAI API key. I can still show you the search results I found.",
                        "search_results": search_results,
                        "is_web_search": True,
                        "improved_query": improved_query,
                        "raw_search_results": self.format_search_results(search_results)[:1000] + "..." if len(self.format_search_results(search_results)) > 1000 else self.format_search_results(search_results),
                        "language": language
                    }
                else:
                    st.error(f"Error generating search response: {error_message}")
                    return {
                        "answer": f"I found some information through web search, but encountered an error while processing it: {error_message}. Here are the raw search results I found.",
                        "search_results": search_results,
                        "is_web_search": True,
                        "improved_query": improved_query,
                        "raw_search_results": self.format_search_results(search_results)[:1000] + "..." if len(self.format_search_results(search_results)) > 1000 else self.format_search_results(search_results),
                        "language": language
                    }
        except Exception as e:
            st.error(f"Error in web search determination: {str(e)}")
            return {
                "answer": f"I encountered an error while determining whether web search is needed: {str(e)}. Let me try to answer based on what I know.",
                "search_results": None,
                "is_web_search": False,
                "language": language
            }

    def determine_search_need_with_llm(self, query: str, chat_history: List[Dict[str, str]], language: str = "English", personality: str = None) -> Dict[str, Any]:
        """
        Use an LLM to determine if a web search is needed for the query
        
        Args:
            query: The user's query
            chat_history: Recent chat history
            language: Language to respond in
            personality: Optional personality to use (not used in current implementation)
            
        Returns:
            Dict containing decision and reformulated query if needed
        """
        # Using the user's API key for search determination logic
        
        # Format the chat history for context
        formatted_history = ""
        for msg in chat_history[-5:]:  # Use last 5 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['message']}\n"
        
        # Create the prompt for determining search need
        prompt_template = PromptTemplate(
            input_variables=["query", "chat_history", "current_date", "language"],
            template="""
            You are a specialized AI designed to determine whether a web search would be helpful to answer a user's query.
            
            Current date: {current_date}
            Language to use: {language}
            
            User's query: {query}
            
            Recent conversation:
            ---------------------
            {chat_history}
            ---------------------
            
            Please analyze this query and determine:
            1. If it's likely to need up-to-date information (news, current events, recent developments)
            2. If it's asking about specific facts that an AI model might not have training data on
            3. If it's asking about something time-sensitive or location-specific
            4. If it's about popular culture, sports, or current trends
            5. If it's about prices, availability, or other volatile information
            
            First, classify the query into one of these categories:
            - DEFINITE_SEARCH: Query definitely requires web search (current events, specific facts not in training data)
            - LIKELY_SEARCH: Query would probably benefit from web search
            - NO_SEARCH: Query can be answered without web search (general knowledge, opinions, creative tasks)
            
            Then, if search is needed, provide a reformulated search query optimized for web search engines.
            
            Return your response in this JSON format:
            {{
              "search_needed": "DEFINITE_SEARCH|LIKELY_SEARCH|NO_SEARCH",
              "reasoning": "Brief explanation of your decision",
              "reformulated_query": "Optimized search query if search is needed"
            }}
            
            IMPORTANT: Ensure that any reformulated query is appropriate for searching for content in {language}.
            
            Only respond with the JSON. No other text.
            """
        )
        
        # Create the chain for determining search need
        search_determination_chain = LLMChain(
            llm=self.internal_llm,
            prompt=prompt_template
        )
        
        # Get the current date
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # Get the decision from the model
            
            response = search_determination_chain.invoke({
                "query": query,
                "chat_history": formatted_history,
                "current_date": current_date,
                "language": language
            })
            
            # Parse the response
            import json
            try:
                result = json.loads(response["text"])
                
                if result["search_needed"] in ["DEFINITE_SEARCH", "LIKELY_SEARCH"]:
                    reformulated_query = result.get("reformulated_query", query)
                    return {
                        "search_needed": True,
                        "reformulated_query": reformulated_query,
                        "reasoning": result["reasoning"],
                        "language": language  # Add language to return value
                    }
                else:
                    return {
                        "search_needed": False,
                        "reformulated_query": query,
                        "reasoning": result["reasoning"],
                        "language": language  # Add language to return value
                    }
            except json.JSONDecodeError:
                st.warning(f"Failed to parse LLM response as JSON. Using heuristic determination instead.")
                # Fall back to the original needs_search method
                search_needed = self.needs_search(query)
                return {
                    "search_needed": search_needed,
                    "reformulated_query": query,
                    "reasoning": "Fallback to heuristic determination due to parsing error.",
                    "language": language  # Add language to return value
                }
        except Exception as e:
            st.warning(f"Error determining search need with LLM: {str(e)}. Falling back to heuristic method.")
            # Fall back to the original needs_search method
            search_needed = self.needs_search(query)
            return {
                "search_needed": search_needed,
                "reformulated_query": query,
                "reasoning": f"Fallback to heuristic determination due to error: {str(e)}",
                "language": language  # Add language to return value
            }
