import os
from typing import Dict, Any, List
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from code_executor import CodeExecutor

class ProgrammingAssistant:
    """AI Programming Assistant that can execute Python code"""
    
    def __init__(self, api_key: str):
        """Initialize the programming assistant with OpenAI API key"""
        self.api_key = api_key
        self.code_executor = CodeExecutor()
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
            temperature=0.2
        )
        
        # Create the prompt template for code generation
        self.code_generation_template = PromptTemplate(
            input_variables=["question", "chat_history", "language", "personality"],
            template="""
            {personality}
            CRITICAL INSTRUCTION: You must respond in {language}. All explanations (not code) must be in {language}.
            
            You MUST embody the {personality} personality described above in ALL your explanations, while maintaining technical accuracy in code.
            
            You are an AI programming assistant specialized in Python. Your task is to help users with programming questions by generating and executing code.
            
            
            
            Previous conversation:
            ---------------------
            {chat_history}
            ---------------------
            
            User's question: {question}
            
            If the user is asking a programming question that requires code execution, follow these steps:
            1. Analyze the problem carefully
            2. Generate Python code that solves the problem
            3. Explain what the code does
            
            Format your response as follows:
            
            ```explanation
            [Your explanation of the approach here - MUST reflect your personality]
            ```
            
            ```python
            [Your Python code here - this remains technically accurate regardless of personality]
            ```
            
            ```explanation
            [Additional explanation or expected output here - MUST reflect your personality]
            ```
            
            IMPORTANT GUIDELINES:
            - Only use standard Python libraries or these allowed libraries: math, random, datetime, collections, itertools, functools, json, re, string, time, numpy, pandas, matplotlib, seaborn, sklearn
            - Do not use these forbidden libraries: os, subprocess, sys, shutil, requests, socket, importlib, pickle, multiprocessing
            - Keep the code simple and focused on solving the specific problem
            - Ensure the code is complete and ready to execute
            - If the question is not a programming question, just provide a helpful response without code
            - ALL explanations (everything outside of code blocks) MUST be in {language} AND MUST reflect your personality
            - The code itself should remain in Python and be technically accurate
            
            Remember to respond in {language}.
            """
        )
        
        # Create the chain for code generation
        self.code_generation_chain = LLMChain(
            llm=self.llm,
            prompt=self.code_generation_template
        )
        
        # Create the prompt template for result interpretation
        self.result_interpretation_template = PromptTemplate(
            input_variables=["question", "code", "execution_result", "language", "personality"],
            template="""
            {personality}

            CRITICAL INSTRUCTION: You must respond in {language}. All explanations (not code) must be in {language}.
            
            You MUST embody the {personality} personality described above in ALL your explanations, while maintaining technical accuracy.
            
            You are an AI programming assistant. You've generated code to answer a user's question, and the code has been executed.
            
            User's question: {question}
            
            Code that was executed:
            ```python
            {code}
            ```
            
            Execution result:
            ```
            {execution_result}
            ```
            
            Please interpret the execution result and provide a helpful response to the user. If there were errors, explain what went wrong and how to fix it.
            
            CRITICAL: Your explanations MUST maintain the personality traits, tone, and style described at the beginning.
            The personality affects HOW you explain things, not the technical accuracy of your explanation.
            
            Remember to respond in {language}.
            """
        )
        
        # Create the chain for result interpretation
        self.result_interpretation_chain = LLMChain(
            llm=self.llm,
            prompt=self.result_interpretation_template
        )
    
    def is_programming_question(self, question: str) -> bool:
        """
        Determine if a question is likely a programming question that needs code execution
        
        Args:
            question: The user's question
            
        Returns:
            bool: True if it's likely a programming question needing code, False otherwise
        """
        # Simple heuristic: check for programming-related keywords
        programming_keywords = [
            "code", "program", "function", "algorithm", "error", 
            "debug", "implement", "script", "syntax", "variable", "class",
            "object", "method", "library", "module", "import", "exception",
            "loop", "array", "dictionary", "dataframe", "pandas",
            "numpy", "matplotlib", "plot", "graph", "calculate", "compute"
        ]
        
        # Strong indicators that the user wants code
        explicit_code_indicators = [
            "write code", "write a program", "code example", "sample code",
            "solve this", "write function", "implement function", "create algorithm",
            "how would you code", "can you code", "coding challenge", "write script",
            "in python", "using python", "python script", "python code", "javascript code",
            "develop a", "programmatically", "automate", "function that", "snippet",
            "code snippet", "class that", "method that", "solution in code"
        ]
        
        # Keywords that indicate informational queries (not needing code execution)
        informational_queries = [
            "who is", "who are", "richest", "wealthiest", "billionaire", "millionaire",
            "top 10", "top ten", "top 5", "top five", "show me", "tell me about",
            "list of", "rank of", "ranking", "wealth", "net worth", "fortune", "money",
            "what are the", "what is the", "person", "people", "individuals", "celebrities",
            "business", "companies", "corporation", "population", "demographics", "statistics",
            "famous", "popular", "influential", "powerful", "successful", "wealthy"
        ]
        
        # Keywords that indicate conceptual questions (not needing code execution)
        conceptual_keywords = [
            "what is", "why use", "difference between", "compare", "versus", "vs",
            "better than", "advantages", "disadvantages", "history of", "when to use",
            "purpose of", "explain", "definition", "concept", "theory", "principle"
        ]
        
        # Keywords that indicate non-programming topics that should be excluded
        non_programming_keywords = [
            "olympia", "bodybuilding", "competition", "sport", "athlete", "championship",
            "tournament", "winner", "won", "match", "game", "player", "team", "league",
            "mr.", "mr ", "miss", "ms.", "champion", "title", "rank", "ranking",
            "contest", "medal", "record", "sports", "season"
        ]
        
        # Words that could be ambiguous (like "list" which could mean Python list or just enumeration)
        ambiguous_terms = {
            "list": ["python list", "create list", "initialize list", "empty list", "list comprehension", 
                    "append to list", "list methods", "array list", "linked list", "list operations"],
            "array": ["numpy array", "array operations", "array methods", "2d array", "initialize array"],
            "function": ["define function", "create function", "write function", "function parameters"]
        }
        
        question_lower = question.lower()
        
        # First check for informational queries that should not trigger code
        for keyword in informational_queries:
            if keyword in question_lower:
                # Check if also contains explicit programming indicators
                if not any(indicator in question_lower for indicator in explicit_code_indicators):
                    return False
        
        # Check if it contains non-programming keywords
        for keyword in non_programming_keywords:
            if keyword in question_lower:
                return False
        
        # Check if it's a conceptual question about programming
        for keyword in conceptual_keywords:
            if keyword in question_lower:
                return False
        
        # Check for explicit code indicators (strongest signal)
        for phrase in explicit_code_indicators:
            if phrase in question_lower:
                return True
        
        # Handle ambiguous terms - only count if they appear in a programming context
        for term, contexts in ambiguous_terms.items():
            if term in question_lower:
                # If term appears in isolation without programming context, don't count it
                if not any(context in question_lower for context in contexts) and not any(kw in question_lower for kw in programming_keywords):
                    continue
                else:
                    return True
        
        # Check for other programming keywords
        for keyword in programming_keywords:
            if keyword in question_lower:
                return True
        
        # If it starts with listing or question words without programming context, it's likely not a programming question
        if any(question_lower.startswith(w) for w in ["list", "who", "when", "where", "what is", "what are", "show", "tell"]):
            # Only if also contains programming keywords
            if not any(kw in question_lower for kw in programming_keywords):
                return False
        
        # Default to not treating as a programming question
        return False
    
    def extract_code_blocks(self, text: str) -> List[str]:
        """
        Extract Python code blocks from text
        
        Args:
            text: The text containing code blocks
            
        Returns:
            List[str]: List of extracted code blocks
        """
        import re
        
        # Pattern to match Python code blocks
        pattern = r"```(?:python)?\s*(.*?)```"
        
        # Find all matches
        matches = re.findall(pattern, text, re.DOTALL)
        
        return matches
    
    def answer_programming_question(self, question: str, chat_history: List[Dict[str, str]], language: str = "English", personality: str = "You are a helpful assistant.") -> Dict[str, Any]:
        """
        Answer a programming question with code execution
        
        Args:
            question: The user's question
            chat_history: List of previous chat messages
            language: Language to respond in
            personality: The personality to use for responses
            
        Returns:
            Dict[str, Any]: A dictionary containing the answer and execution results
        """
        # Format chat history for context
        formatted_history = ""
        for msg in chat_history[-10:]:  # Increased from 5 to 20 messages for better context
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['message']}\n"
        
        # Generate code using the code generation chain
        response = self.code_generation_chain.invoke({
            "question": question,
            "chat_history": formatted_history,
            "language": language,
            "personality": personality
        })
        
        # Extract code blocks from the response
        code_blocks = self.extract_code_blocks(response["text"])
        
        # If no code blocks were found, return the response as is
        if not code_blocks:
            return {
                "answer": response["text"],
                "code_executed": False,
                "execution_result": None
            }
        
        # Get the first code block
        code = code_blocks[0]
        
        # Check if code execution is disabled in session state
        disable_execution = st.session_state.get("disable_code_execution", True)
        
        if disable_execution:
            # Skip execution if disabled
            return {
                "answer": response["text"],
                "code": code,
                "code_executed": False,
                "execution_result": None
            }
        
        # Execute the code if not disabled
        execution_result = self.code_executor.execute_code(code)
        
        # Format the execution result
        formatted_result = ""
        if execution_result["success"]:
            formatted_result += execution_result["output"]
        else:
            formatted_result += f"Error: {execution_result['error']}"
        
        # Interpret the result
        interpretation = self.result_interpretation_chain.invoke({
            "question": question,
            "code": code,
            "execution_result": formatted_result,
            "language": language,
            "personality": personality
        })
        
        return {
            "answer": interpretation["text"],
            "code_executed": True,
            "code": code,
            "execution_result": formatted_result
        } 
