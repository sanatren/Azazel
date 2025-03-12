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
            model="gpt-3.5-turbo",
            openai_api_key=api_key,
            temperature=0.2
        )
        
        # Create the prompt template for code generation
        self.code_generation_template = PromptTemplate(
            input_variables=["question", "chat_history"],
            template="""
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
            [Your explanation of the approach here]
            ```
            
            ```python
            [Your Python code here]
            ```
            
            ```explanation
            [Additional explanation or expected output here]
            ```
            
            IMPORTANT GUIDELINES:
            - Only use standard Python libraries or these allowed libraries: math, random, datetime, collections, itertools, functools, json, re, string, time, numpy, pandas, matplotlib, seaborn, sklearn
            - Do not use these forbidden libraries: os, subprocess, sys, shutil, requests, socket, importlib, pickle, multiprocessing
            - Keep the code simple and focused on solving the specific problem
            - Ensure the code is complete and ready to execute
            - If the question is not a programming question, just provide a helpful response without code
            """
        )
        
        # Create the chain for code generation
        self.code_generation_chain = LLMChain(
            llm=self.llm,
            prompt=self.code_generation_template
        )
        
        # Create the prompt template for result interpretation
        self.result_interpretation_template = PromptTemplate(
            input_variables=["question", "code", "execution_result", "language"],
            template="""
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
            
            IMPORTANT: You must respond in {language}. If you don't know how to speak {language}, do your best to translate your response to {language}.
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
            "loop", "array", "list", "dictionary", "dataframe", "pandas",
            "numpy", "matplotlib", "plot", "graph", "calculate", "compute"
        ]
        
        # Keywords that indicate conceptual questions (not needing code execution)
        conceptual_keywords = [
            "what is", "why use", "difference between", "compare", "versus", "vs",
            "better than", "advantages", "disadvantages", "history of", "when to use",
            "purpose of", "explain", "definition", "concept", "theory", "principle"
        ]
        
        question_lower = question.lower()
        
        # Check if it's a conceptual question about programming
        for keyword in conceptual_keywords:
            if keyword in question_lower:
                return False
        
        # Check if any programming keyword is in the question
        for keyword in programming_keywords:
            if keyword in question_lower:
                return True
        
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
    
    def answer_programming_question(self, question: str, chat_history: List[Dict[str, str]], language: str = "English") -> Dict[str, Any]:
        """
        Answer a programming question with code execution
        
        Args:
            question: The user's question
            chat_history: List of previous chat messages
            language: Language to respond in
            
        Returns:
            Dict[str, Any]: A dictionary containing the answer and execution results
        """
        # Format chat history for context
        formatted_history = ""
        for msg in chat_history[-5:]:  # Use last 5 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['message']}\n"
        
        # Generate code using the code generation chain
        response = self.code_generation_chain.invoke({
            "question": question,
            "chat_history": formatted_history
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
            "language": language
        })
        
        return {
            "answer": interpretation["text"],
            "code_executed": True,
            "code": code,
            "execution_result": formatted_result
        } 