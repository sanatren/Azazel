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
            input_variables=["question", "chat_history", "language", "personality"],
            template="""
            {personality}
            
            You MUST embody the personality described above in ALL your explanations, while maintaining technical accuracy in code.
            
            You are an AI programming assistant specialized in Python. Your task is to help users with programming questions by generating and executing code.
            
            CRITICAL INSTRUCTION: You must respond in {language}. All explanations (not code) must be in {language}.
            
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
            
            You MUST embody the personality described above in ALL your explanations, while maintaining technical accuracy.
            
            You are an AI programming assistant. You've generated code to answer a user's question, and the code has been executed.
            
            CRITICAL INSTRUCTION: You must respond in {language}. All explanations (not code) must be in {language}.
            
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
        import re
        
        # First, check for explicit programming requests
        explicit_programming_patterns = [
            r"\bcode\b.*\bfor\b", r"\bwrite\b.*\bprogram\b", r"\bfunction\b.*\bto\b",
            r"\bimplement\b.*\balgorithm\b", r"\bdebug\b", r"\bsyntax\b", r"\bcompile\b",
            r"\bcoding\b", r"\bscript\b.*\bto\b"
        ]
        
        for pattern in explicit_programming_patterns:
            if re.search(pattern, question.lower()):
                return True
        
        # Check for programming concepts that require code
        programming_concepts = [
            "algorithm", "function", "class", "method", "variable", "loop", "recursion",
            "data structure", "api", "interface", "database query", "sql", "regex",
            "parameter", "argument", "return value", "object", "exception", "module"
        ]
        
        # Only consider it programming if these concepts are paired with action verbs
        action_verbs = [
            "create", "write", "implement", "develop", "code", "build", "design", 
            "fix", "solve", "optimize", "generate", "define", "declare"
        ]
        
        question_lower = question.lower()
        
        # Check for action verb + programming concept pairs
        for verb in action_verbs:
            for concept in programming_concepts:
                if f"{verb} {concept}" in question_lower:
                    return True
        
        # Check if the question is asking for a list or information that shouldn't be code
        information_patterns = [
            r"who (is|are)", r"what (is|are)", r"list of", r"top \d+", 
            r"give me", r"tell me about", r"show me", r"where", r"when", 
            r"richest", r"largest", r"newest", r"oldest", r"best", r"worst",
            r"arrange", r"sort", r"order", r"examples of", r"instances of"
        ]
        
        for pattern in information_patterns:
            if re.search(pattern, question_lower):
                return False
        
        # Check libraries that often indicate programming tasks
        if any(lib in question_lower for lib in ["pandas", "numpy", "tensorflow", "matplotlib", "sklearn"]):
            return True
            
        # Check for programming language mentions paired with tasks
        languages = ["python", "javascript", "java", "c++", "ruby", "php", "go", "rust", "c#"]
        for lang in languages:
            if lang in question_lower and any(v in question_lower for v in action_verbs):
                return True
                
        # Check for specific tasks that should NOT be treated as programming
        non_programming_tasks = [
            "list", "comparison", "difference between", "meaning of", "definition",
            "explain", "summarize", "order", "arrange", "rank", "sort by"
        ]
        
        for task in non_programming_tasks:
            if task in question_lower:
                return False
        
        # Default to not programming unless explicit indicators found
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