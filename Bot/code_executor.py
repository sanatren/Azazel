import sys
import io
import traceback
import ast
import streamlit as st
from contextlib import redirect_stdout, redirect_stderr
import threading
import time

class CodeExecutor:
    """Execute Python code in a controlled environment"""
    
    def __init__(self, timeout=10):
        """Initialize the code executor with a timeout"""
        self.timeout = timeout
        
        # Define a list of potentially dangerous operations
        self.forbidden_modules = [
            'os', 'subprocess', 'sys', 'shutil', 'requests', 
            'socket', 'importlib', 'pickle', 'multiprocessing'
        ]
        
        # Define a list of allowed modules for import
        self.allowed_modules = [
            'math', 'random', 'datetime', 'collections', 'itertools', 
            'functools', 'json', 're', 'string', 'time', 'numpy', 
            'pandas', 'matplotlib', 'seaborn', 'sklearn'
        ]
    
    def _is_safe_code(self, code):
        """
        Check if the code is safe to execute
        
        Args:
            code: The Python code to check
            
        Returns:
            bool: True if the code is safe, False otherwise
        """
        try:
            # Parse the code into an AST
            parsed = ast.parse(code)
            
            # Check for imports of forbidden modules
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name in self.forbidden_modules:
                            return False, f"Import of '{name.name}' is not allowed for security reasons."
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.forbidden_modules:
                        return False, f"Import from '{node.module}' is not allowed for security reasons."
                
                # Check for exec or eval calls
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ['exec', 'eval', 'compile']:
                        return False, f"Use of '{node.func.id}' is not allowed for security reasons."
            
            return True, ""
        
        except SyntaxError as e:
            return False, f"Syntax error in code: {str(e)}"
        except Exception as e:
            return False, f"Error analyzing code: {str(e)}"
    
    def execute_code(self, code):
        """
        Execute Python code and return the result
        
        Args:
            code: The Python code to execute
            
        Returns:
            dict: A dictionary containing the execution result, output, and error
        """
        # Check if the code is safe
        is_safe, message = self._is_safe_code(code)
        if not is_safe:
            return {
                "success": False,
                "output": "",
                "error": message
            }
        
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Create a dictionary for storing the result
        result = {
            "success": False,
            "output": "",
            "error": ""
        }
        
        # Execute the code with a timeout
        def execute():
            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    # Create a restricted globals dictionary
                    restricted_globals = {
                        "__builtins__": {
                            name: __builtins__[name] 
                            for name in dir(__builtins__) 
                            if name not in ['open', 'exec', 'eval', 'compile', '__import__']
                        }
                    }
                    
                    # Add allowed modules to globals
                    for module_name in self.allowed_modules:
                        try:
                            module = __import__(module_name)
                            restricted_globals[module_name] = module
                        except ImportError:
                            pass
                    
                    # Execute the code
                    exec(code, restricted_globals)
                
                result["success"] = True
            except Exception as e:
                result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        # Create and start the execution thread
        thread = threading.Thread(target=execute)
        thread.daemon = True
        thread.start()
        
        # Wait for the thread to complete or timeout
        thread.join(self.timeout)
        
        # Check if the thread is still alive (timeout occurred)
        if thread.is_alive():
            result["error"] = f"Execution timed out after {self.timeout} seconds."
            return result
        
        # Get the output
        result["output"] = stdout_capture.getvalue()
        
        # If there was an error, add it to the result
        if stderr_capture.getvalue():
            result["error"] = stderr_capture.getvalue()
        
        return result 