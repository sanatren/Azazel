# Fix the missing client attribute in DocumentProcessor
import os
import tempfile
import streamlit as st
from typing import List, Dict, Any, Optional
import docx
import pandas as pd
from pptx import Presentation
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from vision_processor import VisionProcessor

class DocumentProcessor:
    """Process various document types for RAG applications"""
    
    def __init__(self):
        """Initialize the document processor"""
        # Initialize the text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize the embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Dictionary to store vectorstores by session ID
        self.vectorstores = {}
        
        # Initialize vision processor
        self.vision_processor = VisionProcessor(os.getenv("OPENAI_API_KEY"))
    
    def process_file(self, uploaded_file, session_id: str) -> bool:
        """
        Process an uploaded file and store its contents in the vector database
        
        Args:
            uploaded_file: The uploaded file object from Streamlit
            session_id: The session ID to associate with the document
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Check if file is image
            if uploaded_file.type.startswith('image/'):
                return self.vision_processor.process_image(uploaded_file, session_id)
            
            # Create a temporary file to save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name
            
            # Extract text based on file type
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                text = self._extract_text_from_pdf(file_path)
            elif file_extension == 'docx':
                text = self._extract_text_from_docx(file_path)
            elif file_extension in ['xlsx', 'xls']:
                text = self._extract_text_from_excel(file_path)
            elif file_extension == 'pptx':
                text = self._extract_text_from_pptx(file_path)
            elif file_extension == 'txt':
                text = self._extract_text_from_txt(file_path)
            else:
                st.error(f"Unsupported file type: {file_extension}")
                return False
            
            # Clean up the temporary file
            os.unlink(file_path)
            
            if not text:
                st.warning(f"No text could be extracted from {uploaded_file.name}")
                return False
            
            # Split the text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create documents with metadata
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": uploaded_file.name,
                        "chunk_id": i,
                        "session_id": session_id
                    }
                )
                for i, chunk in enumerate(chunks)
            ]
            
            # Create or update the vectorstore for this session
            if session_id in self.vectorstores:
                # Add documents to existing vectorstore
                self.vectorstores[session_id].add_documents(documents)
            else:
                # Create a new vectorstore
                self.vectorstores[session_id] = FAISS.from_documents(
                    documents, self.embeddings
                )
            
            return True
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return False
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file"""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from a Word document"""
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.error(f"Error extracting text from Word document: {str(e)}")
            return ""
    
    def _extract_text_from_excel(self, file_path: str) -> str:
        """Extract text from an Excel file"""
        try:
            df = pd.read_excel(file_path)
            return df.to_string()
        except Exception as e:
            st.error(f"Error extracting text from Excel file: {str(e)}")
            return ""
    
    def _extract_text_from_pptx(self, file_path: str) -> str:
        """Extract text from a PowerPoint presentation"""
        try:
            prs = Presentation(file_path)
            text = ""
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from PowerPoint: {str(e)}")
            return ""
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            st.error(f"Error extracting text from text file: {str(e)}")
            return ""
    
    def query_documents(self, query: str, session_id: str, k: int = 8, chunk_size: int = None) -> List[Dict[str, Any]]:
        """
        Query documents for a session with configurable chunk size
        
        Args:
            query: The query string
            session_id: The session ID
            k: Number of documents to retrieve (increased default from 4 to 8)
            chunk_size: Optional custom chunk size for this query (None uses default)
            
        Returns:
            List of documents with content and metadata
        """
        try:
            # Check if the vectorstore exists for this session
            if session_id not in self.vectorstores:
                print(f"No vectorstore found for session {session_id}")
                return []
            
            # Get the vectorstore
            vectorstore = self.vectorstores[session_id]
            
            # Query the vectorstore with a higher k to ensure multiple documents are considered
            try:
                results = vectorstore.similarity_search_with_score(query, k=k)
            except Exception as search_error:
                print(f"Error performing similarity search: {search_error}")
                return []
            
            # Format the results
            docs = []
            unique_sources = set()  # Track unique document sources
            
            try:
                # If a custom chunk size is specified and different from our default,
                # we need to adjust the content returned
                if chunk_size and chunk_size != 1000:  # 1000 is our default chunk size
                    print(f"Using custom chunk size: {chunk_size} characters")
                    
                    # Process each document to adjust chunk size
                    for doc, score in results:
                        try:
                            # Safely extract content and metadata
                            if hasattr(doc, "page_content"):
                                content = doc.page_content
                                metadata = doc.metadata if hasattr(doc, "metadata") else {}
                            elif isinstance(doc, dict):
                                content = doc.get("page_content", doc.get("content", ""))
                                metadata = doc.get("metadata", {})
                            else:
                                # Skip if we can't determine content format
                                continue
                            
                            # Add to unique sources set
                            source = metadata.get("source", "")
                            if source:
                                unique_sources.add(source)
                            
                            # For larger chunks, try to expand context if possible
                            if chunk_size > 1000:
                                # This is a simplified approach - in a real implementation,
                                # you might want to retrieve neighboring chunks from the original text
                                docs.append({
                                    "content": content,  # In a real implementation, you'd expand this
                                    "metadata": metadata,
                                    "score": score
                                })
                            # For smaller chunks, truncate to the specified size
                            elif chunk_size < 1000:
                                # Simple truncation - in a real implementation, you might
                                # want to do smarter truncation that preserves sentence boundaries
                                truncated_content = content[:chunk_size]
                                docs.append({
                                    "content": truncated_content,
                                    "metadata": metadata,
                                    "score": score
                                })
                        except Exception as doc_error:
                            print(f"Error processing document: {doc_error}")
                            continue
                else:
                    # Use the default chunk size
                    for doc, score in results:
                        try:
                            # Safely extract content and metadata
                            if hasattr(doc, "page_content"):
                                content = doc.page_content
                                metadata = doc.metadata if hasattr(doc, "metadata") else {}
                            elif isinstance(doc, dict):
                                content = doc.get("page_content", doc.get("content", ""))
                                metadata = doc.get("metadata", {})
                            else:
                                # Skip if we can't determine content format
                                continue
                            
                            # Add to unique sources set
                            source = metadata.get("source", "")
                            if source:
                                unique_sources.add(source)
                            
                            docs.append({
                                "content": content,
                                "metadata": metadata,
                                "score": score
                            })
                        except Exception as doc_error:
                            print(f"Error processing document: {doc_error}")
                            continue
                
                # Log the number of unique documents found
                if len(unique_sources) > 0:
                    print(f"Retrieved content from {len(unique_sources)} unique documents: {', '.join(unique_sources)}")
                
                return docs
            except Exception as format_error:
                print(f"Error formatting search results: {format_error}")
                return []
        except Exception as e:
            print(f"Error querying documents: {str(e)}")
            return []
    
    def has_documents(self, session_id: str) -> bool:
        """
        Check if documents exist for a session
        
        Args:
            session_id: The session ID
            
        Returns:
            bool: True if documents exist, False otherwise
        """
        # Simply check if the session_id is in the vectorstores dictionary
        return session_id in self.vectorstores
    
    def clear_documents(self, session_id: str) -> bool:
        """Clear all documents for a session"""
        if session_id in self.vectorstores:
            del self.vectorstores[session_id]
            return True
        return False