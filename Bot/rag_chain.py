import os
from typing import List, Dict, Any
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from document_processor import DocumentProcessor

class RAGChain:
    """Retrieval-Augmented Generation chain with multimodal support"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.document_processor = DocumentProcessor()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0.5)
        self.personality = "You are a helpful assistant."
        
        self.prompt_template = PromptTemplate(
            input_variables=["question", "context", "chat_history", "language", "personality"],
            template="""
            {personality}
            
            You MUST embody the personality described above in ALL your responses, regardless of the document content.
            
            You are an AI assistant that answers questions using documents and images.
            
            CRITICAL INSTRUCTION: You must respond in {language}. All text outside of direct quotes must be in {language}.
            
            Available Context (which may include multiple documents and images):
            ---------------------
            {context}
            ---------------------
            
            Previous conversation:
            ---------------------
            {chat_history}
            ---------------------
            
            Answer the question: {question}
            
            When answering, consider information from ALL available documents. If multiple documents are referenced, 
            synthesize information across them. Mention document names when appropriate to clarify sources.
            
            IMPORTANT: Your response MUST maintain the personality traits, tone, and style described at the beginning. 
            The personality should affect HOW you respond, not WHAT information you provide from the documents.
            
            Remember to respond in {language}.
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def is_relevant_to_documents(self, query: str, session_id: str) -> bool:
        """Smart relevance check considering both text and images"""
        has_images = self.has_images_for_session(session_id)
        has_docs = self.document_processor.has_documents(session_id)
        
        # If no documents or images, nothing to be relevant to
        if not (has_docs or has_images):
            return False
            
        # Check for common document-related query patterns
        query_lower = query.lower()
        
        # If force_rag is enabled in the Streamlit session state, always consider relevant
        if st.session_state.get("force_rag", False):
            return True
        
        # Image-related keywords
        image_keywords = ["image", "picture", "photo", "graphic", "visual", "diagram", "figure", 
                         "illustration", "appearance", "wearing", "looks", "scene", "background"]
        if has_images and any(kw in query_lower for kw in image_keywords):
            return True
            
        # Document-related keywords and phrases
        doc_keywords = [
            "document", "file", "pdf", "text", "content", "uploaded", "attachment",
            "summary", "summarize", "summarization", "overview", "brief", "extract",
            "what does the document say", "what is in the document", "according to the document",
            "based on the document", "from the document", "in the document", "the document mentions",
            "tell me about the document", "explain the document", "analyze the document",
            "what is the main point", "key points", "main ideas", "important information",
            "resume", "cv", "report", "paper", "article", "letter", "email", "memo",
            "suitable", "appropriate", "good for", "match", "fit", "compatible"
        ]
        
        if has_docs and any(kw in query_lower for kw in doc_keywords):
            return True
            
        # Check for questions that are likely about the document or image content
        question_starters = ["what", "who", "where", "when", "why", "how", "is", "are", "can", 
                            "could", "would", "should", "tell me", "describe", "explain"]
        if (has_docs or has_images) and any(query_lower.startswith(starter) for starter in question_starters):
            # For questions, use more permissive matching
            return True
                
        # For very short queries (likely follow-up questions), be more permissive
        if len(query_lower.split()) < 5:
            return True
                
        # Default case: if we have any content, treat it as relevant for safety
        return True

    def get_relevant_documents(self, question: str, session_id: str, k: int = 5) -> List:
        """
        Get relevant documents for a question with configurable number of results
        
        Args:
            question: The user's question
            session_id: The session ID
            k: Number of document chunks to retrieve (default: 5)
            
        Returns:
            List of document chunks
        """
        # Determine appropriate chunk size based on query complexity
        query_words = len(question.split())
        
        # For complex queries (longer questions), retrieve larger chunks for more context
        # For simple queries, retrieve smaller chunks for more precise answers
        if query_words > 15:  # Complex question
            chunk_size = 1500  # Larger chunk size
        elif query_words > 8:  # Medium complexity
            chunk_size = 1000  # Default chunk size
        else:  # Simple question
            chunk_size = 800   # Smaller chunk size
        
        # Get text documents with dynamic chunk size
        text_docs = self.document_processor.query_documents(
            question, 
            session_id, 
            k=k,
            chunk_size=chunk_size
        )
        
        # Convert to the format expected by the template
        converted_docs = []
        for doc in text_docs:
            try:
                # Check if the document is already in the right format
                if isinstance(doc, dict):
                    if "content" in doc and "metadata" in doc:
                        # Standard format from query_documents
                        converted_docs.append({
                            "page_content": doc["content"],
                            "metadata": doc.get("metadata", {})
                        })
                    elif "page_content" in doc and "metadata" in doc:
                        # Already in the right format
                        converted_docs.append(doc)
                    else:
                        # Unknown format, try to adapt
                        content = doc.get("content", doc.get("text", doc.get("page_content", "")))
                        metadata = doc.get("metadata", {})
                        if content:  # Only add if we found some content
                            converted_docs.append({
                                "page_content": content,
                                "metadata": metadata
                            })
                elif hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                    # It's an object with the right attributes
                    converted_docs.append({
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    })
                else:
                    # Try to convert any other format
                    content = str(doc)
                    converted_docs.append({
                        "page_content": content,
                        "metadata": {"source": "unknown"}
                    })
            except Exception as e:
                # Log the error but continue with other documents
                print(f"Error converting document: {e}")
                continue
        
        return converted_docs

    def answer_question(self, question: str, session_id: str, chat_history: List[Dict[str, str]], language: str = "English") -> Dict[str, Any]:
        """Generate answer with combined context"""
        image_analysis = self.document_processor.vision_processor.analyze_images(session_id, question)
        text_docs = self.document_processor.query_documents(question, session_id)
        context = self._combine_contexts(text_docs, image_analysis)
        
        formatted_history = "\n".join(
            f"{msg['role']}: {msg['message']}" 
            for msg in chat_history[-7:]
        )

        result = self.chain.invoke({
            "question": question,
            "context": context,
            "chat_history": formatted_history,
            "language": language,
            "personality": self.personality
        })
        
        return {
            "answer": result["text"],
            "sources": text_docs,
            "image_analysis": image_analysis
        }
        
    def _combine_contexts(self, text_docs: List, image_analysis: List) -> str:
        """Combine text and image contexts with improved formatting"""
        context = []
        
        if text_docs:
            # Group documents by source
            docs_by_source = {}
            for doc in text_docs:
                try:
                    if isinstance(doc, dict):
                        if "metadata" in doc and "source" in doc["metadata"]:
                            source = doc["metadata"]["source"]
                        else:
                            source = "Unknown Source"
                        
                        if source not in docs_by_source:
                            docs_by_source[source] = []
                        docs_by_source[source].append(doc)
                    else:
                        # For objects with metadata attribute
                        source = getattr(doc, "metadata", {}).get("source", "Unknown Source")
                        if source not in docs_by_source:
                            docs_by_source[source] = []
                        docs_by_source[source].append(doc)
                except Exception as e:
                    print(f"Error processing document: {e}")
                    continue
            
            # Add header for text documents
            context.append("📄 TEXT DOCUMENTS ANALYSIS:")
            
            # Add content from each source
            for source, docs in docs_by_source.items():
                context.append(f"\nDocument Source: {source}")
                for i, doc in enumerate(docs):
                    try:
                        if isinstance(doc, dict):
                            if "page_content" in doc:
                                content = doc["page_content"]
                            elif "content" in doc:
                                content = doc["content"]
                            else:
                                content = str(doc)
                        else:
                            # Try to get page_content attribute
                            content = getattr(doc, "page_content", str(doc))
                        
                        context.append(f"Content Section {i+1}: {content}")
                    except Exception as e:
                        print(f"Error extracting content: {e}")
                        continue
        
        if image_analysis:
            context.append("\n🖼️ IMAGE ANALYSIS:")
            context.extend([
                f"Image Description: {analysis['content']}"
                for analysis in image_analysis
            ])
        
        return "\n\n".join(context)
        
    def process_file(self, uploaded_file, session_id: str) -> bool:
        """Process an uploaded file"""
        return self.document_processor.process_file(uploaded_file, session_id)
        
    def clear_documents(self, session_id: str) -> bool:
        """Clear all documents for a session"""
        return self.document_processor.clear_documents(session_id)
        
    def has_documents_for_session(self, session_id: str) -> bool:
        """Check if there are any documents for the given session"""
        has_text_docs = self.document_processor.has_documents(session_id)
        has_images = self.has_images_for_session(session_id)
        return has_text_docs or has_images
    
    def has_images_for_session(self, session_id: str) -> bool:
        """Check if there are any images for the given session"""
        return (session_id in self.document_processor.vision_processor.image_store and 
                len(self.document_processor.vision_processor.image_store[session_id]) > 0) 