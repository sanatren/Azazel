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
        self.llm = ChatOpenAI(model="gpt-4", openai_api_key=api_key, temperature=0.5)
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
        
        # Image-related keywords
        image_keywords = ["image", "picture", "photo", "graphic", "visual", "diagram", "figure", "illustration"]
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
            "resume", "cv", "report", "paper", "article", "letter", "email", "memo"
        ]
        
        if has_docs and any(kw in query_lower for kw in doc_keywords):
            return True
            
        # Check for questions that are likely about the document content
        question_starters = ["what", "who", "where", "when", "why", "how", "is", "are", "can", "could", "would", "should", "tell me"]
        if has_docs and any(query_lower.startswith(starter) for starter in question_starters):
            # For questions, use semantic search with a more permissive threshold
            docs = self.document_processor.query_documents(query, session_id, k=1)
            if docs and docs[0]["score"] < 0.8:  # More permissive threshold (0.8 instead of 0.7)
                return True
                
        # For very short queries (likely follow-up questions), be more permissive
        if len(query_lower.split()) < 5:
            return True
                
        # Default case: if we only have images, use them
        return has_images and not has_docs

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
        """Combine text and image contexts"""
        context = []
        
        if text_docs:
            # Group documents by source
            docs_by_source = {}
            for doc in text_docs:
                source = doc['metadata']['source']
                if source not in docs_by_source:
                    docs_by_source[source] = []
                docs_by_source[source].append(doc)
            
            # Add header for text documents
            context.append("Text Documents:")
            
            # Add content from each source
            for source, docs in docs_by_source.items():
                context.append(f"\nDocument: {source}")
                for doc in docs:
                    context.append(f"Content: {doc['content']}")
        
        if image_analysis:
            context.append("\nImage Analysis:")
            context.extend([
                f"Image Analysis: {analysis['content']}"
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