import os
import streamlit as st
from typing import Dict, Any, List, Tuple
from transformers import pipeline
import torch
import numpy as np

class SentimentAnalyzer:
    """Analyze user sentiment and emotion to adjust chatbot responses"""
    
    def __init__(self):
        """Initialize the sentiment analyzer with Hugging Face models"""
        # Initialize sentiment analysis pipeline
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
            
            # Initialize emotion detection pipeline
            self.emotion_detector = pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base", 
                return_all_scores=True
            )
            
            # Cache for storing recent sentiment scores
            self.sentiment_history = {}
            
            # Mapping of emotions to response strategies
            self.emotion_strategies = {
                "joy": {
                    "tone": "enthusiastic",
                    "style": "conversational",
                    "emoji": True
                },
                "sadness": {
                    "tone": "empathetic",
                    "style": "supportive",
                    "emoji": False
                },
                "anger": {
                    "tone": "calm",
                    "style": "direct",
                    "emoji": False
                },
                "fear": {
                    "tone": "reassuring",
                    "style": "clear",
                    "emoji": False
                },
                "surprise": {
                    "tone": "informative",
                    "style": "explanatory",
                    "emoji": True
                },
                "disgust": {
                    "tone": "neutral",
                    "style": "factual",
                    "emoji": False
                },
                "neutral": {
                    "tone": "balanced",
                    "style": "informative",
                    "emoji": True
                }
            }
            
            # Flag to indicate if models loaded successfully
            self.models_loaded = True
            
        except Exception as e:
            st.error(f"Error loading sentiment analysis models: {str(e)}")
            self.models_loaded = False
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment and emotion of a text
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict containing sentiment and emotion analysis results
        """
        if not self.models_loaded or not text:
            return {
                "sentiment": "neutral",
                "sentiment_score": 0.5,
                "emotion": "neutral",
                "emotion_score": 1.0,
                "confidence": 0.0
            }
        
        try:
            # Get sentiment (positive/negative)
            sentiment_results = self.sentiment_analyzer(text)[0]
            
            # Convert to a simpler format
            sentiment_dict = {item["label"]: item["score"] for item in sentiment_results}
            
            # Determine the dominant sentiment
            dominant_sentiment = max(sentiment_dict, key=sentiment_dict.get)
            sentiment_score = sentiment_dict[dominant_sentiment]
            
            # Normalize sentiment score to a -1 to 1 scale where:
            # -1 is very negative, 0 is neutral, 1 is very positive
            normalized_score = sentiment_dict.get("POSITIVE", 0.5) - sentiment_dict.get("NEGATIVE", 0.5)
            
            # Get emotion (joy, sadness, anger, fear, surprise, disgust)
            emotion_results = self.emotion_detector(text)[0]
            
            # Convert to a simpler format
            emotion_dict = {item["label"]: item["score"] for item in emotion_results}
            
            # Determine the dominant emotion
            dominant_emotion = max(emotion_dict, key=emotion_dict.get)
            emotion_score = emotion_dict[dominant_emotion]
            
            # Calculate overall confidence
            confidence = (sentiment_score + emotion_score) / 2
            
            return {
                "sentiment": dominant_sentiment.lower(),
                "sentiment_score": normalized_score,
                "emotion": dominant_emotion.lower(),
                "emotion_score": emotion_score,
                "confidence": confidence,
                "raw_sentiment": sentiment_dict,
                "raw_emotion": emotion_dict
            }
            
        except Exception as e:
            st.error(f"Error analyzing sentiment: {str(e)}")
            return {
                "sentiment": "neutral",
                "sentiment_score": 0.5,
                "emotion": "neutral",
                "emotion_score": 1.0,
                "confidence": 0.0
            }
    
    def track_sentiment(self, session_id: str, text: str) -> Dict[str, Any]:
        """
        Track sentiment over time for a session
        
        Args:
            session_id: The session ID
            text: The text to analyze
            
        Returns:
            Dict containing current and historical sentiment analysis
        """
        # Analyze current sentiment
        current_analysis = self.analyze_sentiment(text)
        
        # Initialize history for this session if it doesn't exist
        if session_id not in self.sentiment_history:
            self.sentiment_history[session_id] = []
        
        # Add current analysis to history (keep last 10 entries)
        self.sentiment_history[session_id].append(current_analysis)
        if len(self.sentiment_history[session_id]) > 10:
            self.sentiment_history[session_id].pop(0)
        
        # Calculate trend
        if len(self.sentiment_history[session_id]) > 1:
            sentiment_scores = [entry["sentiment_score"] for entry in self.sentiment_history[session_id]]
            current_score = sentiment_scores[-1]
            previous_score = sentiment_scores[-2]
            trend = current_score - previous_score
        else:
            trend = 0
        
        return {
            "current": current_analysis,
            "history": self.sentiment_history[session_id],
            "trend": trend
        }
    
    def get_response_strategy(self, session_id: str, text: str) -> Dict[str, Any]:
        """
        Get a response strategy based on sentiment analysis
        
        Args:
            session_id: The session ID
            text: The text to analyze
            
        Returns:
            Dict containing response strategy parameters
        """
        # Track sentiment
        sentiment_data = self.track_sentiment(session_id, text)
        current = sentiment_data["current"]
        
        # Get the base strategy for the detected emotion
        emotion = current["emotion"]
        base_strategy = self.emotion_strategies.get(
            emotion, 
            self.emotion_strategies["neutral"]
        )
        
        # Adjust strategy based on sentiment score
        sentiment_score = current["sentiment_score"]
        
        # Determine response adjustments
        adjustments = {}
        
        # Adjust based on sentiment intensity
        if abs(sentiment_score) > 0.8:
            # Strong sentiment (very positive or very negative)
            adjustments["intensity"] = "high"
        elif abs(sentiment_score) > 0.4:
            # Moderate sentiment
            adjustments["intensity"] = "medium"
        else:
            # Mild or neutral sentiment
            adjustments["intensity"] = "low"
        
        # Adjust based on sentiment trend
        if sentiment_data["trend"] < -0.2:
            # Sentiment is becoming more negative
            adjustments["trend"] = "declining"
        elif sentiment_data["trend"] > 0.2:
            # Sentiment is becoming more positive
            adjustments["trend"] = "improving"
        else:
            # Sentiment is stable
            adjustments["trend"] = "stable"
        
        # Special case: detect confusion or frustration
        text_lower = text.lower()
        
        # Check for confusion
        if any(term in text_lower for term in ["confusion", "confused", "don't understand", "not clear", "what do you mean"]):
            adjustments["special_case"] = "confusion"
        
        # Check for frustration with the assistant
        elif any(term in text_lower for term in ["you are bad", "you suck", "terrible assistant", "useless", "not helpful", 
                                               "don't like you", "hate you", "stupid assistant", "dumb", "stop using", 
                                               "didn't try", "don't try", "just did", "just using"]):
            adjustments["special_case"] = "assistant_frustration"
        
        # Check for general frustration
        elif any(term in text_lower for term in ["frustrated", "annoying", "not working", "doesn't work", "fed up", "tired of"]):
            adjustments["special_case"] = "frustration"
        
        # Combine base strategy with adjustments
        strategy = {**base_strategy, **adjustments}
        
        # Add the raw sentiment data for reference
        strategy["sentiment_data"] = sentiment_data
        
        return strategy
    
    def generate_system_prompt(self, strategy: Dict[str, Any]) -> str:
        """
        Generate a system prompt based on the response strategy
        
        Args:
            strategy: The response strategy
            
        Returns:
            str: The system prompt
        """
        # Base system prompt
        base_prompt = "You are a helpful AI assistant."
        
        # Add tone instruction
        tone_map = {
            "enthusiastic": "Be enthusiastic and positive in your response.",
            "empathetic": "Show empathy and understanding in your response.",
            "calm": "Maintain a calm and measured tone in your response.",
            "reassuring": "Be reassuring and supportive in your response.",
            "informative": "Focus on providing clear information in your response.",
            "neutral": "Maintain a neutral and objective tone in your response.",
            "balanced": "Provide a balanced perspective in your response."
        }
        
        tone_instruction = tone_map.get(strategy.get("tone"), "")
        
        # Add style instruction
        style_map = {
            "conversational": "Use a friendly, conversational style.",
            "supportive": "Offer support and encouragement.",
            "direct": "Be direct and to the point.",
            "clear": "Use simple, clear language.",
            "explanatory": "Provide detailed explanations.",
            "factual": "Focus on facts and avoid opinions.",
            "informative": "Provide comprehensive information."
        }
        
        style_instruction = style_map.get(strategy.get("style"), "")
        
        # Add emoji instruction
        emoji_instruction = "Feel free to use appropriate emojis." if strategy.get("emoji", False) else "Avoid using emojis."
        
        # Add special case instructions
        special_case_map = {
            "confusion": "The user seems confused. Break down complex concepts, use simpler language, and provide step-by-step explanations.",
            "frustration": "The user seems frustrated. Acknowledge their frustration, offer clear solutions, and be extra patient and helpful.",
            "assistant_frustration": "The user is expressing frustration with you as an assistant. Acknowledge their concerns, apologize sincerely, and focus on providing a thoughtful, personalized response without relying on external searches or generic answers. Show that you're listening and adapting to their feedback.",
            "frustrated_with_info_need": "The user is expressing frustration but also needs information. Begin by acknowledging their frustration and showing empathy. Then provide accurate, helpful information that directly addresses their question. Be especially thorough and clear in your explanation while maintaining a supportive tone."
        }
        
        special_case_instruction = special_case_map.get(strategy.get("special_case", ""), "")
        
        # Add intensity adjustments
        intensity_map = {
            "high": "Match the user's emotional intensity appropriately.",
            "medium": "Respond with moderate emotional engagement.",
            "low": "Keep your response measured and calm."
        }
        
        intensity_instruction = intensity_map.get(strategy.get("intensity"), "")
        
        # Add trend adjustments
        trend_map = {
            "declining": "The user's sentiment is becoming more negative. Be extra supportive and helpful.",
            "improving": "The user's sentiment is becoming more positive. Maintain this positive trajectory.",
            "stable": "Maintain consistency in your tone and approach."
        }
        
        trend_instruction = trend_map.get(strategy.get("trend"), "")
        
        # Combine all instructions
        instructions = [
            base_prompt,
            tone_instruction,
            style_instruction,
            emoji_instruction,
            special_case_instruction,
            intensity_instruction,
            trend_instruction
        ]
        
        # Filter out empty instructions and join
        system_prompt = " ".join([instr for instr in instructions if instr])
        
        return system_prompt 