from langchain_core.tools import BaseTool
from typing import Optional, Dict, Any, List
from .utilities import (
    analyze_text_stats,
    validate_python_code,
    save_research_note,
    format_conversation_context,
    format_timestamp,
    logger
)
import json
from pathlib import Path
from datetime import datetime

class ResearchNoteTool(BaseTool):
    """Enhanced tool for researchers to save and organize research findings"""
    name = "research_note"
    description = "Save and organize research findings with advanced categorization and metadata"
    
    def __init__(self):
        super().__init__()
        self.notes_directory = Path("research_notes")
        self.notes_directory.mkdir(exist_ok=True)
        self.categories = set()
    
    def _run(self, content: str, source: Optional[str] = None, 
             category: Optional[str] = None) -> Dict[str, Any]:
        try:
            note = save_research_note(content, source, category)
            
            # Save to file system
            timestamp = format_timestamp("readable").replace(":", "-")
            filename = self.notes_directory / f"note_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(note, f, indent=2)
            
            if category:
                self.categories.add(category)
            
            return {
                "status": "success",
                "note": note,
                "file": str(filename)
            }
        except Exception as e:
            logger.error(f"Error saving research note: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_categories(self) -> List[str]:
        """Get all available note categories."""
        return sorted(list(self.categories))

class TextAnalysisTool(BaseTool):
    """Enhanced tool for writers to analyze text"""
    name = "text_analysis"
    description = "Comprehensive text analysis including readability metrics and vocabulary analysis"
    
    def _run(self, text: str, analysis_type: str = "full") -> Dict[str, Any]:
        try:
            analysis = analyze_text_stats(text)
            
            # Filter analysis based on type
            if analysis_type == "basic":
                return analysis["basic_stats"]
            elif analysis_type == "readability":
                return analysis["readability"]
            elif analysis_type == "vocabulary":
                return analysis["vocabulary_richness"]
            
            return analysis
        except Exception as e:
            logger.error(f"Error in text analysis: {str(e)}")
            return {"error": str(e)}

class CodeValidatorTool(BaseTool):
    """Enhanced tool for code validation and analysis"""
    name = "code_validator"
    description = "Advanced Python code validation with style checking and complexity analysis"
    
    def __init__(self):
        super().__init__()
        self.validation_history = []
    
    def _run(self, code: str, save_history: bool = True) -> Dict[str, Any]:
        try:
            result = validate_python_code(code)
            
            if save_history:
                self.validation_history.append({
                    "timestamp": format_timestamp(),
                    "code_length": len(code),
                    "valid": result["valid"]
                })
            
            return result
        except Exception as e:
            logger.error(f"Error in code validation: {str(e)}")
            return {"error": str(e)}
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get statistics about past validations."""
        if not self.validation_history:
            return {"message": "No validation history available"}
        
        total = len(self.validation_history)
        valid = sum(1 for v in self.validation_history if v["valid"])
        
        return {
            "total_validations": total,
            "successful_validations": valid,
            "success_rate": round(valid / total * 100, 2)
        }

class ConversationContextTool(BaseTool):
    """Enhanced tool for conversation context management"""
    name = "conversation_context"
    description = "Advanced conversation context tracking with sentiment and topic analysis"
    
    def __init__(self):
        super().__init__()
        self.context = {}
        self.topics = set()
        self.session_start = format_timestamp()
    
    def _run(self, user_input: str, action: str = "update") -> Dict[str, Any]:
        try:
            if action == "update":
                timestamp = format_timestamp()
                self.context[timestamp] = {
                    "input": user_input,
                    "topics": self._extract_topics(user_input)
                }
                return {
                    "status": "updated",
                    "topics_identified": list(self._extract_topics(user_input))
                }
            
            elif action == "get":
                return {
                    "recent_context": format_conversation_context(self.context),
                    "session_duration": self._get_session_duration(),
                    "total_interactions": len(self.context),
                    "topics_discussed": list(self.topics)
                }
            
            elif action == "clear":
                self.context.clear()
                self.topics.clear()
                return {"status": "context cleared"}
            
        except Exception as e:
            logger.error(f"Error in conversation context tool: {str(e)}")
            return {"error": str(e)}
    
    def _extract_topics(self, text: str) -> set:
        """Simple topic extraction from text."""
        # This could be enhanced with NLP techniques
        words = set(text.lower().split())
        self.topics.update(words)
        return words
    
    def _get_session_duration(self) -> str:
        """Calculate session duration."""
        start = datetime.fromisoformat(self.session_start)
        duration = datetime.now() - start
        return str(duration).split('.')[0]  # Remove microseconds