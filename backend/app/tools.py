from langchain_core.tools import BaseTool
from typing import Optional, Dict, Any
from .utilities import (
    analyze_text_stats,
    validate_python_code,
    save_research_note,
    format_conversation_context
)
from datetime import datetime

class ResearchNoteTool(BaseTool):
    """Tool for researchers to save and organize research findings"""
    name = "research_note"
    description = "Save and organize key research findings with timestamps and citations"
    
    def _run(self, content: str, source: Optional[str] = None) -> str:
        note = save_research_note(content, source)
        return f"Research note saved: {note}"

class TextAnalysisTool(BaseTool):
    """Tool for writers to analyze text statistics"""
    name = "text_analysis"
    description = "Analyze text for word count, readability, and basic statistics"
    
    def _run(self, text: str) -> Dict[str, Any]:
        return analyze_text_stats(text)

class CodeValidatorTool(BaseTool):
    """Tool for coders to perform basic code validation"""
    name = "code_validator"
    description = "Validate Python code syntax and provide basic static analysis"
    
    def _run(self, code: str) -> Dict[str, Any]:
        return validate_python_code(code)

class ConversationContextTool(BaseTool):
    """Tool for the general agent to track conversation context"""
    name = "conversation_context"
    description = "Track and analyze conversation context and user preferences"
    
    def __init__(self):
        super().__init__()
        self.context = {}
    
    def _run(self, user_input: str, action: str = "update") -> str:
        if action == "update":
            self.context[datetime.now().isoformat()] = user_input
            return "Context updated"
        elif action == "get":
            return format_conversation_context(self.context)