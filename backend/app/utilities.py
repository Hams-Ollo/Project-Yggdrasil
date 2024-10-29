import re
import json
import datetime
from typing import Dict, List, Any

def format_timestamp() -> str:
    """Generate formatted ISO timestamp."""
    return datetime.datetime.now().isoformat()

def analyze_text_stats(text: str) -> Dict[str, Any]:
    """Analyze text and return statistics."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_words_per_sentence": round(len(words) / max(len(sentences), 1), 2),
        "unique_words": len(set(words))
    }

def validate_python_code(code: str) -> Dict[str, Any]:
    """Validate Python code and return analysis results."""
    try:
        compile(code, '<string>', 'exec')
        
        # Basic code analysis
        analysis = {
            "has_imports": bool(re.search(r'^(import|from)\s+\w+', code, re.MULTILINE)),
            "has_functions": bool(re.search(r'^def\s+\w+', code, re.MULTILINE)),
            "has_classes": bool(re.search(r'^class\s+\w+', code, re.MULTILINE)),
            "line_count": len(code.splitlines()),
            "is_empty": not bool(code.strip())
        }
        
        return {
            "valid": True,
            "analysis": analysis,
            "issues": _generate_code_issues(analysis)
        }
    except SyntaxError as e:
        return {
            "valid": False,
            "error": str(e),
            "line_number": e.lineno
        }

def _generate_code_issues(analysis: Dict[str, Any]) -> List[str]:
    """Generate list of potential code issues based on analysis."""
    issues = []
    if not analysis["has_imports"]:
        issues.append("No imports found - check if required")
    if analysis["is_empty"]:
        issues.append("Empty code block")
    if not analysis["has_functions"] and not analysis["has_classes"]:
        issues.append("No functions or classes defined")
    return issues if issues else ["No basic issues found"]

def save_research_note(content: str, source: str = None) -> Dict[str, Any]:
    """Format and save research notes."""
    return {
        "content": content,
        "source": source or "Not specified",
        "timestamp": format_timestamp(),
        "tags": extract_tags(content)
    }

def extract_tags(text: str) -> List[str]:
    """Extract hashtags or keywords from text."""
    hashtags = re.findall(r'#(\w+)', text)
    return list(set(hashtags))

def format_conversation_context(context_dict: Dict[str, str], limit: int = 5) -> str:
    """Format recent conversation context for display."""
    recent = dict(list(context_dict.items())[-limit:])
    return json.dumps(recent, indent=2) 