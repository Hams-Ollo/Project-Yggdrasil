import re
import json
import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_timestamp(format_type: str = "iso") -> str:
    """Generate formatted timestamp with multiple format options."""
    now = datetime.datetime.now()
    formats = {
        "iso": now.isoformat(),
        "readable": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S")
    }
    return formats.get(format_type, formats["iso"])

def analyze_text_stats(text: str) -> Dict[str, Any]:
    """Enhanced text analysis with readability metrics."""
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    characters = len(text)
    paragraphs = text.split('\n\n')
    
    # Calculate readability metrics
    syllable_count = sum(count_syllables(word) for word in words)
    flesch_score = calculate_flesch_score(len(words), len(sentences), syllable_count)
    
    return {
        "basic_stats": {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "character_count": characters,
            "paragraph_count": len(paragraphs),
            "avg_words_per_sentence": round(len(words) / max(len(sentences), 1), 2),
            "unique_words": len(set(words))
        },
        "readability": {
            "flesch_score": flesch_score,
            "readability_level": interpret_flesch_score(flesch_score),
            "avg_syllables_per_word": round(syllable_count / max(len(words), 1), 2)
        },
        "vocabulary_richness": analyze_vocabulary(words)
    }

def validate_python_code(code: str) -> Dict[str, Any]:
    """Enhanced Python code validation and analysis."""
    try:
        compile(code, '<string>', 'exec')
        
        analysis = {
            "structure": {
                "has_imports": bool(re.search(r'^(import|from)\s+\w+', code, re.MULTILINE)),
                "has_functions": bool(re.search(r'^def\s+\w+', code, re.MULTILINE)),
                "has_classes": bool(re.search(r'^class\s+\w+', code, re.MULTILINE)),
                "has_docstrings": bool(re.search(r'"""[\s\S]*?"""', code)),
                "has_type_hints": bool(re.search(r':\s*(str|int|float|bool|list|dict|Any)', code)),
                "line_count": len(code.splitlines()),
                "is_empty": not bool(code.strip())
            },
            "complexity": analyze_code_complexity(code),
            "style": check_code_style(code)
        }
        
        return {
            "valid": True,
            "analysis": analysis,
            "issues": _generate_code_issues(analysis)
        }
    except SyntaxError as e:
        logger.error(f"Syntax error in code validation: {str(e)}")
        return {
            "valid": False,
            "error": str(e),
            "line_number": e.lineno,
            "suggested_fix": suggest_syntax_fix(str(e))
        }

def save_research_note(content: str, source: Optional[str] = None, 
                      category: Optional[str] = None) -> Dict[str, Any]:
    """Enhanced research note saving with categorization and metadata."""
    tags = extract_tags(content)
    keywords = extract_keywords(content)
    
    note = {
        "content": content,
        "metadata": {
            "source": source or "Not specified",
            "category": category or "Uncategorized",
            "timestamp": format_timestamp("readable"),
            "tags": tags,
            "keywords": keywords,
            "word_count": len(content.split()),
            "summary": generate_summary(content)
        },
        "references": extract_references(content)
    }
    
    # Log the save operation
    logger.info(f"Research note saved with {len(tags)} tags and {len(keywords)} keywords")
    return note

# Helper functions
def count_syllables(word: str) -> int:
    """Count syllables in a word using basic rules."""
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    previous_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            count += 1
        previous_was_vowel = is_vowel
    
    if word.endswith('e'):
        count -= 1
    if count == 0:
        count = 1
    return count

def calculate_flesch_score(words: int, sentences: int, syllables: int) -> float:
    """Calculate Flesch Reading Ease score."""
    if sentences == 0 or words == 0:
        return 0
    return 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)

def interpret_flesch_score(score: float) -> str:
    """Interpret Flesch Reading Ease score."""
    if score >= 90: return "Very Easy"
    elif score >= 80: return "Easy"
    elif score >= 70: return "Fairly Easy"
    elif score >= 60: return "Standard"
    elif score >= 50: return "Fairly Difficult"
    elif score >= 30: return "Difficult"
    else: return "Very Difficult"

def analyze_vocabulary(words: List[str]) -> Dict[str, Any]:
    """Analyze vocabulary usage and complexity."""
    word_lengths = [len(word) for word in words]
    return {
        "avg_word_length": round(sum(word_lengths) / max(len(words), 1), 2),
        "longest_words": sorted(set(words), key=len, reverse=True)[:5],
        "vocabulary_size": len(set(words))
    }

def analyze_code_complexity(code: str) -> Dict[str, Any]:
    """Analyze code complexity metrics."""
    return {
        "cyclomatic_complexity": count_decision_points(code),
        "nesting_depth": max_nesting_depth(code),
        "function_count": len(re.findall(r'^def\s+\w+', code, re.MULTILINE))
    }

def check_code_style(code: str) -> Dict[str, bool]:
    """Check Python code style guidelines."""
    return {
        "follows_pep8_naming": bool(re.match(r'^[a-z_][a-z0-9_]*$', code)),
        "has_comments": bool(re.search(r'#.*$', code, re.MULTILINE)),
        "consistent_indentation": check_indentation(code)
    }

def suggest_syntax_fix(error: str) -> str:
    """Suggest potential fixes for common syntax errors."""
    common_fixes = {
        "EOF while scanning": "Check for missing closing quotes or parentheses",
        "invalid syntax": "Check for missing colons or incorrect indentation",
        "expected an indented block": "Add indentation after function/class definition"
    }
    for error_type, fix in common_fixes.items():
        if error_type in error:
            return fix
    return "Unable to suggest specific fix for this error"

def extract_tags(text: str) -> List[str]:
    """Extract hashtags from text."""
    return re.findall(r'#(\w+)', text)

def extract_keywords(text: str) -> List[str]:
    """Extract important keywords from text."""
    # Simple keyword extraction based on word frequency
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = {}
    for word in words:
        if len(word) > 3:  # Skip short words
            word_freq[word] = word_freq.get(word, 0) + 1
    return sorted(word_freq, key=word_freq.get, reverse=True)[:5]

def generate_summary(text: str, max_length: int = 100) -> str:
    """Generate a brief summary of the text."""
    sentences = re.split(r'[.!?]+', text)
    if not sentences:
        return ""
    return sentences[0].strip()[:max_length] + "..."

def extract_references(text: str) -> List[str]:
    """Extract references and citations from text."""
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    citations = re.findall(r'\([^)]+\d{4}[^)]*\)', text)
    return urls + citations

def count_decision_points(code: str) -> int:
    """Count decision points in code for cyclomatic complexity."""
    decision_keywords = ['if', 'elif', 'for', 'while', 'and', 'or']
    count = 1  # Base complexity
    for keyword in decision_keywords:
        count += len(re.findall(rf'\b{keyword}\b', code))
    return count

def max_nesting_depth(code: str) -> int:
    """Calculate maximum nesting depth in code."""
    lines = code.split('\n')
    max_depth = current_depth = 0
    for line in lines:
        indent = len(line) - len(line.lstrip())
        current_depth = indent // 4  # Assuming 4 spaces per indent level
        max_depth = max(max_depth, current_depth)
    return max_depth

def check_indentation(code: str) -> bool:
    """Check if code has consistent indentation."""
    lines = code.split('\n')
    indent_size = None
    
    for line in lines:
        if line.strip():  # Skip empty lines
            space_count = len(line) - len(line.lstrip())
            if space_count > 0:
                if indent_size is None:
                    indent_size = space_count
                elif space_count % indent_size != 0:
                    return False
    return True

def _generate_code_issues(analysis: Dict[str, Any]) -> List[str]:
    """Generate list of potential code issues based on analysis."""
    issues = []
    if analysis["complexity"]["cyclomatic_complexity"] > 10:
        issues.append("High cyclomatic complexity - consider refactoring")
    if analysis["complexity"]["nesting_depth"] > 4:
        issues.append("Deep nesting - consider flattening structure")
    if not analysis["style"]["has_comments"]:
        issues.append("Missing comments - add documentation")
    return issues