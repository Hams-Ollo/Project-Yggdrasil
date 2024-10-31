CODER_VALIDATION_TESTS = {
    "code_quality": [
        {
            "prompt": "Write a Python function to implement a binary search tree",
            "validation_criteria": [
                "Correct implementation",
                "Error handling", 
                "Input validation",
                "Documentation",
                "Consistent style"
            ]
        }
    ],
    "debugging": [
        {
            "prompt": "Debug this code: def sort_list(lst): return sorted(lst, key=lambda x: x[1])",
            "validation_criteria": [
                "Problem identification",
                "Solution explanation",
                "Error prevention",
                "Alternative approaches",
                "Testing suggestions"
            ]
        }
    ],
    "system_design": [
        {
            "prompt": "Design a basic REST API for a todo application",
            "validation_criteria": [
                "Architecture explanation",
                "Code structure",
                "API endpoints",
                "Error handling",
                "Security considerations"
            ]
        }
    ],
    "algorithm_implementation": [
        {
            "prompt": "Implement a solution for the traveling salesman problem",
            "validation_criteria": [
                "Algorithm explanation",
                "Complexity analysis",
                "Optimization suggestions",
                "Code efficiency",
                "Example usage"
            ]
        }
    ],
    "code_review": [
        {
            "prompt": "Review this authentication implementation: [code snippet]",
            "validation_criteria": [
                "Security analysis",
                "Best practices check",
                "Performance review",
                "Style consistency",
                "Improvement suggestions"
            ]
        }
    ]
}

def validate_code_syntax(code_string, language):
    """
    Validates code syntax using appropriate parser
    """
    try:
        if language == "python":
            import ast
            ast.parse(code_string)
            return True
    except SyntaxError:
        return False
    return True

def validate_coder_response(prompt_type, response):
    """
    Validates coder agent responses against defined criteria
    
    Args:
        prompt_type (str): Type of test being validated
        response (str): Agent's response
        
    Returns:
        dict: Validation results with scores and feedback
    """
    test_case = CODER_VALIDATION_TESTS[prompt_type][0]
    criteria = test_case["validation_criteria"]
    
    results = {
        "criteria_met": [],
        "criteria_missed": [],
        "score": 0,
        "feedback": [],
        "improvement_suggestions": [],
        "security_concerns": [],
        "performance_notes": []
    }
    
    # Extract code blocks from response
    import re
    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
    
    # Validate based on prompt type
    if prompt_type == "code_quality":
        # Check for code presence
        if code_blocks:
            results["criteria_met"].append("Code implementation")
            
            # Validate syntax
            if validate_code_syntax(code_blocks[0], "python"):
                results["criteria_met"].append("Correct syntax")
            else:
                results["criteria_missed"].append("Correct syntax")
                results["feedback"].append("Code contains syntax errors")
            
            # Check for documentation
            if '"""' in code_blocks[0] or "'''" in code_blocks[0]:
                results["criteria_met"].append("Documentation")
            else:
                results["criteria_missed"].append("Documentation")
                results["feedback"].append("Missing docstrings")
            
            # Check for error handling
            if "try" in code_blocks[0] and "except" in code_blocks[0]:
                results["criteria_met"].append("Error handling")
            else:
                results["criteria_missed"].append("Error handling")
                results["feedback"].append("Missing error handling")
        else:
            results["criteria_missed"].append("Code implementation")
            results["feedback"].append("No code provided in response")
    
    # Calculate score
    results["score"] = (len(results["criteria_met"]) / len(criteria)) * 100
    
    return results