WRITER_VALIDATION_TESTS = {
    "content_creation": [
        {
            "prompt": "Write a blog post about artificial intelligence in healthcare",
            "validation_criteria": [
                "Clear structure (intro, body, conclusion)",
                "Engaging hook",
                "Appropriate tone", 
                "Logical flow",
                "Call to action"
            ]
        }
    ],
    "technical_writing": [
        {
            "prompt": "Create a user guide for a new mobile app",
            "validation_criteria": [
                "Clear instructions",
                "Step-by-step format",
                "Technical term definitions",
                "Visual cues (bullets, numbers)",
                "User-friendly language"
            ]
        }
    ],
    "creative_writing": [
        {
            "prompt": "Write a short story about a programmer's first day",
            "validation_criteria": [
                "Character development",
                "Engaging narrative",
                "Descriptive language",
                "Coherent plot",
                "Satisfying conclusion"
            ]
        }
    ],
    "business_writing": [
        {
            "prompt": "Write a product launch email for a new SaaS platform",
            "validation_criteria": [
                "Professional tone",
                "Clear value proposition",
                "Compelling headlines",
                "Effective CTA",
                "Brand consistency"
            ]
        }
    ],
    "content_optimization": [
        {
            "prompt": "Optimize this blog post for SEO: [sample content]",
            "validation_criteria": [
                "Keyword integration",
                "Header optimization",
                "Meta description",
                "Readability improvements",
                "Internal linking suggestions"
            ]
        }
    ]
}

def validate_writer_response(prompt_type, response):
    """
    Validates writer agent responses against defined criteria
    
    Args:
        prompt_type (str): Type of test being validated
        response (str): Agent's response
        
    Returns:
        dict: Validation results with scores and feedback
    """
    test_case = WRITER_VALIDATION_TESTS[prompt_type][0]
    criteria = test_case["validation_criteria"]
    
    results = {
        "criteria_met": [],
        "criteria_missed": [],
        "score": 0,
        "feedback": [],
        "improvement_suggestions": []
    }
    
    # Add specific validation logic for each criteria type
    if prompt_type == "content_creation":
        # Check for structure
        if "introduction" in response.lower() and "conclusion" in response.lower():
            results["criteria_met"].append("Clear structure")
        else:
            results["criteria_missed"].append("Clear structure")
            results["feedback"].append("Missing clear introduction or conclusion")
            
        # Check for hook
        first_paragraph = response.split('\n')[0]
        if len(first_paragraph) > 20 and ('?' in first_paragraph or '!' in first_paragraph):
            results["criteria_met"].append("Engaging hook")
        else:
            results["criteria_missed"].append("Engaging hook")
            results["feedback"].append("Opening could be more engaging")
            
        # Additional checks can be added here...
    
    # Calculate score
    results["score"] = (len(results["criteria_met"]) / len(criteria)) * 100
    
    return results

def run_writer_validation_suite():
    """
    Runs all writer validation tests and generates a comprehensive report
    """
    results = {}
    for test_type in WRITER_VALIDATION_TESTS.keys():
        test_case = WRITER_VALIDATION_TESTS[test_type][0]
        # Placeholder for actual response testing
        response = "Test response"
        results[test_type] = validate_writer_response(test_type, response)
    
    return results

if __name__ == "__main__":
    results = run_writer_validation_suite()
    print("\n=== Writer Agent Validation Results ===")
    for test_type, result in results.items():
        print(f"\n{test_type.upper()}")
        print(f"Score: {result['score']}%")
        print("Criteria Met:", ", ".join(result["criteria_met"]))
        print("Needs Improvement:", ", ".join(result["criteria_missed"]))
        if result["feedback"]:
            print("Feedback:", "\n- ".join(result["feedback"]))