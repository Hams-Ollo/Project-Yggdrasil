RESEARCHER_VALIDATION_TESTS = {
    "information_accuracy": [
        {
            "prompt": "What is quantum computing and what are its latest developments?",
            "validation_criteria": [
                "Includes basic explanation",
                "Mentions recent developments",
                "Cites sources or timeframes",
                "Distinguishes facts from predictions"
            ]
        }
    ],
    "analysis_quality": [
        {
            "prompt": "Compare React, Angular, and Vue frameworks",
            "validation_criteria": [
                "Structured comparison",
                "Multiple aspects covered",
                "Balanced perspective",
                "Clear recommendations"
            ]
        }
    ],
    "handling_uncertainty": [
        {
            "prompt": "What will be the impact of quantum computing on cryptography?",
            "validation_criteria": [
                "Acknowledges uncertainties",
                "Provides current understanding",
                "Mentions different viewpoints",
                "Suggests areas for monitoring"
            ]
        }
    ],
    "data_synthesis": [
        {
            "prompt": "What are the current trends in AI adoption across industries?",
            "validation_criteria": [
                "Statistical data included",
                "Time periods specified",
                "Industry-specific insights",
                "Trend analysis provided"
            ]
        }
    ],
    "technical_communication": [
        {
            "prompt": "Explain how blockchain works to a non-technical person",
            "validation_criteria": [
                "Uses analogies",
                "Avoids jargon",
                "Progressive complexity",
                "Practical examples"
            ]
        }
    ]
}

def validate_researcher_response(prompt_type, response):
    """
    Validates researcher agent responses against defined criteria
    
    Args:
        prompt_type (str): Type of test being validated
        response (str): Agent's response
        
    Returns:
        dict: Validation results with scores and feedback
    """
    criteria = RESEARCHER_VALIDATION_TESTS[prompt_type][0]["validation_criteria"]
    results = {
        "criteria_met": [],
        "criteria_missed": [],
        "score": 0
    }
    
    # Add validation logic here
    
    return results 