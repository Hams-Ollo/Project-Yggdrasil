import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.main import run_multi_agent
from test_prompts import get_test_prompts_by_category
import time

RESEARCHER_VALIDATION_TESTS = {
    "factual": [{
        "prompt": "What is the capital of France?",
        "validation_criteria": ["Contains Paris", "Mentions it's the capital"]
    }],
    # Add other test types as needed
}

def validate_researcher_response(prompt_type, response):
    """Validates researcher response against test criteria"""
    test_case = RESEARCHER_VALIDATION_TESTS[prompt_type][0]
    criteria_met = []
    criteria_missed = []
    
    for criterion in test_case['validation_criteria']:
        if criterion.lower() in response.lower():
            criteria_met.append(criterion)
        else:
            criteria_missed.append(criterion)
            
    return {
        'criteria_met': criteria_met,
        'criteria_missed': criteria_missed,
        'score': len(criteria_met) / len(test_case['validation_criteria'])
    }

def run_test_prompts(category=None):
    """
    Run test prompts and log results
    
    Args:
        category (str, optional): Specific category to test. If None, runs all categories.
    """
    prompts = get_test_prompts_by_category()
    
    if category:
        if category not in prompts:
            print(f"Invalid category: {category}")
            return
        categories = {category: prompts[category]}
    else:
        categories = prompts

    results = []
    
    for cat, prompt_list in categories.items():
        print(f"\n=== Testing {cat.upper()} ===")
        for prompt in prompt_list:
            print(f"\nTesting prompt: {prompt}")
            try:
                start_time = time.time()
                response = run_multi_agent(prompt)
                end_time = time.time()
                
                results.append({
                    "category": cat,
                    "prompt": prompt,
                    "response": response,
                    "time": end_time - start_time,
                    "status": "success"
                })
                
                print(f"Response: {response[:100]}...")
                print(f"Time taken: {end_time - start_time:.2f}s")
                
            except Exception as e:
                results.append({
                    "category": cat,
                    "prompt": prompt,
                    "error": str(e),
                    "status": "failed"
                })
                print(f"Error: {str(e)}")
            
            time.sleep(1)  # Prevent rate limiting
    
    return results

def run_researcher_validation(prompt_type):
    """
    Runs validation tests for researcher agent
    
    Args:
        prompt_type (str): Type of test to run
    """
    test_case = RESEARCHER_VALIDATION_TESTS[prompt_type][0]
    response = run_multi_agent(test_case["prompt"])
    
    results = validate_researcher_response(prompt_type, response)
    
    print(f"\n=== Researcher Validation Results for {prompt_type} ===")
    print(f"Prompt: {test_case['prompt']}")
    print(f"Criteria Met: {len(results['criteria_met'])}/{len(test_case['validation_criteria'])}")
    print("Met Criteria:", "\n- ".join(results['criteria_met']))
    print("Missed Criteria:", "\n- ".join(results['criteria_missed']))
    print(f"Overall Score: {results['score']}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run test prompts for multi-agent system')
    parser.add_argument('--category', type=str, help='Specific category to test')
    
    args = parser.parse_args()
    results = run_test_prompts(args.category)
    
    # Print summary
    success_count = len([r for r in results if r["status"] == "success"])
    print(f"\n=== Test Summary ===")
    print(f"Total tests: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
