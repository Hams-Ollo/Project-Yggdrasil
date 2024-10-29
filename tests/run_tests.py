import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.main import run_multi_agent
from test_prompts import get_test_prompts_by_category
import time

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
