# Test prompts for Multi-Agent Assistant validation

SUPERVISOR_TEST_PROMPTS = [
    # Edge cases to test supervisor routing
    "Can you help me with something?",  # Test general conversation
    "I'm not sure which agent I need",  # Test ambiguous requests
    "Hello!",  # Test basic greeting
    "Both write code and explain quantum computing",  # Test mixed requests
]

RESEARCHER_TEST_PROMPTS = [
    # Information gathering
    "What are the latest developments in quantum computing?",
    "Explain the difference between AI and machine learning",
    "What are the environmental impacts of electric vehicles?",
    "Research the history of blockchain technology",
    
    # Analysis requests
    "Compare and contrast React and Angular frameworks",
    "Analyze the trends in renewable energy adoption",
    "What are the pros and cons of different Python web frameworks?",
    
    # Fact-checking
    "What is the current world population?",
    "What are the most used programming languages in 2024?",
    "What is the market share of different cloud providers?",
]

WRITER_TEST_PROMPTS = [
    # Content creation
    "Write a blog post about the future of AI",
    "Create a product description for a smart home device",
    "Write a tutorial on getting started with Python",
    "Draft an email announcing a new software release",
    
    # Editing and formatting
    "Help me structure a technical documentation outline",
    "Write a professional LinkedIn post about AI trends",
    "Create a user guide for a mobile app",
    
    # Creative writing
    "Write a story about a programmer in the year 2050",
    "Create a metaphor explaining how blockchain works",
    "Write a catchy slogan for a tech startup",
]

CODER_TEST_PROMPTS = [
    # Basic coding tasks
    "Write a Python function to reverse a string",
    "Show me how to implement a binary search in JavaScript",
    "Create a simple REST API using FastAPI",
    
    # Debugging scenarios
    "Debug this code: print(sorted([1,2,3,None,'4'])",
    "Why am I getting a KeyError in my Python dictionary?",
    "Fix this JavaScript promise chain",
    
    # Implementation requests
    "How do I implement authentication in a Flask app?",
    "Create a React component for a todo list",
    "Write a Python script to process CSV files",
    
    # Code explanation
    "Explain how async/await works in Python",
    "What's the difference between map, filter, and reduce?",
    "How does garbage collection work in Python?",
]

COMPLEX_TEST_PROMPTS = [
    # Multi-step problems
    "Research machine learning frameworks, then write code for a simple neural network",
    "Analyze cloud providers and write a deployment script for AWS",
    "Research microservices architecture and implement a basic service in Python",
    
    # Edge cases
    "Can you both write code and analyze its performance?",
    "I need research on databases and then code to implement one",
    "Write documentation for this code: def foo(x): return x*2",
]

def get_all_test_prompts():
    """Return all test prompts in a flat list"""
    return (
        SUPERVISOR_TEST_PROMPTS +
        RESEARCHER_TEST_PROMPTS +
        WRITER_TEST_PROMPTS +
        CODER_TEST_PROMPTS +
        COMPLEX_TEST_PROMPTS
    )

def get_test_prompts_by_category():
    """Return test prompts organized by category"""
    return {
        "supervisor": SUPERVISOR_TEST_PROMPTS,
        "researcher": RESEARCHER_TEST_PROMPTS,
        "writer": WRITER_TEST_PROMPTS,
        "coder": CODER_TEST_PROMPTS,
        "complex": COMPLEX_TEST_PROMPTS
    }

# Example usage:
if __name__ == "__main__":
    print("Total number of test prompts:", len(get_all_test_prompts()))
    for category, prompts in get_test_prompts_by_category().items():
        print(f"\n{category.upper()} Tests ({len(prompts)} prompts):")
        for prompt in prompts:
            print(f"- {prompt}")
