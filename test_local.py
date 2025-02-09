# test_local.py
from app.handler import WeightedAdapterHandler
import json

def test_handler():
    handler = WeightedAdapterHandler()
    
    # Test different voice combinations
    test_cases = [
        {
            "name": "Pure Davinci",
            "payload": {
                "inputs": "What is the meaning of life?",
                "voice_weights": {
                    "davinci": 1.0,
                    "davinci-instruct": 0.0,
                    "text-davinci": 0.0,
                    "opus-calm": 0.0,
                    "opus-manic": 0.0
                },
                "max_length": 100,
                "temperature": 0.7,
                "top_p": 0.9
            }
        },
        {
            "name": "Blended Voice",
            "payload": {
                "inputs": "What is the meaning of life?",
                "voice_weights": {
                    "davinci": 0.6,
                    "opus-calm": 0.4,
                    "davinci-instruct": 0.0,
                    "text-davinci": 0.0,
                    "opus-manic": 0.0
                },
                "max_length": 100,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
    ]
    
    for test in test_cases:
        print(f"\nTesting {test['name']}...")
        try:
            response = handler(test['payload'])
            print(f"Input: {test['payload']['inputs']}")
            print(f"Response: {response['generated_text']}")
            print("Test successful!")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_handler()