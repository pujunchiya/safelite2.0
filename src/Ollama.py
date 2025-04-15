import requests
import json

def call_deepseek(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "model": "deepseek-r1:latest",  # Ensure this matches your model
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an exception for bad status codes

        result = response.json()
        return result['response']

    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None

if __name__ == "__main__":
    user_prompt = "Write a short Python function to calculate the factorial of a number."
    deepseek_response = call_deepseek(user_prompt)

    if deepseek_response:
        print("DeepSeek's Response:")
        print(deepseek_response)