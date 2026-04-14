import json
import urllib.request

def get_ollama_response(prompt, model="minimax-m2.5:cloud"):
    """
    Sends a prompt to the local Ollama API and returns the response.
    """
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 150,
            "temperature": 0.2
        }
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            return res_data.get("response", "")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return None

async def get_ollama_response_async(prompt, model="minimax-m2.5:cloud"):
    """
    Async version of the Ollama response function.
    Since we are using urllib (synchronous), we'll run it in a thread to keep it non-blocking.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_ollama_response, prompt, model)
