# minimal_aiohttp_ollama_test.py
import asyncio
import aiohttp
import json

async def test_ollama():
    payload = {
        "model": "gemma3:1b", # Or your desired model
        "messages": [{"role": "user", "content": "Hello Ollama from Python!"}],
        "stream": False
    }
    url = "http://localhost:11434/api/chat"
    print(f"Attempting to POST to {url} with payload: {payload}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                print(f"Status: {response.status}")
                print(f"Headers: {response.headers}")
                response_text = await response.text()
                print(f"Response Text: {response_text[:500]}...") # Print first 500 chars
                response.raise_for_status()
                response_json = json.loads(response_text) # Or await response.json() if sure it's JSON
                print("\nSuccessfully received and parsed JSON response:")
                print(json.dumps(response_json, indent=2))
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama())
