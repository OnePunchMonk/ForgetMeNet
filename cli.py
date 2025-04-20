# cli.py
import requests
import argparse
import json

API_URL = "http://localhost:8000/unlearn/"

def send_unlearn_request(text):
    payload = {"text": text}
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        print("✅ Certification Result:")
        print(json.dumps(result, indent=2))
    else:
        print("❌ Error:", response.status_code, response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unlearn a text sample via REST API.")
    parser.add_argument("text", type=str, help="Text to unlearn")

    args = parser.parse_args()
    send_unlearn_request(args.text)
