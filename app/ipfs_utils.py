# app/ipfs_utils.py
import ipfshttpclient

client = ipfshttpclient.connect()

def upload_text(text: str) -> str:
    return client.add_str(text)

def upload_json(data: dict) -> str:
    import json
    return client.add_str(json.dumps(data, indent=2))

# from ipfs_api_py import Client

# client = Client("http://127.0.0.1:5001")
# info = client.version()
# print(info)

