import click
import requests
import json
import sys

API_URL = "http://127.0.0.1:8000"
CERTIFY_ENDPOINT = f"{API_URL}/certify-unlearning"

@click.group()
def cli():
    """
    A Command-Line Interface to interact with the 
    Financial Unlearning Certification System API.
    
    Make sure the FastAPI server is running before using this CLI.
    """
    pass

@cli.command()
def status():
    """Checks the status of the backend API and its services."""
    try:
        response = requests.get(f"{API_URL}/status", timeout=3)
        response.raise_for_status()
        click.echo("✅ API is online.")
        click.echo(json.dumps(response.json()['services'], indent=2))
    except requests.exceptions.RequestException as e:
        click.echo(f"❌ Could not connect to the API at {API_URL}. Is it running?", err=True)
        sys.exit(1)


@cli.command()
@click.option('--record-id', required=True, help='The unique ID of the record (e.g., account number).')
@click.option('--text', required=True, help='The feedback text to analyze.')
@click.option('--full-record-json', required=True, help='The full data record as a JSON string.')
def process(record_id: str, text: str, full_record_json: str):
    """
    Sends a single record to the API for unlearning certification.

    Example:
    
    python cli.py process --record-id "CLI-001" --text "This service is slow and unresponsive." --full-record-json '{"account_number": "CLI-001", "account_name": "CLITest"}'
    """
    try:
        # Validate that the input is valid JSON
        full_record = json.loads(full_record_json)
    except json.JSONDecodeError:
        click.echo("Error: --full-record-json is not a valid JSON string.", err=True)
        sys.exit(1)

    payload = {
        "id": record_id,
        "text": text,
        "full_record": full_record
    }

    click.echo(f"Sending request for record '{record_id}' to {CERTIFY_ENDPOINT}...")

    try:
        response = requests.post(CERTIFY_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        click.echo("\n--- ✅ Request Successful ---")
        click.echo(json.dumps(response.json(), indent=2))

    except requests.exceptions.HTTPError as e:
        click.echo(f"\n--- ❌ Request Failed (HTTP Error) ---", err=True)
        click.echo(f"Status Code: {e.response.status_code}", err=True)
        click.echo(f"Response: {e.response.text}", err=True)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        click.echo(f"\n❌ Could not connect to the API at {API_URL}. Is it running?", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()

