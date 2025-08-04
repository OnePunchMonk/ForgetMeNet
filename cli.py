import click
import requests
import json
import sys
import pandas as pd

API_URL = "http://127.0.0.1:8000"

@click.group()
def cli():
    """A CLI to interact with the Verifiable Transformer Unlearning System."""
    pass

@cli.command()
def status():
    """Checks the status of the backend API and its services."""
    try:
        response = requests.get(f"{API_URL}/status", timeout=600)
        response.raise_for_status()
        click.echo("✅ API is online.")
        click.echo(json.dumps(response.json(), indent=2))
    except requests.exceptions.RequestException as e:
        click.echo(f"❌ Could not connect to the API at {API_URL}. Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--csv-path', required=True, type=click.Path(exists=True), help='Path to the CSV file for training.')
def train(csv_path: str):
    """Sends a CSV to the backend to train the sharded models and create a benchmark set."""
    click.echo(f"Reading data from {csv_path} and sending to backend for training...")
    try:
        df = pd.read_csv(csv_path)
        payload = {"data": df.to_dict('records')}
        response = requests.post(f"{API_URL}/train", json=payload, timeout=600)
        response.raise_for_status()
        click.echo("✅ " + response.json()['message'])
        click.echo("Monitor the backend server console to see the training progress.")
    except Exception as e:
        click.echo(f"❌ An error occurred: {e}", err=True)
        sys.exit(1)

@cli.command()
def benchmark():
    """Triggers the benchmark accuracy test on the backend."""
    click.echo("📊 Requesting benchmark from the backend...")
    try:
        response = requests.get(f"{API_URL}/benchmark", timeout=600)
        response.raise_for_status()
        click.echo("--- ✅ Benchmark Successful ---")
        click.echo(json.dumps(response.json(), indent=2))
    except Exception as e:
        click.echo(f"❌ An error occurred: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--record-id', required=True, help='The unique ID of the record.')
@click.option('--text', required=True, help='The feedback text to analyze.')
@click.option('--full-record-json', required=True, help='The full data record as a JSON string.')
def process(record_id: str, text: str, full_record_json: str):
    """Sends a single record to the API for unlearning certification."""
    try:
        full_record = json.loads(full_record_json)
    except json.JSONDecodeError:
        click.echo("Error: --full-record-json is not a valid JSON string.", err=True)
        sys.exit(1)

    payload = { "id": record_id, "text": text, "full_record": full_record }
    click.echo(f"Sending request for record '{record_id}'...")

    try:
        # Using a long timeout because retraining a Transformer shard can be slow
        response = requests.post(f"{API_URL}/certify-unlearning", json=payload, timeout=300) 
        response.raise_for_status()
        click.echo("\n--- ✅ Request Successful ---")
        click.echo(json.dumps(response.json(), indent=2))
    except requests.exceptions.HTTPError as e:
        click.echo(f"\n--- ❌ Request Failed ---", err=True)
        click.echo(f"Status Code: {e.response.status_code}", err=True)
        click.echo(f"Response: {e.response.text}", err=True)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        click.echo(f"\n❌ Could not connect to the API. Error: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    cli()