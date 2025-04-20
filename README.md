# Unlearning Certification System

## Overview
This project implements a **Unlearning Certification System** that utilizes **BERT-based sentiment analysis** to certify the unlearning of data. The system integrates with **IPFS** for data storage and provides a **REST API** interface for triggering unlearning and logging certification. The project ensures that data can be unlearned and certified through a decentralized approach using IPFS, enhancing data privacy and integrity.

## Purpose
The purpose of this system is to provide a framework for certifying the unlearning of data, particularly useful in scenarios where data privacy is critical. The system uses a **sentiment analysis model** (based on BERT) to evaluate whether the data being "unlearned" has been processed correctly and if the certification can be logged and stored securely.

## Features
- **BERT-based Sentiment Classification**: Uses BERT for sentiment classification to ensure the model processes textual data appropriately.
- **Unlearning Triggering**: Allows for the triggering of unlearning operations through the REST API.
- **IPFS Integration**: Data is uploaded to IPFS for decentralized storage, ensuring that data cannot be tampered with.
- **Logging Certification**: The system logs certification of unlearning in a secure, immutable manner.


### IPFS Daemon
Make sure the IPFS daemon is installed and running locally:

1. **Install IPFS**: Follow the instructions on [IPFS installation guide](https://ipfs.io/docs/install/) to install and set up IPFS on your machine.
2. **Start the IPFS daemon**:
ipfs daemon

### API Server
Run the FastAPI server using **Uvicorn**:
uvicorn app.api:app --reload


This will start the server at `http://127.0.0.1:8000`. You can then make API requests to trigger unlearning and certification processes.

## How to Run

### 1. **Run the API Server**

To start the FastAPI server and begin using the unlearning certification system:

uvicorn app.api:app --reload

This will start the server on `http://localhost:8000`.

### 2. **Interact with the API**

You can interact with the API through **REST endpoints**. The following endpoints are available:

- **POST /certify-unlearning**
  - Trigger the unlearning certification process.
  - Example:
    ```bash
    curl -X 'POST' 'http://127.0.0.1:8000/certify-unlearning' -H 'Content-Type: application/json' -d '{"text": "Your data here"}'
    ```

- **GET /status**
  - Get the current status of the IPFS node.
  - Example:
    ```bash
    curl http://127.0.0.1:8000/status
    ```

### 3. **Testing the System**

You can also use the system to test unlearning and certification functionalities with a sample dataset. Here's how to trigger the unlearning process for a sample entry:

curl -X 'POST' 'http://127.0.0.1:8000/certify-unlearning' -H 'Content-Type: application/json' -d '{"text": "This is a test sentence for unlearning certification."}'


## Approach

### 1. **Sentiment Analysis with BERT**:
   - The system uses a **pre-trained BERT model** to classify the sentiment of the data being processed. This ensures that the system only certifies unlearning for valid entries.

### 2. **IPFS Integration**:
   - After certification, data is stored in **IPFS** (InterPlanetary File System) for decentralization and immutability. This ensures that once the data is "unlearned" and certified, it is securely stored and cannot be altered.
   
### 3. **Logging and Tracking**:
   - The certification logs are stored to track the unlearning process. This ensures accountability and can be used for auditing purposes.

### 4. **Unlearning Process**:
   - The unlearning process is initiated via a REST API, triggering a sequence that involves data validation, sentiment classification, IPFS upload, and logging of the certification.

## Future Improvements
- **Scalability**: Optimizing the system for handling larger datasets and enabling parallel processing.
- **Enhanced Security**: Incorporating encryption for added data privacy during the unlearning and certification processes.
- **Extended API Capabilities**: Adding more endpoints for advanced data queries and status updates.

## Conclusion
This Unlearning Certification System provides a robust solution for certifying the unlearning of data using sentiment analysis, IPFS, and secure logging. It offers a decentralized and immutable approach to managing sensitive data in line with modern data privacy standards.

