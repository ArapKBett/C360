# CyberGuard360

An all-in-one cybersecurity tool with URL detection, network scanning, file integrity checking, password strength analysis, and vulnerability scanning.

## Features
- **Malicious URL Detector**: Uses a pre-trained BERT model with JAX.
- **Network Scanner**: Scans open ports and services using nmap.
- **File Integrity Checker**: Monitors file changes with SHA-256 hashing.
- **Password Strength Analyzer**: Evaluates passwords using zxcvbn.
- **Vulnerability Scanner**: Checks for common web vulnerabilities.

## Setup

1. Clone the repository:
   `git clone https://github.com/ArapKBett/C360.git
   cd C360`

   **Install dependencies**
   `pip install -r requirements.txt`

   **Run locally**
   `flask --app app run`

   **Deployment on Render**
   Create a Render account and set up a new web service.
   Connect your GitHub repository.
   Use the provided render.yaml for configuration.
   Deploy and access your app via the Render-provided URL.


