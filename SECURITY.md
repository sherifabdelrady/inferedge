# Security Policy

## Supported Versions

This repository is a research and portfolio project. Security patches are applied to the latest version only.

| Version | Supported |
|---------|-----------|
| latest (main) | :white_check_mark: |
| older tags | :x: |

## Reporting a Vulnerability

If you discover a security vulnerability, **please do not open a public GitHub issue**.

Instead, report it privately:

- **Email:** sherifabdelrady@gmail.com
- **Subject:** [SECURITY] inferedge — brief description

I will acknowledge your report within 48 hours and aim to release a fix within 14 days for confirmed vulnerabilities.

## Scope

This project contains research code and ML training pipelines. The primary security concerns are:

- Unsafe model loading (pickle deserialization) — use only trusted checkpoint files
- Dependency vulnerabilities — pin versions in `requirements.txt` and audit regularly
- Data poisoning — validate training data sources before use

## Out of Scope

- Vulnerabilities in third-party libraries (report to the respective project)
- Issues requiring physical access to infrastructure
- Social engineering attacks
