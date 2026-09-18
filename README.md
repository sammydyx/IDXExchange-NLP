# IDXExchange-NLP

A 12-week Natural Language Processing project for real estate listing intelligence.

## Week 0: Environment Setup

This project uses Python, Docker, MySQL, and pytest.

### Start the local environment

```bash
source .venv/bin/activate
docker compose up -d
pytest -q tests/test_setup.py
