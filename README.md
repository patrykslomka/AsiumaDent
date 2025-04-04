# AsiumaDent

A Flask web app for analyzing dental X-rays using a PyTorch model, with bounding box annotations, dental ontology, and optional Claude AI reports.

## Features
- Upload and analyze dental X-rays
- Annotated images with condition labels
- Detailed condition info via ontology
- AI-generated reports (Claude API)

## Setup
1. Clone: `git clone https://github.com/patrykslomka/AsiumaDent.git`
2. Install: `pip install -r requirements.txt`
3. Add model files: `models/best_model.pth`, `models/class_names.json`
4. Set `.env`: `ANTHROPIC_API_KEY=your_key`
5. Run: `python app.py`

## Usage
- Visit `http://localhost:5000`
- Upload an X-ray (max 16MB)
- View predictions and reports

## Requirements
- Python 3.11+
- PyTorch, Flask, Pillow, NumPy
- Anthropic API key (optional)

## Troubleshooting
- **"Invalid load key"**: Ensure `best_model.pth` is a valid PyTorch file.
- **No report**: Check `ANTHROPIC_API_KEY`.
