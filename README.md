# Neural Network Chess Bot

A chess bot powered by neural networks, built with Python.

## Setup

### Prerequisites
- Python 3.9 or higher
- pip

### Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. Install dependencies:
```bash
pip install -e .
```

Or install with development dependencies:
```bash
pip install -e ".[dev]"
```

## Project Structure

```
nn-chess-bot/
├── src/              # Source code
├── tests/            # Test files
├── models/           # Saved neural network models
├── data/             # Training data
└── requirements.txt  # Dependencies (alternative to pyproject.toml)
```

## Dependencies

- **python-chess**: Chess board representation and move generation
- **NumPy**: Numerical computing

