# AI Research Assistant

An AI-powered research assistant that iteratively searches the web, creates summaries, and identifies research gaps to provide comprehensive research results.

## Features

- Web interface built with Flask and TailwindCSS
- Uses LangChain for AI agent orchestration
- Powered by Llama 3.2 for local LLM processing
- Web search capabilities using Tavily API
- Iterative research process with configurable number of iterations
- Automatic identification of research gaps
- Comprehensive summaries with source tracking

## Prerequisites

1. Python 3.8 or higher
2. Llama 3.2 model file (GGUF format)
3. Tavily API key

## Setup

1. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your actual values
# LLAMA_MODEL_PATH: Path to your Llama model file
# TAVILY_API_KEY: Your API key from tavily.com
```

## Usage

1. Start the web server:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. In the web interface:
   - Enter your research topic/problem
   - Set the number of research iterations
   - Click "Start Research"
   - Wait for the results to appear

## Output

The web interface will display:
- A comprehensive final summary
- List of all sources used (with clickable links)
- Summaries from each iteration of research

## Project Structure

```
.
├── app.py              # Flask web application
├── research_agent.py   # Core research agent logic
├── requirements.txt    # Python dependencies
├── templates/         
│   └── index.html     # Web interface template
├── .env               # Environment variables (create from .env.example)
└── .env.example       # Template for environment variables
```

## Note

Make sure you have sufficient compute resources to run Llama 3.2 locally. The model requires significant RAM and processing power. 