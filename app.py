from flask import Flask, render_template, request, jsonify, Response, make_response
from flask_cors import CORS
from research_agent import ResearchAgent
import os
from dotenv import load_dotenv
import json
from queue import Queue
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Configure CORS to allow all origins and methods
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

# Initialize the research agent
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not tavily_api_key:
    raise ValueError("Please set TAVILY_API_KEY in .env file")

agent = ResearchAgent(tavily_api_key=tavily_api_key)

# Store progress updates
progress_updates = {}

@app.route('/')
def home():
    response = make_response(render_template('index.html'))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

def generate_progress_events(session_id: str):
    """Generator function for SSE events"""
    if session_id not in progress_updates:
        progress_updates[session_id] = Queue()
    
    while True:
        try:
            update = progress_updates[session_id].get()
            if update is None:  # Signal to stop
                break
            yield f"data: {json.dumps(update)}\n\n"
        except Exception as e:
            print(f"Error in generate_progress_events: {str(e)}")
            break

@app.route('/research', methods=['POST', 'OPTIONS'])
def research():
    if request.method == 'OPTIONS':
        # Respond to preflight request
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    data = request.json
    topic = data.get('topic')
    iterations = int(data.get('iterations', 3))
    word_count = int(data.get('wordCount', 300))
    temperature = float(data.get('temperature', 0.3))
    
    if not topic:
        return jsonify({'error': 'Topic is required'}), 400
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    progress_updates[session_id] = Queue()
    
    def run_research():
        try:
            # Update LLM temperature
            agent.llm.temperature = temperature
            
            # Initialize progress
            progress_updates[session_id].put({
                "status": "started",
                "progress": 0,
                "message": "Starting research..."
            })
            
            all_sources = []
            summaries = []
            current_summary = ""
            current_gaps = ""
            
            for i in range(iterations):
                # Generate search query
                query = agent.generate_query(topic, current_summary, current_gaps)
                progress_updates[session_id].put({
                    "status": "searching",
                    "progress": (i * 100) // iterations,
                    "message": f"Iteration {i+1}/{iterations}: Searching for '{query}'"
                })
                
                # Perform web search
                search_results = agent.search_web(query)
                if search_results:
                    all_sources.extend(search_results)
                    progress_updates[session_id].put({
                        "status": "analyzing",
                        "progress": (i * 100) // iterations + 50//iterations,
                        "message": f"Iteration {i+1}/{iterations}: Analyzing {len(search_results)} results"
                    })
                    
                    # Create summary and identify gaps
                    result = agent.create_summary(search_results, current_summary, word_count)
                    current_summary = result["summary"]
                    current_gaps = result["gaps"]
                    summaries.append({
                        "summary": current_summary,
                        "sources": search_results,
                        "query": query
                    })
                else:
                    progress_updates[session_id].put({
                        "status": "warning",
                        "progress": (i * 100) // iterations + 50//iterations,
                        "message": f"Iteration {i+1}/{iterations}: No results found. Stopping research."
                    })
                    break
            
            # Final result with iteration-specific sources and queries
            progress_updates[session_id].put({
                "status": "complete",
                "progress": 100,
                "message": "Research complete!",
                "data": {
                    "summary": summaries[-1]["summary"] if summaries else "No results found.",
                    "sources": all_sources,
                    "iteration_summaries": [s["summary"] for s in summaries],
                    "iteration_sources": [s["sources"] for s in summaries],
                    "queries": [s["query"] for s in summaries]
                }
            })
            
        except Exception as e:
            print(f"Error in research: {str(e)}")
            progress_updates[session_id].put({
                "status": "error",
                "progress": 0,
                "message": f"Error: {str(e)}"
            })
        finally:
            progress_updates[session_id].put(None)  # Signal end of updates
    
    # Start research in background thread
    from threading import Thread
    Thread(target=run_research).start()
    
    response = jsonify({"session_id": session_id})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/research-progress')
def research_progress():
    session_id = request.args.get('session_id')
    if not session_id or session_id not in progress_updates:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    response = Response(
        generate_progress_events(session_id),
        mimetype='text/event-stream',
        headers={
            'Access-Control-Allow-Origin': '*',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
            'Content-Type': 'text/event-stream'
        }
    )
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5001) 