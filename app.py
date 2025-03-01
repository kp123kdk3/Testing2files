from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from transformer import GPT
from tokenizer import SimpleTokenizer
import requests
from bs4 import BeautifulSoup
import wikipedia
import scholarly

app = Flask(__name__)
CORS(app)

# Initialize model and tokenizer
tokenizer = SimpleTokenizer(10000)
model = GPT(10000)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        web_search = data.get('web_search', False)
        model_type = data.get('model', 'general')
        
        response_text = ""
        
        if web_search:
            # Perform web search
            search_results = search_web(prompt)
            response_text = f"Web Search Results:\n{search_results}"
        else:
            # Use appropriate model based on type
            if model_type == 'research':
                response_text = deep_research(prompt)
            else:
                # Basic response for demo
                response_text = f"Response to: {prompt}\nThis is a demo response. Replace with actual AI model output."
        
        return jsonify({
            'generated_text': response_text
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def search_web(query):
    url = f'https://html.duckduckgo.com/html/?q={query}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        for result in soup.select('.result__body')[:3]:
            title = result.select_one('.result__title').get_text(strip=True)
            snippet = result.select_one('.result__snippet').get_text(strip=True)
            results.append(f"{title}\n{snippet}\n")
        
        return "\n".join(results)
    except Exception as e:
        return f"Error searching web: {str(e)}"

def deep_research(topic):
    results = []
    
    try:
        # Get Wikipedia summary
        wiki_summary = wikipedia.summary(topic, sentences=2)
        results.append(f"Wikipedia:\n{wiki_summary}\n")
    except:
        pass
    
    # Add web search results
    web_results = search_web(topic)
    if web_results:
        results.append(f"Web Results:\n{web_results}")
    
    return "\n".join(results)

if __name__ == '__main__':
    app.run(debug=True) 