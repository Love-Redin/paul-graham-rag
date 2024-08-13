from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
from modules.data_processing import get_top_n_similar_paragraphs_including_context
from modules.query_rewriter import rewrite_query

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI API client
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the Flask app
app = Flask(__name__)

# Load data
paragraph_df = pd.read_csv('data/paragraph_embeddings.csv', sep=";")
paragraph_df['embedding'] = paragraph_df['embedding'].apply(lambda x: np.array(json.loads(x)))

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to answer the question
def answer_question(query, top_paragraphs):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful and knowledgeable RAG agent assistant."},
        {"role": "user", "content": f"Answer the query: {query}\nUse the following paragraphs from Paul Graham's essays to generate the answer: {'\n\n'.join(top_paragraphs)}\nProvide a concise, concrete and assertive answer in the tone of Paul Graham.\nAnswer: "},
    ]
    )
    return completion.choices[0].message.content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form['query']
    rag_query = rewrite_query(user_query, client)
    top_paragraphs = get_top_n_similar_paragraphs_including_context(rag_query, model, paragraph_df)
    answer = answer_question(user_query, top_paragraphs)
    print(answer)
    
    # For simplicity, let's return the top paragraphs as a JSON response
    # top_paragraphs is a list of strings, where each string contains the full paragraph text
    results = {'query': user_query, 'rag_query': rag_query, 'top_paragraphs': top_paragraphs, 'answer': answer}
    return jsonify(results)

if __name__ == '__main__':
    # Local deployment
    #app.run(debug=True, port=7777)
    
    # Render deployment
    # Get the port from the environment variable, or default to 7777
    port = int(os.environ.get('PORT', 7777))
    app.run(debug=True, host='0.0.0.0', port=port)