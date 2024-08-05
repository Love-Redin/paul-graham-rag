import streamlit as st
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
from modules.data_processing import get_top_n_similar_paragraphs_including_context
from modules.query_rewriter import rewrite_query

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI API client
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        {"role": "user", 
         "content": f"""Answer the query: {query}
         Use the following paragraphs from Paul Graham's essays to generate the answer: {'\\n\\n'.join(top_paragraphs)}
         Provide a concise, concrete and assertive answer in the tone of Paul Graham.
         Answer: """},
    ]
    )
    return completion.choices[0].message.content

# Streamlit UI
st.title('Paul Graham RAG')
st.write('Access the distilled insights from Paul Graham through a RAG application based on PG\'s essays')

# Input form
query = st.text_input('Enter your query:', '')

if st.button('Submit'):
    if query:
        # Process the query and display results
        rag_query = rewrite_query(query, client)
        top_paragraphs = get_top_n_similar_paragraphs_including_context(rag_query, model, paragraph_df)
        answer = answer_question(query, top_paragraphs)

        st.write('### Answer')
        st.write(answer)
        
        st.write('### RAG query')
        st.write(rag_query)
        
        st.write('### Top paragraphs')
        for i, paragraph in enumerate(top_paragraphs):
            formatted_paragraph = f"<strong>Match #{i + 1}</strong><br>{paragraph.replace('\n', '<br>')}".replace('$', '\\$')
            st.markdown(formatted_paragraph, unsafe_allow_html=True)
        
    else:
        st.warning('Please enter a query.')