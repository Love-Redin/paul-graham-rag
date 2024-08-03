import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ast
import json
import os


# Purpose of file: fetch stored data, process, create embeddings and store in csv file

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv('data/articles.csv')

# Function to get embedding for a single paragraph
def get_embedding(paragraph):
    embedding = model.encode([paragraph])[0]
    return embedding


 # Create a list to store the paragraph data
paragraph_data = []

# Process each article and create embeddings
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    article_name = row['article']
    article_url = row['url']
    text = row['text']
    paragraphs = ast.literal_eval(text)
    
    for i, paragraph in enumerate(paragraphs):
        paragraph_text_cleaned = paragraph.replace('\n', ' ').replace('\r', ' ').strip()
        embedding = get_embedding(paragraph_text_cleaned)
        embedding_str = json.dumps(embedding.tolist())
        paragraph_data.append([article_name, article_url, i + 1, paragraph_text_cleaned, embedding_str])

# Convert the paragraph data to a DataFrame
paragraph_df = pd.DataFrame(paragraph_data, columns=['article_name', 'article_url', 'paragraph_number', 'paragraph_text', 'embedding'])

# Save the paragraph DataFrame to a CSV file
paragraph_df.to_csv(os.path.join('data/paragraph_embeddings.csv'), index=False, sep=";")