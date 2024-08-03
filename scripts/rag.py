import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ast
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import sys

# Add the parent directory of `modules` to the Python path
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#from modules.data_processing import load_data, get_top_n_similar_paragraphs, get_top_n_similar_paragraphs_including_context, get_embedding
#from modules.query_rewriter import rewrite_query


load_dotenv()
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Purpose of file: create query, rewrite query to RAG-friendly format
# Embed query and find top N similar paragraphs
# Then, use the similar paragraphs as input to the RAG model to generate answers

# Load the paragraph embeddings
paragraph_df = pd.read_csv('data/paragraph_embeddings.csv', sep=";")




# json load embeddings
paragraph_df['embedding'] = paragraph_df['embedding'].apply(lambda x: np.array(json.loads(x)))


#paragraph_df = load_data(os.path.join('..', 'data', 'paragraph_embeddings.csv'))

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


# Function to get embedding for a single paragraph
def get_embedding(text):
    embedding = model.encode([text])[0]
    return embedding

# Function to rewrite the query in a RAG-friendly format
def rewrite_query(query):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful and knowledgeable assistant."},
        {"role": "user", "content": f"Rewrite the query in a more informative way, perfect for Retrial Augmented Generation (RAG) models.\nQ: {query}\nRAG query: "},
    ]
    )
    return completion.choices[0].message.content

# Function to get the top N most similar paragraphs
def get_top_n_similar_paragraphs(query, n=10):
    query_embedding = get_embedding(query)
    paragraph_df['similarity'] = paragraph_df['embedding'].apply(lambda x: np.dot(x, query_embedding))
    top_n_similar_paragraphs = paragraph_df.sort_values(by='similarity', ascending=False).head(n)['paragraph_text'].tolist()
    return top_n_similar_paragraphs

def get_top_n_similar_paragraphs_including_context(query, n=10):
    # Like above, but include the paragraph before and after the most similar paragraphs
    # For the top 3, include three paragraphs before and after
    # For the next 3, include two paragraphs before and after
    # For the rest, include one paragraph before and after
    query_embedding = get_embedding(query)
    paragraph_df['similarity'] = paragraph_df['embedding'].apply(lambda x: np.dot(x, query_embedding))
    top_n_similar_paragraphs = paragraph_df.sort_values(by='similarity', ascending=False).head(n).reset_index(drop=True)
    top_paragraphs = []
    for index, row in top_n_similar_paragraphs.iterrows():
        paragraph_text = row['paragraph_text']
        paragraph_number = row['paragraph_number']
        article_name = row['article_name']
        article_url = row['article_url']
        similarity = row['similarity']
        if index > 5:
            if paragraph_number > 1:
                previous_paragraph = paragraph_df[(paragraph_df['article_name'] == article_name) & (paragraph_df['paragraph_number'] == paragraph_number - 1)]['paragraph_text'].values[0]
                if pd.isna(previous_paragraph):
                    next_paragraph = ""
            else:
                previous_paragraph = ""
            if paragraph_number < paragraph_df[paragraph_df['article_name'] == article_name]['paragraph_number'].max():
                next_paragraph = paragraph_df[(paragraph_df['article_name'] == article_name) & (paragraph_df['paragraph_number'] == paragraph_number + 1)]['paragraph_text'].values[0]
                if pd.isna(next_paragraph):
                    next_paragraph = ""
            else:
                next_paragraph = ""
            full_paragraph_text = str(previous_paragraph) + " ---> " + str(paragraph_text) + " ---> " + str(next_paragraph)
        elif index <= 5 and index > 2:
            if paragraph_number > 2:
                previous_paragraph = paragraph_df[(paragraph_df['article_name'] == article_name) & (paragraph_df['paragraph_number'] == paragraph_number - 2)]['paragraph_text'].values[0]
                if pd.isna(previous_paragraph):
                    previous_paragraph = ""
            else:
                previous_paragraph = ""

            if paragraph_number > 1:
                previous_paragraph_1 = paragraph_df[(paragraph_df['article_name'] == article_name) & (paragraph_df['paragraph_number'] == paragraph_number - 1)]['paragraph_text'].values[0]
                if pd.isna(previous_paragraph_1):
                    previous_paragraph_1 = ""
            else:
                previous_paragraph_1 = ""
            if paragraph_number < paragraph_df[paragraph_df['article_name'] == article_name]['paragraph_number'].max():
                next_paragraph = paragraph_df[(paragraph_df['article_name'] == article_name) & (paragraph_df['paragraph_number'] == paragraph_number + 1)]['paragraph_text'].values[0]
                if pd.isna(next_paragraph):
                    next_paragraph = ""
            else:
                next_paragraph = ""

            if paragraph_number < paragraph_df[paragraph_df['article_name'] == article_name]['paragraph_number'].max() - 1:
                next_paragraph_1 = paragraph_df[(paragraph_df['article_name'] == article_name) & (paragraph_df['paragraph_number'] == paragraph_number + 2)]['paragraph_text'].values[0]
                if pd.isna(next_paragraph_1):
                    next_paragraph_1 = ""
            else:
                next_paragraph_1 = ""

            full_paragraph_text = str(previous_paragraph) + " -1--> " + str(previous_paragraph_1) + " -2--> " + str(paragraph_text) + " -3--> " + str(next_paragraph) + " -4--> " + str(next_paragraph_1)
        elif index <= 2:
            if paragraph_number > 3:
                previous_paragraph = paragraph_df[(paragraph_df['article_name'] == article_name) & (paragraph_df['paragraph_number'] == paragraph_number - 3)]['paragraph_text'].values[0]
                if pd.isna(previous_paragraph):
                    previous_paragraph = ""
            else:
                previous_paragraph = ""
            if paragraph_number > 2:
                previous_paragraph_1 = paragraph_df[(paragraph_df['article_name'] == article_name) & (paragraph_df['paragraph_number'] == paragraph_number - 2)]['paragraph_text'].values[0]
                if pd.isna(previous_paragraph_1):
                    previous_paragraph_1 = ""
            else:
                previous_paragraph_1 = ""
            if paragraph_number > 1:
                previous_paragraph_2 = paragraph_df[(paragraph_df['article_name'] == article_name) & (paragraph_df['paragraph_number'] == paragraph_number - 1)]['paragraph_text'].values[0]
                if pd.isna(previous_paragraph_2):
                    previous_paragraph_2 = ""
            else:
                previous_paragraph_2 = ""
            if paragraph_number < paragraph_df[paragraph_df['article_name'] == article_name]['paragraph_number'].max():
                next_paragraph = paragraph_df[(paragraph_df['article_name'] == article_name) & (paragraph_df['paragraph_number'] == paragraph_number + 1)]['paragraph_text'].values[0]
                if pd.isna(next_paragraph):
                    next_paragraph = ""
            else:
                next_paragraph = ""
            if paragraph_number < paragraph_df[paragraph_df['article_name'] == article_name]['paragraph_number'].max() - 1:
                next_paragraph_1 = paragraph_df[(paragraph_df['article_name'] == article_name) & (paragraph_df['paragraph_number'] == paragraph_number + 2)]['paragraph_text'].values[0]
                if pd.isna(next_paragraph_1):
                    next_paragraph_1 = ""
            else:
                next_paragraph_1 = ""
            if paragraph_number < paragraph_df[paragraph_df['article_name'] == article_name]['paragraph_number'].max() - 2:
                next_paragraph_2 = paragraph_df[(paragraph_df['article_name'] == article_name) & (paragraph_df['paragraph_number'] == paragraph_number + 3)]['paragraph_text'].values[0]
                if pd.isna(next_paragraph_2):
                    next_paragraph_2 = ""
            else:
                next_paragraph_2 = ""
            full_paragraph_text = str(previous_paragraph) + " -1--> " + str(previous_paragraph_1) + " -2--> " + str(previous_paragraph_2) + " -3--> " + str(paragraph_text) + " -4--> " + str(next_paragraph) + " -5--> " + str(next_paragraph_1) + " -6--> " + str(next_paragraph_2)
        full_paragraph_text = f"{article_name}: {similarity} (top {index+1})" + "\n" + full_paragraph_text
        top_paragraphs.append(full_paragraph_text) 
    return top_paragraphs


query = "What is the best programming language to learn first?"
query = "How much money should founders raise in their first round of funding?"
query = "What is the best way to find a co-founder for a startup?"
query = "What is the best way to build a successful startup?"
rag_query = rewrite_query(query)
top_paragraphs = get_top_n_similar_paragraphs_including_context(query)
print(query, "--->", rag_query)
for p in top_paragraphs:
    print(p)
    print("-"*50)


# Function to answer the question
def answer_question(query, top_paragraphs):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful and knowledgeable RAG agent assistant."},
        {"role": "user", "content": f"Answer the query: {query}\nUse the following paragraphs from Paul Graham's essays to generate the answer: {"\n\n".join(top_paragraphs)}\nAnswer: "},
    ]
    )
    return completion.choices[0].message.content

answer = answer_question(query, top_paragraphs)
print(query)
print(answer)

# TODO (senare): fixa källhänvisningar i svaret