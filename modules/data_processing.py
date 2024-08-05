import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import re

# Load data from data/ directory
def load_data(filepath):
    df = pd.read_csv(filepath, sep=";")
    df['embedding'] = df['embedding'].apply(lambda x: np.array(json.loads(x)))
    return df

# Function to get embedding for a single paragraph
def get_embedding(text, model):
    embedding = model.encode([text])[0]
    return embedding
"""
# Function to get embedding for a single paragraph
def get_embedding(text):
    embedding = model.encode([text])[0]
    return embedding
"""

# Function to get the top N most similar paragraphs
def get_top_n_similar_paragraphs(query, model, paragraph_df, n=5):
    query_embedding = get_embedding(query, model)
    paragraph_df['similarity'] = paragraph_df['embedding'].apply(lambda x: np.dot(x, query_embedding))
    top_n_similar_paragraphs = paragraph_df.sort_values(by='similarity', ascending=False).head(n)
    return top_n_similar_paragraphs

"""
# Function to get the top N most similar paragraphs
def get_top_n_similar_paragraphs(query, n=5):
    query_embedding = get_embedding(query)
    paragraph_df['similarity'] = paragraph_df['embedding'].apply(lambda x: np.dot(x, query_embedding))
    top_n_similar_paragraphs = paragraph_df.sort_values(by='similarity', ascending=False).head(n)
    return top_n_similar_paragraphs
"""


"""
def get_top_n_similar_paragraphs_including_context(query, model, paragraph_df, n=10):
    # Like above, but include the paragraph before and after the most similar paragraphs
    # For the top 3, include three paragraphs before and after
    # For the next 3, include two paragraphs before and after
    # For the rest, include one paragraph before and after
    query_embedding = get_embedding(query, model)
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
"""

def get_top_n_similar_paragraphs_including_context(query, model, paragraph_df, n=10):
    # Like above, but include the paragraph before and after the most similar paragraphs
    # For the top 3, include three paragraphs before and after
    # For the next 3, include two paragraphs before and after
    # For the rest, include one paragraph before and after
    query_embedding = get_embedding(query, model)
    paragraph_df['similarity'] = paragraph_df['embedding'].apply(lambda x: np.dot(x, query_embedding))
    top_n_similar_paragraphs = paragraph_df.sort_values(by='similarity', ascending=False).head(n).reset_index(drop=True)
    top_paragraphs = []
    for index, row in top_n_similar_paragraphs.iterrows():
        paragraph_text = row['paragraph_text']
        paragraph_number = row['paragraph_number']
        article_name = row['article_name']
        article_url = row['article_url']
        similarity = row['similarity']
        article_date = row['created_at']
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
            full_paragraph_text = str(previous_paragraph) + "\n\n" + str(paragraph_text) + "\n\n" + str(next_paragraph)
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

            full_paragraph_text = str(previous_paragraph) + "\n\n" + str(previous_paragraph_1) + "\n\n" + str(paragraph_text) + "\n\n" + str(next_paragraph) + "\n\n" + str(next_paragraph_1)
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

            if re.match(r'^\d+\.\s.+$', paragraph_text.strip()) or (len(paragraph_text.split()) <= 1): # if last paragraph is just one word (or empty), don't include it # also exclude the format '9. Investors look for founders like the current stars.', which is a new title
                full_paragraph_text = str(previous_paragraph) + "\n\n" + str(previous_paragraph_1) + "\n\n" + str(previous_paragraph_2) + "\n\n" + str(paragraph_text) + "\n\n" + str(next_paragraph) + "\n\n" + str(next_paragraph_1)
            else:
                full_paragraph_text = str(previous_paragraph) + "\n\n" + str(previous_paragraph_1) + "\n\n" + str(previous_paragraph_2) + "\n\n" + str(paragraph_text) + "\n\n" + str(next_paragraph) + "\n\n" + str(next_paragraph_1) + "\n\n" + str(next_paragraph_2)
        full_paragraph_text = full_paragraph_text.strip() # remove leading/trailing whitespace
        #full_paragraph_text = f"{article_name} ({round(similarity*100,1)}% match)" + "\n\n" + full_paragraph_text
        if not pd.isna(article_date):
            full_paragraph_text = f"<strong><a href='{article_url}' target='_blank'>{article_name}</a> ({100*similarity:.1f}% match) - {article_date}</strong><br><br>{full_paragraph_text}"
        else:
            full_paragraph_text = f"<strong><a href='{article_url}' target='_blank'>{article_name}</a> ({100*similarity:.1f}% match)</strong><br><br>{full_paragraph_text}"
        top_paragraphs.append(full_paragraph_text) 
    return top_paragraphs