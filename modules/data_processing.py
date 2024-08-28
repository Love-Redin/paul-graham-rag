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

def get_top_n_similar_paragraphs_including_context_old(query, model, paragraph_df, n=10):
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

def get_top_n_similar_paragraphs_including_context_wrong(query, model, paragraph_df, n=10, batch_size=1000):
    query_embedding = get_embedding(query, model)
    
    top_n_similar_paragraphs = pd.DataFrame()
    
    for start in range(0, len(paragraph_df), batch_size):
        # Process data in batches
        end = start + batch_size
        batch = paragraph_df[start:end].copy()
        
        batch['similarity'] = batch['embedding'].apply(lambda x: np.dot(x, query_embedding))
        batch_sorted = batch.sort_values(by='similarity', ascending=False).head(n)
        
        # Append to top_n_similar_paragraphs and then sort again to keep only top n overall
        top_n_similar_paragraphs = pd.concat([top_n_similar_paragraphs, batch_sorted]).sort_values(by='similarity', ascending=False).head(n)
    
    top_paragraphs = []
    for index, row in top_n_similar_paragraphs.iterrows():
        paragraph_text = row['paragraph_text']
        paragraph_number = row['paragraph_number']
        article_name = row['article_name']
        article_url = row['article_url']
        similarity = row['similarity']
        
        # Simplify the context retrieval by handling edges more robustly
        context_range = 3 if index <= 2 else 2 if index <= 5 else 1
        
        context_paragraphs = []
        for i in range(-context_range, context_range + 1):
            if i == 0:
                continue  # Skip the original paragraph
            
            context_paragraph_number = paragraph_number + i
            if context_paragraph_number > 0 and context_paragraph_number <= paragraph_df[paragraph_df['article_name'] == article_name]['paragraph_number'].max():
                context_paragraph = paragraph_df[(paragraph_df['article_name'] == article_name) & 
                                                 (paragraph_df['paragraph_number'] == context_paragraph_number)]['paragraph_text'].values
                if context_paragraph.size > 0:
                    context_paragraphs.append(context_paragraph[0])
        
        full_paragraph_text = f"{article_name}: {similarity} (top {index+1})\n{' ---> '.join(context_paragraphs)} ---> {paragraph_text}"
        top_paragraphs.append(full_paragraph_text)
    
    return top_paragraphs


def get_top_n_similar_paragraphs_including_context(query, model, paragraph_df, n=10, batch_size=100):
    query_embedding = get_embedding(query, model)
    
    top_n_similar_paragraphs = pd.DataFrame()
    
    for start in range(0, len(paragraph_df), batch_size):
        # Process data in batches
        end = start + batch_size
        batch = paragraph_df[start:end].copy()
        
        batch['similarity'] = batch['embedding'].apply(lambda x: np.dot(x, query_embedding))
        batch_sorted = batch.sort_values(by='similarity', ascending=False).head(n)
        
        # Append to top_n_similar_paragraphs and then sort again to keep only top n overall
        top_n_similar_paragraphs = pd.concat([top_n_similar_paragraphs, batch_sorted]).sort_values(by='similarity', ascending=False).head(n)
    
    top_paragraphs = []
    for index, row in top_n_similar_paragraphs.iterrows():
        paragraph_text = row['paragraph_text']
        paragraph_number = row['paragraph_number']
        article_name = row['article_name']
        article_url = row['article_url']
        similarity = row['similarity']
        article_date = row['created_at']
        
        # Determine context range based on the ranking of the paragraph
        context_range = 3 if index <= 2 else 2 if index <= 5 else 1
        
        previous_paragraphs = []
        next_paragraphs = []

        for i in range(1, context_range + 1):
            if paragraph_number - i > 0:
                prev_paragraph = paragraph_df[(paragraph_df['article_name'] == article_name) & 
                                              (paragraph_df['paragraph_number'] == paragraph_number - i)]['paragraph_text'].values
                if prev_paragraph.size > 0:
                    previous_paragraphs.append(prev_paragraph[0])

            if paragraph_number + i <= paragraph_df[paragraph_df['article_name'] == article_name]['paragraph_number'].max():
                next_paragraph = paragraph_df[(paragraph_df['article_name'] == article_name) & 
                                              (paragraph_df['paragraph_number'] == paragraph_number + i)]['paragraph_text'].values
                if next_paragraph.size > 0:
                    next_paragraphs.append(next_paragraph[0])

        previous_paragraphs.reverse()
        full_paragraph_text = "\n\n".join(previous_paragraphs) + "\n\n" + paragraph_text + "\n\n" + "\n\n".join(next_paragraphs)
        
        # Format with HTML as in the original function
        if not pd.isna(article_date):
            full_paragraph_text = f"<strong><a href='{article_url}' target='_blank'>{article_name}</a> ({100*similarity:.1f}% match) - {article_date}</strong><br><br>{full_paragraph_text}"
        else:
            full_paragraph_text = f"<strong><a href='{article_url}' target='_blank'>{article_name}</a> ({100*similarity:.1f}% match)</strong><br><br>{full_paragraph_text}"
        
        top_paragraphs.append(full_paragraph_text)
    
    return top_paragraphs
