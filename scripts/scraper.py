import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from tqdm import tqdm
import os

# Purpose of file: fetch data and store in csv file

def fetch_articles():
    # Fetches all essays from Paul Graham's website
    url = 'https://www.paulgraham.com/articles.html'

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    links = soup.find_all('a')

    articles = {}

    for link in links:
        path = link.get('href')
        
        # Use regex to see if path is an article on PG's site, on the format abcd.html
        if re.match(r'[a-z]+\.html', str(path)) and "index.html" not in path:
            article_name = link.text
            article_url = f'https://www.paulgraham.com/{path}'
            print(article_name, article_url)
            articles[article_name] = article_url

    return articles


articles = fetch_articles()
#df = pd.DataFrame(articles.items(), columns=['Article', 'URL'])
#df.to_csv(os.path.join('..', 'data', 'articles.txt'), index=False)

def get_paragraphs_from_article(article_name, article_url):
    response = requests.get(article_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the paragraphs by splitting on <br/><br/>
    html_content = str(soup)
    paragraphs = html_content.split('<br/><br/>')


    # Helpful for filtering out non-text elements
    non_text_start = ['<link', '<table', '<font', '<html', '<hr/>', '</br>', '<script']
    title_start = '<b>'
    title_end = '</b>'

    actual_text = []

    # Print each paragraph separated by a line
    for p_index, p in enumerate(paragraphs):
        stripped_p = p.strip()
        if stripped_p.startswith(title_start) and stripped_p.endswith(title_end):
            paragraph_type = "TITLE"
        else:
            paragraph_type = "TEXT"

        for start in non_text_start:
            if stripped_p.startswith(start):
                paragraph_type = "OTHER"
                break

        if stripped_p == "":
            paragraph_type = "EMPTY"

        #print(paragraph_type, "\n", p)
        #print("----------------")

        paragraph_without_tags = re.sub('<[^<]+?>', '', p)

        if paragraph_type in ["TEXT", "TITLE"]:
            actual_text.append(paragraph_without_tags)

    return actual_text

""" 
df = pd.DataFrame()

article_count = 0
for article_name, article_url in articles.items():
    article_count += 1
    actual_text = get_paragraphs_from_article(article_name, article_url)
    print(article_name)
    print("\n\n".join(actual_text[:3]))
    print("-"*50)
    df.at[article_name, "url"] = article_url
    df.at[article_name, "text"] = actual_text
     for p in actual_text:
        sentences = p.split(".")
        if len(sentences) <= 10:
            # then use the full paragraph as an embedding


df.to_csv(os.path.join('..', 'data', 'articles.csv')) """

# Create a DataFrame to store the articles
data = []
# Process each article
for article_name, article_url in tqdm(articles.items()):
    actual_text = get_paragraphs_from_article(article_name, article_url)
    data.append([article_name, article_url, str(actual_text)])

df = pd.DataFrame(data, columns=['article', 'url', 'text'])

# Save the DataFrame to a CSV file
df.to_csv('data/articles.csv', index=False)

# We have the title and the contents of the article
# TODO: extract the date when the essay was written


def extract_date(text):
    # Extracts the date from the text
    pass


