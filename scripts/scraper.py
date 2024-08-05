import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from tqdm import tqdm
import os

# Purpose of script: fetch data and store in csv file

def extract_date(paragraph):
    # Extracts the date from the paragraph, if possible - else returns None
    # Regex to match the specific format
    date_pattern = r'<font face="verdana" size="2">([A-Za-z]+)\s(\d{4})'
    match = re.search(date_pattern, paragraph.strip())
    if match:
        # Extract the month and year
        month = match.group(1)
        year = match.group(2)
        month_year = f"{month} {year}"
        return month_year
    

def fetch_articles():
    # Fetches all essays from Paul Graham's website
    url = 'https://www.paulgraham.com/articles.html'

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    links = soup.find_all('a')

    articles = {}

    for link in links:#[:5]: # only fetch the first 5 articles when in development
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
    print(paragraphs)

    created_at = None

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

        month_year = None
        for start in non_text_start:
            if stripped_p.startswith(start):
                paragraph_type = "OTHER"
                if start == '<font':
                    month_year = extract_date(stripped_p)
                    if month_year:
                        paragraph_type = "DATE"
                        created_at = month_year
                break

        if stripped_p == "":
            paragraph_type = "EMPTY"

        print(paragraph_type, "\n", p)
        print("----------------")

        paragraph_without_tags = re.sub('<[^<]+?>', '', p)

        if paragraph_type in ["TEXT", "TITLE"]:
            actual_text.append(paragraph_without_tags)

    return actual_text, created_at


# Create a DataFrame to store the articles
data = []
# Process each article
for article_name, article_url in tqdm(articles.items()):
    actual_text, created_at = get_paragraphs_from_article(article_name, article_url)
    data.append([article_name, article_url, str(actual_text), created_at])

df = pd.DataFrame(data, columns=['article', 'url', 'text', 'created_at'])

print(df.head())

# Save the DataFrame to a CSV file
df.to_csv('data/articles.csv', index=False) # comment out this line when in development