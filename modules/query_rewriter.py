from openai import OpenAI

# Function to rewrite the query in a RAG-friendly format
def rewrite_query(query, client):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful and knowledgeable assistant."},
        {"role": "user", "content": f"Rewrite the query in an informative way, perfect for Retrial Augmented Generation (RAG) models. Do not overcomplicate the language, use simple-to-understand words. Only return the query.\nQ: {query}\nRAG query: "},
    ]
    )
    return completion.choices[0].message.content
