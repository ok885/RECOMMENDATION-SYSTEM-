# Minimal Book Recommender using Collaborative Filtering
# Author: Bharat Bhandari

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
books = pd.read_csv("Books.csv")
users = pd.read_csv("Users.csv")
ratings = pd.read_csv("Ratings.csv")

# Merge and clean data
data = ratings.merge(books, on='Book-ID').merge(users, on='User-ID')
data = data.dropna(subset=['Book-Rating'])

# Filter popular books
popular_books = data['Book-Title'].value_counts()
popular_books = popular_books[popular_books >= 50].index
data = data[data['Book-Title'].isin(popular_books)]

# Create pivot table
user_book_matrix = data.pivot_table(index='User-ID', columns='Book-Title', values='Book-Rating').fillna(0)

# Compute cosine similarity
similarity = cosine_similarity(user_book_matrix.T)
similarity_df = pd.DataFrame(similarity, index=user_book_matrix.columns, columns=user_book_matrix.columns)

# Recommend function
def recommend(book_name, top_n=5):
    if book_name not in similarity_df:
        return "Book not found."
    similar_books = similarity_df[book_name].sort_values(ascending=False)[1:top_n+1]
    return similar_books

# Example usage
if __name__ == "__main__":
    book = "Harry Potter and the Sorcerer's Stone"
    print(f"Recommendations for '{book}':")
    print(recommend(book))
