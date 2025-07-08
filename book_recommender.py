import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
books = pd.read_csv("Books.csv")
users = pd.read_csv("Users.csv")
ratings = pd.read_csv("Ratings.csv")

# Merge datasets
data = ratings.merge(books, on="Book-ID").merge(users, on="User-ID")

# Filter data
data = data.dropna()
popular_books = data.groupby("Book-Title").filter(lambda x: len(x) >= 50)

# Create pivot table
user_book_matrix = popular_books.pivot_table(index="User-ID", columns="Book-Title", values="Book-Rating")
user_book_matrix.fillna(0, inplace=True)

# Calculate cosine similarity
similarity = cosine_similarity(user_book_matrix.T)
similarity_df = pd.DataFrame(similarity, index=user_book_matrix.columns, columns=user_book_matrix.columns)

# Recommendation function
def recommend(book_title, top_n=5):
    if book_title not in similarity_df:
        return "Book not found in database."
    similar_books = similarity_df[book_title].sort_values(ascending=False)[1:top_n+1]
    return similar_books

# Example usage
if __name__ == "__main__":
    book_name = "Harry Potter and the Sorcerer's Stone"
    print(f"If you liked '{book_name}', you might also enjoy:")
    print(recommend(book_name))
