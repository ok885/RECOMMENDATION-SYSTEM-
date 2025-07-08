from flask import Flask, render_template, request
import pickle
import numpy as np

# Load required files
top_books_df = pickle.load(open('top_books.pkl', 'rb'))
user_book_matrix = pickle.load(open('user_book_matrix.pkl', 'rb'))
book_details = pickle.load(open('book_details.pkl', 'rb'))
similarity_scores = pickle.load(open('user_similarity.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           book_names=list(top_books_df['Title'].values),
                           authors=list(top_books_df['Author'].values),
                           images=list(top_books_df['Image'].values),
                           ratings=list(top_books_df['AverageRating'].values),
                           votes=list(top_books_df['RatingCount'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')

    try:
        index = np.where(user_book_matrix.index == int(user_input))[0][0]
    except:
        return render_template('recommend.html', error='❌ User ID not found. Try a different one.')

    similar_users = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:4]

    recommended_books = []
    for i in similar_users:
        user_id = user_book_matrix.index[i[0]]
        user_ratings = user_book_matrix.loc[user_id]
        top_rated_books = user_ratings[user_ratings > 8].sort_values(ascending=False)
        for book in top_rated_books.index:
            if book not in recommended_books and book in book_details['Title'].values:
                book_info = book_details[book_details['Title'] == book].drop_duplicates('Title')
                if not book_info.empty:
                    recommended_books.append([
                        book_info['Title'].values[0],
                        book_info['Author'].values[0],
                        book_info['Image'].values[0]
                    ])
            if len(recommended_books) >= 5:
                break
        if len(recommended_books) >= 5:
            break

    return render_template('recommend.html', data=recommended_books)

if __name__ == '__main__':
    app.run(debug=True)
