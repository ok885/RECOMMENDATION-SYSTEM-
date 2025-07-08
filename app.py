from flask import Flask, render_template, request
import pickle
import numpy as np

top_books_df = pickle.load(open('top_books.pkl', 'rb'))
user_book_matrix = pickle.load(open('user_book_matrix.pkl', 'rb'))
book_details = pickle.load(open('book_details.pkl', 'rb'))
similarity_matrix = pickle.load(open('user_similarity.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html',
                           titles=list(top_books_df['Title'].values),
                           authors=list(top_books_df['Author'].values),
                           images=list(top_books_df['Image'].values),
                           ratings=list(top_books_df['AverageRating'].values),
                           votes=list(top_books_df['RatingCount'].values))

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend_books():
    user_input = request.form.get('user_input')
    try:
        index = np.where(user_book_matrix.index == user_input)[0][0]
    except:
        return render_template('recommend.html', error='User not found. Try another user.')

    similar_users = sorted(list(enumerate(similarity_matrix[index])), key=lambda x: x[1], reverse=True)[1:4]

    recommended_books = set()
    for i in similar_users:
        similar_user = user_book_matrix.index[i[0]]
        books_rated = user_book_matrix.loc[similar_user]
        top_books = books_rated[books_rated > 8].sort_values(ascending=False)
        for book in top_books.index:
            if book not in user_book_matrix.columns or book in recommended_books:
                continue
            recommended_books.add(book)
            if len(recommended_books) >= 5:
                break
        if len(recommended_books) >= 5:
            break

    data = []
    for book in recommended_books:
        info = book_details[book_details['Title'] == book]
        if not info.empty:
            data.append([info['Title'].values[0], info['Author'].values[0], info['Image'].values[0]])

    return render_template('recommend.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
