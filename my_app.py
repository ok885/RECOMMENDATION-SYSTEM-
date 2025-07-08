from flask import Flask, render_template, request
import pickle
import numpy as np

# Load saved data
popular_books = pickle.load(open('popular_books.pkl', 'rb'))
pivot_table = pickle.load(open('pivot_table.pkl', 'rb'))
book_data = pickle.load(open('book_info.pkl', 'rb'))
similarity_matrix = pickle.load(open('similarity_matrix.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html',
                           book_titles=list(popular_books['Book-Title'].values),
                           authors=list(popular_books['Book-Author'].values),
                           images=list(popular_books['Image-URL-M'].values),
                           ratings=list(popular_books['avg_rating'].values),
                           votes=list(popular_books['rating_count'].values))

@app.route('/suggest')
def suggest_ui():
    return render_template('suggest.html')

@app.route('/suggest_books', methods=['POST'])
def suggest_books():
    user_input = request.form.get('book_input')

    try:
        index = np.where(pivot_table.index == user_input)[0][0]
    except:
        return render_template('suggest.html', error="Book not found. Try another title.")

    similar_books = sorted(list(enumerate(similarity_matrix[index])), key=lambda x: x[1], reverse=True)[1:6]

    results = []
    for i in similar_books:
        book_title = pivot_table.index[i[0]]
        book_info = book_data[book_data['Book-Title'] == book_title].drop_duplicates('Book-Title')

        if not book_info.empty:
            title = book_info['Book-Title'].values[0]
            author = book_info['Book-Author'].values[0]
            image = book_info['Image-URL-M'].values[0]
            results.append([title, author, image])

    return render_template('suggest.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
