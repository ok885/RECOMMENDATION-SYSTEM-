#  Recommender-System

**COMPANY**: CODTECH IT SOLUTIONS  
**NAME**: BHARAT BHANDARI  
**INTERN ID**: CT04DF123  
**DOMAIN**: MACHINE LEARNING  
**DURATION**: 4 WEEKS  
**MENTOR**: NEELA SANTOSH

---

This project implements a basic **Book Recommendation System** using **item-based collaborative filtering**. It suggests books similar to a given title based on user ratings. The implementation is kept minimal and clean, perfect for understanding the fundamentals of recommendation systems.

---

## 🧾 About the Project

The system uses the Book-Crossing dataset and creates a similarity matrix between books using **cosine similarity**. The recommendation is based on finding books that were rated similarly by many users.

---

## 📘 Dataset

The following CSV files are used:

- `Books.csv` – Contains metadata about books (Book-ID, Title, Author)
- `Users.csv` – Contains user demographics
- `Ratings.csv` – Contains user ratings for books

---

## ⚙️ How It Works

1. **Load and Merge Data**  
   Datasets are merged using common columns to get a single DataFrame with book ratings.

2. **Data Cleaning**  
   - Rows with missing values are removed  
   - Books with less than 50 ratings are filtered out to reduce sparsity

3. **Matrix Creation**  
   A **user-book pivot table** is created, where each row is a user and each column is a book. Missing ratings are filled with 0.

4. **Cosine Similarity**  
   Cosine similarity is calculated between the columns (books) to identify books with similar rating patterns.

5. **Recommendation Function**  
   For a given book title, the system returns the top 5 most similar books.

---

## 🧠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn (`cosine_similarity` from `sklearn.metrics.pairwise`)  

---

## ▶️ How to Use

### 1. Install Required Libraries

```bash
pip install pandas numpy scikit-learn
```

### 2. Make Sure You Have the Datasets

Place the following files in the same directory as the script:

- `Books.csv`  
- `Users.csv`  
- `Ratings.csv`

### 3. Run the Python Script

```bash
python book_recommender_minimal.py
```

---

## 🔍 Sample Output

**Input:**  
`Harry Potter and the Sorcerer's Stone`

**Suggested Books:**
- Harry Potter and the Chamber of Secrets  
- The Hobbit  
- Eragon  
- The Chronicles of Narnia  
- Percy Jackson and the Olympians

---

## 📌 Learnings

- Understood collaborative filtering at a fundamental level  
- Learned how cosine similarity can be used for recommendation  
- Practiced data merging, pivoting, and matrix operations  
- Developed a basic, readable recommender system from scratch

---

## 🚀 Future Additions

- Add user-based collaborative filtering  
- Include matrix factorization techniques like SVD or NMF  
- Add a web interface (Flask or Streamlit)  
- Deploy the system as an interactive app

---

👨‍💻 **Created by:** Bharat Bhandari  
📆 **Internship at Codtech IT Solutions**  
🧠 **Guided by:** Neela Santosh
