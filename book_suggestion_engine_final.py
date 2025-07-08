#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


book_info_merged_data = pd.read_csv('book_info_merged_data.csv')
user_profiles_merged_data = pd.read_csv('user_profiles_merged_data.csv')
user_ratings_merged_data = pd.read_csv('user_ratings_merged_data.csv')


# In[ ]:


book_info_merged_data['Image-URL-M'][1]


# In[ ]:


user_profiles_merged_data.head()


# In[ ]:


user_ratings_merged_data.head()


# In[ ]:


print(book_info_merged_data.shape)
print(user_ratings_merged_data.shape)
print(user_profiles_merged_data.shape)


# In[ ]:


book_info_merged_data.isnull().sum()


# In[ ]:


user_profiles_merged_data.isnull().sum()


# In[ ]:


user_ratings_merged_data.isnull().sum()


# In[ ]:


book_info_merged_data.duplicated().sum()


# In[ ]:


user_ratings_merged_data.duplicated().sum()


# In[ ]:


user_profiles_merged_data.duplicated().sum()


# ## Popularity Based Recommender System

# In[ ]:


user_ratings_merged_data_with_name = user_ratings_merged_data.merge(book_info_merged_data,on='ISBN')


# In[ ]:


num_rating_merged_data = user_ratings_merged_data_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_merged_data.rename(columns={'Book-Rating':'num_user_ratings_merged_data'},inplace=True)
num_rating_merged_data


# In[ ]:


avg_rating_merged_data = user_ratings_merged_data_with_name.groupby('Book-Title').mean()['Book-Rating'].reset_index()
avg_rating_merged_data.rename(columns={'Book-Rating':'avg_rating'},inplace=True)
avg_rating_merged_data


# In[ ]:


popular_merged_data = num_rating_merged_data.merge(avg_rating_merged_data,on='Book-Title')
popular_merged_data


# In[ ]:


popular_merged_data = popular_merged_data[popular_merged_data['num_user_ratings_merged_data']>=250].sort_values('avg_rating',ascending=False).head(50)


# In[ ]:


popular_merged_data = popular_merged_data.merge(book_info_merged_data,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_user_ratings_merged_data','avg_rating']]


# In[ ]:


popular_merged_data['Image-URL-M'][0]


# ## Collaborative Filtering Based Recommender System

# In[ ]:


x = user_ratings_merged_data_with_name.groupby('User-ID').count()['Book-Rating'] > 200
padhe_likhe_user_profiles_merged_data = x[x].index


# In[ ]:


filtered_rating = user_ratings_merged_data_with_name[user_ratings_merged_data_with_name['User-ID'].isin(padhe_likhe_user_profiles_merged_data)]


# In[ ]:


y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_book_info_merged_data = y[y].index


# In[ ]:


final_user_ratings_merged_data = filtered_rating[filtered_rating['Book-Title'].isin(famous_book_info_merged_data)]


# In[ ]:


pt = final_user_ratings_merged_data.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')


# In[ ]:


pt.fillna(0,inplace=True)


# In[ ]:


pt


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


similarity_scores = cosine_similarity(pt)


# In[ ]:


similarity_scores.shape


# In[ ]:


def recommend(book_name):
    ## index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_merged_data = book_info_merged_data[book_info_merged_data['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_merged_data.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_merged_data.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_merged_data.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)

    return data


# In[ ]:


recommend('1984')


# In[ ]:


pt.index[545]


# In[ ]:


import pickle
pickle.dump(popular_merged_data,open('popular.pkl','wb'))


# In[ ]:


book_info_merged_data.drop_duplicates('Book-Title')


# In[ ]:


pickle.dump(pt,open('pt.pkl','wb'))
pickle.dump(book_info_merged_data,open('book_info_merged_data.pkl','wb'))
pickle.dump(similarity_scores,open('similarity_scores.pkl','wb'))


# In[ ]:





# ## 🔢 Part 2: Matrix Factorization using SVD
# Here we apply matrix factorization with TruncatedSVD to extract latent features.

# In[ ]:


# Import SVD
from sklearn.decomposition import TruncatedSVD

# Reduce dimensions to find latent features
svd_model = TruncatedSVD(n_components=20, random_state=42)
latent_matrix = svd_model.fit_transform(user_book_matrix.T)

# Create a DataFrame of latent features
latent_df = pd.DataFrame(latent_matrix, index=user_book_matrix.columns)
latent_similarity = cosine_similarity(latent_df)
latent_similarity_df = pd.DataFrame(latent_similarity, index=latent_df.index, columns=latent_df.index)


# ### 🔍 Define SVD-based Recommendation Function

# In[ ]:


def recommend_svd(book_name, top_n=5):
    if book_name not in latent_similarity_df:
        return "Book not found in latent feature space."
    similar = latent_similarity_df[book_name].sort_values(ascending=False)[1:top_n+1]
    return similar


# ### 📊 Evaluate Example Recommendation from SVD Model

# In[ ]:


test_book = "Harry Potter and the Sorcerer's Stone"
print(f"Recommendations using SVD for '{test_book}':")
print(recommend_svd(test_book))

