from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load your dataset
data = pd.read_csv('archive (12)/goodreads_data.csv')

# Convert 'Num_Ratings' to numeric (remove commas if necessary)
data['Num_Ratings'] = data['Num_Ratings'].str.replace(',', '').astype(float)

# Step 1: Vectorize Genres (TF-IDF)
tfidf = TfidfVectorizer(stop_words='english')
genres_matrix = tfidf.fit_transform(data['Genres'])

# Step 2: Normalize Avg_Rating and Num_Ratings
scaler = MinMaxScaler()
rating_matrix = scaler.fit_transform(data[['Avg_Rating', 'Num_Ratings']])

# Combine Genres, Avg_Rating, and Num_Ratings into a feature matrix
features = np.hstack((genres_matrix.toarray(), rating_matrix))

# Step 3: Function to recommend books based on user input
def recommend_books(user_genre, user_avg_rating, user_num_ratings, num_recommendations=10):
    # Convert user inputs to feature space
    user_genre_vector = tfidf.transform([user_genre]).toarray()
    user_rating_vector = scaler.transform([[user_avg_rating, user_num_ratings]])
    user_features = np.hstack((user_genre_vector, user_rating_vector))
    
    # Compute cosine similarity between user input and all books
    cosine_sim = cosine_similarity(user_features, features)
    sim_scores = cosine_sim[0]
    top_indices = sim_scores.argsort()[-num_recommendations:][::-1]
    
    # Return relevant book information
    return data[['Book', 'Author', 'Description', 'URL']].iloc[top_indices].to_dict(orient='records')

@app.route('/')
def home():
    return render_template('index_3.html')

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    data = request.json
    genre = data['genre']
    avg_rating = float(data['avr_rating'])
    num_ratings = 1000000  # Example: using a fixed number of ratings; you can change this based on your needs
    num_results = int(data['num_results'])
    
    try:
        recommendations = recommend_books(genre, avg_rating, num_ratings, num_results)
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
