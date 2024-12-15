from flask import Flask, request, jsonify
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
from flask_cors import CORS  # Import CORS

# TMDB API Key
API_KEY = "3776781c8aea2e47d76bd18d3b21e3d2"
BASE_URL = "https://api.themoviedb.org/3"

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/recommendations": {"origins": "http://localhost:5173"}})  # Allow requests from the frontend

# Function to fetch movie details from TMDB
def fetch_movie_details(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}"
    response = requests.get(url)
    return response.json()

# Function to fetch multiple movies for building the dataset
def fetch_movies():
    url = f"{BASE_URL}/discover/movie?api_key={API_KEY}&sort_by=popularity.desc&page=1"
    response = requests.get(url)
    return response.json().get("results", [])

# Preprocess genres
def preprocess_genres(movies):
    genre_ids = [[genre['id'] for genre in movie.get("genres", [])] for movie in movies]
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(genre_ids)
    return genre_matrix, mlb

# Preprocess text (overview and tagline)
def preprocess_text(movies):
    combined_texts = [f"{movie.get('overview', '')} {movie.get('tagline', '')}" for movie in movies]
    tfidf = TfidfVectorizer(stop_words="english")
    text_matrix = tfidf.fit_transform(combined_texts)
    return text_matrix, tfidf

# Build KNN model
def build_knn_model():
    movies = []
    for movie_summary in fetch_movies():
        movie_details = fetch_movie_details(movie_summary['id'])
        movies.append(movie_details)

    genre_matrix, genre_encoder = preprocess_genres(movies)
    text_matrix, tfidf_vectorizer = preprocess_text(movies)
    combined_features = hstack([genre_matrix, text_matrix])

    knn = NearestNeighbors(n_neighbors=5, metric="cosine")
    knn.fit(combined_features)

    return knn, movies, genre_encoder, tfidf_vectorizer

# Recommend movies based on a clicked movie
def recommend_movies(knn, movies, genre_encoder, tfidf_vectorizer, clicked_movie_id):
    clicked_movie = fetch_movie_details(clicked_movie_id)
    genres = genre_encoder.transform([[genre['id'] for genre in clicked_movie.get("genres", [])]])
    text = tfidf_vectorizer.transform([f"{clicked_movie.get('overview', '')} {clicked_movie.get('tagline', '')}"])

    clicked_features = hstack([genres, text])
    distances, indices = knn.kneighbors(clicked_features, n_neighbors=6)  # Fetch 10 recommendations
    recommendations = [movies[i] for i in indices.flatten()]

    # Filter out the clicked movie from the recommendations
    recommendations = [movie for movie in recommendations if movie['id'] != clicked_movie_id]

    return recommendations[:5]  # Ensure we return exactly 5 recommendations



# Endpoint to get movie recommendations
@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    # Get the movie ID from the request arguments
    movie_id = request.args.get('movie_id', type=int)

    if not movie_id:
        return jsonify({"error": "Movie ID is required"}), 400

    # Get recommendations for the clicked movie
    recommendations = recommend_movies(knn_model, movies_data, genre_enc, tfidf_vec, movie_id)

    # Return the full movie objects as JSON
    return jsonify(recommendations)

# Main flow
if __name__ == "__main__":
    # Build the KNN model and store relevant data
    knn_model, movies_data, genre_enc, tfidf_vec = build_knn_model()

    # Run Flask app
    app.run(debug=True)
