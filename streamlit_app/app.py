import streamlit as st # type: ignore
import requests # type: ignore
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('../model/movies_data.pkl', 'rb') as file:
    movies, cosine_sim = pickle.load(file)  
    
  
# Function to get movie recommendations based on cosine similarity
def get_recommendations(title, cosine_sim, num_recommendations):
    if title in movies['title'].values:
        indx = movies[movies['title'] == title].index[0]
        sim_score = list(enumerate(cosine_sim[indx]))
        sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
        sim_score = sim_score[1:num_recommendations+1]  # get the top N similar movies
        movie_indices = [i[0] for i in sim_score]
        return movies.iloc[movie_indices]
    else:
        # Use TF-IDF to find related movies based on the keyword similarity
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(movies['title'].values)
        query_vec = tfidf_vectorizer.transform([title])
        cosine_sim_keywords = cosine_similarity(query_vec, tfidf_matrix).flatten()
        related_indices = cosine_sim_keywords.argsort()[-num_recommendations:][::-1]
        return movies.iloc[related_indices]

# Function to fetch movie poster from The Movie Database (TMDb)
def fetch_poster(movie_id):
    api_key = 'a174222fc313ec9905f3539ddd7498fe'
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}'
    response = requests.get(url)
    data = response.json()
    poster_path = data['poster_path']
    full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
    return full_path

# Streamlit app title
st.title("Movie Recommendation System")

# Sidebar for movie selection and number of recommendations
with st.sidebar:
    # Text input to enter a movie title
    selected_movie = st.text_input("Enter a movie title")
    
    
    # attaching the id to the search
    # selected_movie = st.selectbox("Select a movie", movies['title'].values)

    # Display the selected movie's description if it exists
    if selected_movie in movies['title'].values:
        movie_description = movies[movies['title'] == selected_movie]['overview'].values[0]
        st.write("Movie Description:")
        st.write(movie_description)
    else:
        st.write("Movie not found in the dataset. Showing recommendations based on similar titles.")

    # Slider to select the number of recommendations
    num_recommendations = st.slider("Number of movies to recommend", min_value=5, max_value=25, value=10)

# Recommend button
if st.button("Recommend"):
    # Get movie recommendations
    recommendations = get_recommendations(selected_movie, cosine_sim, num_recommendations)

    if not recommendations.empty:
        st.write(f"Top {num_recommendations} Recommended Movies:")
        # Display recommendations in rows
        for i in range(0, num_recommendations, 5):  # Loop over rows (5 movies per row)
            cols = st.columns(5)  # Create 5 columns for each row
            for col, j in zip(cols, range(i, i + 5)):
                if j < len(recommendations):
                    movie_title = recommendations.iloc[j]['title']
                    movie_id = recommendations.iloc[j]['movie_id']
                    poster_url = fetch_poster(movie_id)
                    with col:
                        st.image(poster_url, width=120)
                        st.write(movie_title)
    else:
        st.write("Sorry, no similar movies found based on your input. Please try a different title.")
