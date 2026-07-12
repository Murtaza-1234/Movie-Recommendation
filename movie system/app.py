import streamlit as st
import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit command)
# ---------------------------------------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

# ---------------------------------------------------------
# LOAD + PREPROCESS DATA (cached so it only runs once)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    movies = movies[['title', 'genres', 'popularity', 'vote_average']]
    movies.dropna(inplace=True)
    movies.reset_index(drop=True, inplace=True)

    movies['genres'] = movies['genres'].apply(
        lambda x: [g['name'] for g in ast.literal_eval(x)]
    )
    return movies

@st.cache_resource
def build_model(movies):
    genres_encoded = movies['genres'].str.join('|').str.get_dummies()

    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(movies[['popularity', 'vote_average']])
    numeric_scaled = pd.DataFrame(numeric_scaled, columns=['popularity', 'vote_average'])

    features = pd.concat([genres_encoded, numeric_scaled], axis=1)

    knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
    knn.fit(features)

    return genres_encoded, scaler, features, knn

movies = load_data()
genres_encoded, scaler, features, knn = build_model(movies)

all_genre_names = sorted(genres_encoded.columns.tolist())

# ---------------------------------------------------------
# HELPER: build a feature row from manual genre/popularity/rating input
# ---------------------------------------------------------
def build_user_vector(genre_list, popularity, vote_average):
    total_genres = pd.DataFrame(0, index=[0], columns=genres_encoded.columns)
    for genre in genre_list:
        if genre in total_genres.columns:
            total_genres.loc[0, genre] = 1

    numeric_scaled = scaler.transform([[popularity, vote_average]])
    numeric_df = pd.DataFrame(numeric_scaled, columns=['popularity', 'vote_average'], index=[0])

    user_vector = pd.concat([total_genres, numeric_df], axis=1)
    return user_vector

def get_recommendations_by_title(movie_name):
    matches = movies[movies['title'].str.lower() == movie_name.lower()]
    if matches.empty:
        return None
    idx = matches.index[0]
    distances, indices = knn.kneighbors([features.iloc[idx]])
    results = [movies.iloc[i].title for i in indices[0][1:]]
    return results

def get_recommendations_by_features(genre_list, popularity, vote_average):
    user_vector = build_user_vector(genre_list, popularity, vote_average)
    distances, indices = knn.kneighbors(user_vector)
    results = [movies.iloc[i].title for i in indices[0]]
    return results

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🎬 Movie Recommendation System")
st.write("Type a movie you like, and get similar movies — powered by a KNN model.")

movie_name = st.text_input("Enter a movie name", placeholder="e.g. Avatar")

if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("Please type a movie name first.")
    else:
        results = get_recommendations_by_title(movie_name)
        if results is None:
            st.error(f"'{movie_name}' was not found in the dataset.")
            st.session_state["show_manual_form"] = True
        else:
            st.session_state["show_manual_form"] = False
            st.success(f"Movies similar to '{movie_name}':")
            for r in results:
                st.write("🎥", r)

# ---------------------------------------------------------
# FALLBACK: manual feature entry if movie not found
# ---------------------------------------------------------
if st.session_state.get("show_manual_form"):
    st.divider()
    st.subheader("Movie not found — describe it instead")
    st.write("Pick the genres and rough stats, and we'll find similar real movies.")

    selected_genres = st.multiselect("Genres", options=all_genre_names)
    popularity_input = st.slider("Popularity", 0.0, 800.0, 50.0)
    vote_input = st.slider("Vote average", 0.0, 10.0, 6.5)

    if st.button("Get recommendations from features"):
        if not selected_genres:
            st.warning("Please select at least one genre.")
        else:
            results = get_recommendations_by_features(selected_genres, popularity_input, vote_input)
            st.success("Movies matching your description:")
            for r in results:
                st.write("🎥", r)
