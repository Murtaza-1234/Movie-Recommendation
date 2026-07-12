import streamlit as st
import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit command)
# ---------------------------------------------------------
st.set_page_config(page_title="CineMatch", page_icon="🎞️", layout="centered")

# ---------------------------------------------------------
# CUSTOM STYLING (cinema marquee theme)
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 50% 0%, #1c1c28 0%, #101018 60%, #0b0b10 100%);
    color: #EDE6D6;
}

/* Hide default streamlit chrome for a cleaner look */
#MainMenu, footer, header {visibility: hidden;}

/* Title block */
.cine-title {
    font-family: 'Playfair Display', serif;
    font-weight: 800;
    font-size: 3rem;
    text-align: center;
    background: linear-gradient(90deg, #D4A94A, #F1D68A, #D4A94A);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
    letter-spacing: 1px;
}

.cine-subtitle {
    text-align: center;
    color: #9C94A8;
    font-size: 0.95rem;
    margin-bottom: 1.2rem;
    letter-spacing: 0.5px;
}

/* Film strip perforation divider */
.filmstrip {
    height: 18px;
    background-image: repeating-linear-gradient(
        90deg,
        #D4A94A 0px, #D4A94A 8px,
        transparent 8px, transparent 20px
    );
    opacity: 0.55;
    border-radius: 3px;
    margin-bottom: 1.8rem;
}

/* Section headers */
.section-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #F1D68A;
    margin-top: 1.5rem;
    margin-bottom: 0.4rem;
    border-left: 4px solid #8B2E3F;
    padding-left: 10px;
}

/* Recommendation cards */
.movie-card {
    background: #1e1e29;
    border-left: 3px solid #D4A94A;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 1.05rem;
    color: #EDE6D6;
    transition: all 0.2s ease-in-out;
}
.movie-card:hover {
    border-left: 3px solid #8B2E3F;
    background: #262635;
    transform: translateX(4px);
}

/* Input box */
.stTextInput input {
    background-color: #1a1a24 !important;
    color: #EDE6D6 !important;
    border: 1px solid #3a3a4a !important;
    border-radius: 6px !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(90deg, #8B2E3F, #6e2432);
    color: #F1D68A;
    border: 1px solid #D4A94A;
    border-radius: 6px;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s ease-in-out;
}
.stButton button:hover {
    background: linear-gradient(90deg, #D4A94A, #C79A3C);
    color: #14141c;
    border: 1px solid #F1D68A;
}

/* Multiselect + slider labels */
label, .stSlider label, .stMultiSelect label {
    color: #C9BFAF !important;
}
</style>
""", unsafe_allow_html=True)

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
# HELPERS
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
    return [movies.iloc[i].title for i in indices[0][1:]]

def get_recommendations_by_features(genre_list, popularity, vote_average):
    user_vector = build_user_vector(genre_list, popularity, vote_average)
    distances, indices = knn.kneighbors(user_vector)
    return [movies.iloc[i].title for i in indices[0]]

def render_movie_cards(titles):
    for t in titles:
        st.markdown(f'<div class="movie-card">🎬 &nbsp; {t}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.markdown('<div class="cine-title">CineMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="cine-subtitle">Find your next favorite film — powered by KNN</div>', unsafe_allow_html=True)
st.markdown('<div class="filmstrip"></div>', unsafe_allow_html=True)

movie_name = st.text_input("Enter a movie you like", placeholder="e.g. Avatar")

if st.button("🎟️ Get Recommendations"):
    if movie_name.strip() == "":
        st.warning("Please type a movie name first.")
    else:
        results = get_recommendations_by_title(movie_name)
        if results is None:
            st.error(f"'{movie_name}' isn't in the dataset.")
            st.session_state["show_manual_form"] = True
        else:
            st.session_state["show_manual_form"] = False
            st.markdown(f'<div class="section-label">Because you liked {movie_name}</div>', unsafe_allow_html=True)
            render_movie_cards(results)

# ---------------------------------------------------------
# FALLBACK: manual feature entry if movie not found
# ---------------------------------------------------------
if st.session_state.get("show_manual_form"):
    st.markdown('<div class="section-label">Describe the movie instead</div>', unsafe_allow_html=True)
    st.caption("Pick genres and rough stats, and we'll find real movies that match.")

    selected_genres = st.multiselect("Genres", options=all_genre_names)
    popularity_input = st.slider("Popularity", 0.0, 800.0, 50.0)
    vote_input = st.slider("Vote average", 0.0, 10.0, 6.5)

    if st.button("✨ Find matching movies"):
        if not selected_genres:
            st.warning("Please select at least one genre.")
        else:
            results = get_recommendations_by_features(selected_genres, popularity_input, vote_input)
            st.markdown('<div class="section-label">Movies matching your description</div>', unsafe_allow_html=True)
            render_movie_cards(results)
