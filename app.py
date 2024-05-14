import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Define styles for your app
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading CSS: {e}")

# Load the dataset (assuming it's preprocessed and ready for use)
# @st.cache(allow_output_mutation=True)
def load_data():
    try:
        data = pd.read_json('/home/oem/CRE-Recommendation-System/tokenized_project_works.json', lines=True)
        data['title'].fillna('No Title', inplace=True)
        data['keywords'].fillna('', inplace=True)
        data['keywords'] = data['keywords'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        data['text'] = data['title'].str.lower() + " " + data['keywords'].str.lower()
        return data
    except ValueError as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

data = load_data()

# Initialize TF-IDF Vectorizer and compute cosine similarity matrix
# @st.cache(allow_output_mutation=True)
def tfidf_and_cosine(data):
    if not data.empty:
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['text'])
        return cosine_similarity(tfidf_matrix, tfidf_matrix)
    return None

cosine_sim = tfidf_and_cosine(data)

def get_recommendations_by_title(input_title):
    if cosine_sim is not None:
        normalized_title = input_title.lower()
        matches = data[data['title'].str.lower().str.contains(normalized_title)]
        if matches.empty:
            return "Project title not found. Please check the title and try again."
        else:
            project_index = matches.index[0]
            sim_scores = list(enumerate(cosine_sim[project_index]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
            project_indices = [i[0] for i in sim_scores]
            return data.loc[project_indices, ['author', 'title', 'keywords']]
    else:
        return "Data or cosine similarity matrix is not available."

# Load custom CSS to improve the aesthetics
local_css("/home/oem/CRE-Recommendation-System/style.css")

# Streamlit user interface
st.title('Project Work Recommendation System')
st.subheader('Explore similar project works:')
input_title = st.text_input("Enter a project title", "", help="Type the title of a project work to find similar projects.")

if st.button('Get Recommendations'):
    if input_title:
        results = get_recommendations_by_title(input_title)
        if isinstance(results, pd.DataFrame):
            st.dataframe(results.style.applymap(lambda x: 'background-color : lightblue'))
        else:
            st.error(results)
    else:
        st.error("Please enter a title to get recommendations.")
