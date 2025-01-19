import streamlit as st
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

from data_retrieval import load_data
from sklearn.neighbors import NearestNeighbors
import time
import plotly.express as px

import missingno as msno
from plotly.subplots import make_subplots
import geopandas as gpd
from shapely.geometry import Point
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from transformers import pipeline
# import transformers
# import torch
# from langchain.docstore.document import Document
# from langchain_openai import OpenAIEmbeddings


# initialize the key by reading the file key.txt
# OPENAI_API_KEY = open("key.txt", "r").read().strip()




# Load data
@st.cache_data
def get_data():
    return load_data()

df = get_data()
df['release_date'] = pd.to_datetime(df['release_date'])
df = df.query('release_date <= "{}"'.format(pd.Timestamp.now().date()))

st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #ff0000;
        color: white;
    }
    .stSelectbox>div>div>div>div {
        background-color: #ff0000;
        color: white;
    }
    .stSlider>div>div>div>div {
        background-color: #ff0000;
    }
    .stNumberInput>div>div>div>div {
        background-color: #ff0000;
        color: white;
    }
    .stTextInput>div>div>div>input {
        background-color: #ffffff;
        color: black;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ff0000;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎬 Movie Recommendation System")
last_genre = None

# Create tabs
tab1, tab2 = st.tabs(["📊 Descriptive Statistics", "🎥 Movie Recommender"])

#------------------------- Tab 1: Descriptive Statistics
with tab1:
    st.header("Descriptive Statistics")
    
    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(df.describe())
    
    # Display the first 5 rows of the dataframe
    st.subheader("First 5 Rows of the DataFrame")
    st.write(df.head(5))
    
    # Display visualizations
    st.subheader("Visualizations")
    
    # Missing values visualization
    st.write("Missing Values Matrix")
    fig, ax = plt.subplots()
    msno.matrix(df, ax=ax, fontsize=8)
    st.pyplot(fig)

    # Function to create a box plot
    def create_box_plot(column_name, color, title):
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'xy'}]])
        fig.add_trace(go.Box(x=df[column_name], name=column_name))
        fig.update_traces(marker_color=color)
        fig.update_layout(
            title=title,
            showlegend=False,
            template='simple_white',
            font=dict(family='Arial', size=12, color='black')
        )
        st.plotly_chart(fig)

    # Box plots for various features
    create_box_plot('weighted_rating_tmdb', 'Salmon', 'Distribution of Weighted Ratings (TMDB)')
    create_box_plot('weighted_rating_imdb', 'SteelBlue', 'Distribution of Weighted Ratings (IMDB)')
    create_box_plot('budget', 'Gold', 'Distribution of Budget')
    create_box_plot('runtime', 'Purple', 'Distribution of Runtime')
    create_box_plot('revenue', 'Teal', 'Distribution of Revenue')

    # Distribution of ratings by genres
    st.subheader("Distribution of Ratings by Genres")
    def get_unique_values(column):
        return ['All'] + sorted(df[column].dropna().str.split(', ').explode().unique())
    selected_genre = st.selectbox("Select Genre", get_unique_values('genres'))

    if selected_genre != 'All':
        genre_df = df[df['genres'].str.contains(selected_genre, na=False)]
        fig, ax = plt.subplots()
        ax.hist(genre_df['weighted_rating_tmdb'], bins=20, color='#ff0000')
        ax.set_title(f'Distribution of TMDB Ratings for {selected_genre}')
        ax.set_xlabel('TMDB Rating')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    # Heatmap of production countries
    st.subheader("Heatmap of Production Countries")

    # Load world map
    world = gpd.read_file('https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_admin_0_countries.geojson')

    # Count the number of movies produced in each country
    country_counts = df['production_countries'].str.split(', ').explode().value_counts()

    # Create a DataFrame with country names and counts
    country_df = pd.DataFrame({'country': country_counts.index, 'count': country_counts.values})

    # Merge with world map
    world = world.merge(country_df, how='left', left_on='name', right_on='country')

    # Plot the heatmap
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    world.boundary.plot(ax=ax)
    world.plot(column='count', ax=ax, legend=True, cmap='OrRd', missing_kwds={'color': 'lightgrey'})
    ax.set_title('Number of Movies Produced by Country')
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap of Essential Features")

    # Create a copy of the DataFrame
    datacorr = df.copy()

    # List of essential features you want to keep
    essential_columns = [
        'weighted_rating_tmdb', 
        'weighted_rating_imdb', 
        'revenue', 
        'budget', 
        'runtime', 
        'popularity', 
        'genres', 
        'primaryName_1', 
        'director_1'
    ]

    # Step 1: Select only the relevant columns for the correlation matrix
    datacorr = datacorr[essential_columns]

    # Step 2: Identify categorical columns (e.g., 'primaryName_1', 'director_1')
    categorical_columns = datacorr.select_dtypes(include=['object']).columns.tolist()

    # Step 3: Encode categorical columns using LabelEncoder
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        datacorr[column] = label_encoder.fit_transform(datacorr[column])

    # Step 4: Filter for numeric columns (after encoding categorical columns)
    numeric_data = datacorr.select_dtypes(include=['number'])

    # Step 5: Generate the correlation matrix
    correlation_matrix = numeric_data.corr()

    # Step 6: Create the heatmap with a more neutral color palette
    plt.figure(figsize=(12, 10))  # Adjust figure size for readability
    cmap = sns.diverging_palette(230, 20, as_cmap=True)  # Coolwarm color palette
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5)

    # Step 7: Add labels and title
    plt.title('Correlation Heatmap of Essential Features', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)

    # Step 8: Display the heatmap
    st.pyplot(plt)

    # Top 10 Movies Based on IMDb and TMDB Ratings
    st.subheader("Top 10 Movies Based on Ratings")

    # Sorting the dataset to get the top 10 movies based on IMDb and TMDB ratings
    top_10_imdb = df[['title', 'weighted_rating_imdb']].sort_values(by='weighted_rating_imdb', ascending=False).drop_duplicates(subset='title').head(10)
    top_10_tmdb = df[['title', 'weighted_rating_tmdb']].sort_values(by='weighted_rating_tmdb', ascending=False).drop_duplicates(subset='title').head(10)

    # Create the table for IMDb
    table_imdb = go.Table(
        columnorder=[1, 2],
        columnwidth=[250, 100],
        header=dict(
            values=['Movie Title', 'IMDb Rating'],
            line_color='darkslategray',
            fill_color='Salmon',
            height=30
        ),
        cells=dict(
            values=[top_10_imdb['title'], top_10_imdb['weighted_rating_imdb']],
            line_color='darkslategray',
            fill_color='White'
        )
    )

    # Create the table for TMDB
    table_tmdb = go.Table(
        columnorder=[1, 2],
        columnwidth=[250, 100],
        header=dict(
            values=['Movie Title', 'TMDB Rating'],
            line_color='darkslategray',
            fill_color='LightSkyBlue',
            height=30
        ),
        cells=dict(
            values=[top_10_tmdb['title'], top_10_tmdb['weighted_rating_tmdb']],
            line_color='darkslategray',
            fill_color='White'
        )
    )

    # Create separate figures for IMDb and TMDB
    fig_imdb = go.Figure(data=[table_imdb])
    fig_tmdb = go.Figure(data=[table_tmdb])

    # Update layout settings for the IMDb table
    fig_imdb.update_layout(
        showlegend=False,
        title_text='Top 10 Movies Based on IMDb Ratings',
        title_font_size=16,
        title_font_family='Arial',
        title_x=0.5,
        font=dict(family='Arial', size=12, color='black'),
        height=400  # Adjust the figure height for better layout
    )

    # Update layout settings for the TMDB table
    fig_tmdb.update_layout(
        showlegend=False,
        title_text='Top 10 Movies Based on TMDB Ratings',
        title_font_size=16,
        title_font_family='Arial',
        title_x=0.5,
        font=dict(family='Arial', size=12, color='black'),
        height=400  # Adjust the figure height for better layout
    )

    # Display the figures
    st.plotly_chart(fig_imdb)
    st.plotly_chart(fig_tmdb)

    # Genre distribution pie chart
    st.subheader("Genre Distribution")

    # Assuming 'genres' column contains genre strings, we will split them into individual genres
    genre_list = df['genres'].dropna().str.split(',').explode().str.strip()

    # Now count the frequency of each genre
    genre_counts = genre_list.value_counts()

    # Create the pie chart
    fig = go.Figure(data=[go.Pie(labels=genre_counts.index, values=genre_counts.values, hole=0.3)])

    # Update layout settings
    fig.update_layout(
        title_text='Genre Distribution',
        title_font_size=16,
        title_font_family='Arial',
        title_x=0.5,
        font=dict(family='Arial', size=12, color='black'),
        height=500
    )

    # Show the pie chart
    st.plotly_chart(fig)

    # Insights on IMDb and TMDB Ratings
    st.subheader("Insights on IMDb and TMDB Ratings")

    # Calculate the correlation between IMDb and TMDB ratings
    correlation = df['weighted_rating_imdb'].corr(df['weighted_rating_tmdb'])

    # Calculate descriptive statistics for both IMDb and TMDB ratings
    imdb_stats = df['weighted_rating_imdb'].describe()
    tmdb_stats = df['weighted_rating_tmdb'].describe()

    # Compute the mean difference between IMDb and TMDB ratings to see if one tends to be higher
    mean_diff = df['weighted_rating_imdb'].mean() - df['weighted_rating_tmdb'].mean()

    # Count how many movies have higher IMDb ratings vs TMDB ratings
    higher_imdb = len(df[df['weighted_rating_imdb'] > df['weighted_rating_tmdb']])
    higher_tmdb = len(df[df['weighted_rating_tmdb'] > df['weighted_rating_imdb']])

    # Calculate the percentage of movies with higher IMDb ratings
    percent_higher_imdb = (higher_imdb / len(df)) * 100
    percent_higher_tmdb = (higher_tmdb / len(df)) * 100

    # Create a dictionary of insights
    insights = {
        'Correlation between IMDb and TMDB ratings': correlation,
        'IMDb Ratings Descriptive Statistics': imdb_stats,
        'TMDB Ratings Descriptive Statistics': tmdb_stats,
        'Mean Difference between IMDb and TMDB Ratings': mean_diff,
        'Movies with Higher IMDb Rating': higher_imdb,
        'Movies with Higher TMDB Rating': higher_tmdb,
        'Percentage of Movies with Higher IMDb Rating': percent_higher_imdb,
        'Percentage of Movies with Higher TMDB Rating': percent_higher_tmdb
    }

    # Display the insights in Streamlit
    for key, value in insights.items():
        st.write(f"{key}: {value}")

    # Scatter plot comparing IMDb and TMDB ratings
    fig = px.scatter(df, x='weighted_rating_imdb', y='weighted_rating_tmdb', hover_name='title',
                     labels={'weighted_rating_imdb': 'IMDb Rating', 'weighted_rating_tmdb': 'TMDB Rating'},
                     title='Comparison of IMDb and TMDB Ratings')
    st.plotly_chart(fig)

    # Budget and Revenue Statistics
    st.subheader("Budget and Revenue Statistics")

    # Descriptive statistics for 'budget' and 'revenue'
    budget_stats = df['budget'].describe()
    revenue_stats = df['revenue'].describe()

    # Correlation between 'budget', 'revenue', and 'popularity'
    budget_revenue_corr = df[['budget', 'revenue']].corr().iloc[0, 1]
    budget_popularity_corr = df[['budget', 'popularity']].corr().iloc[0, 1]
    revenue_popularity_corr = df[['revenue', 'popularity']].corr().iloc[0, 1]

    # Displaying the statistics in Streamlit
    st.write("Budget Descriptive Statistics:")
    st.write(budget_stats)
    st.write("Revenue Descriptive Statistics:")
    st.write(revenue_stats)
    st.write(f"Correlation between Budget and Revenue: {budget_revenue_corr:.2f}")
    st.write(f"Correlation between Budget and Popularity: {budget_popularity_corr:.2f}")
    st.write(f"Correlation between Revenue and Popularity: {revenue_popularity_corr:.2f}")

    # Mean difference between budget and revenue
    mean_budget = df['budget'].mean()
    mean_revenue = df['revenue'].mean()
    mean_diff = mean_revenue - mean_budget
    st.write(f"Mean Budget: {mean_budget:.2f}")
    st.write(f"Mean Revenue: {mean_revenue:.2f}")
    st.write(f"Mean Difference between Revenue and Budget: {mean_diff:.2f}")

    # Scatter plot for Revenue vs. Budget colored by Popularity
    fig = px.scatter(df, x='budget', y='revenue', color='popularity', hover_name='title',
                     labels={'budget': 'Budget ($)', 'revenue': 'Revenue ($)'},
                     title='Revenue vs. Budget (Colored by Popularity)')
    st.plotly_chart(fig)


with tab2:
    st.header("Movie Recommender")
    
    method = st.radio("Select Recommendation Method", 
                      ['Explore all movies', 'Find similar movies to'], index=0, key='method_radio', label_visibility='collapsed')
#                      ['Explore all movies', 'Find similar movies to', 'LLM RAG'], index=0, key='method_radio', label_visibility='collapsed')
    
    if method == 'Explore all movies':
        rating = st.selectbox("Select Rating", ['weighted_rating_tmdb', 'weighted_rating_imdb', 'popularity'])
        min_score, max_score = st.slider("Select Score Interval", 0.0, 10.0, (0.0, 10.0), step=0.1)
        num_movies = st.number_input("Number of Best Movies to Display", 1, 100, 10)
        
        def get_unique_values(column):
            return ['All'] + sorted(df[column].dropna().str.split(', ').explode().unique())
        
        col1, col2 = st.columns(2)
        with col1:
            genres = st.multiselect("Select Genres", get_unique_values('genres'))
            directors = st.multiselect("Select Directors", get_unique_values('directors'))
            production_companies = st.multiselect("Select Production Companies", get_unique_values('production_companies'))
        with col2:
            banned_genres = st.multiselect("Select Banned Genres", get_unique_values('genres'))
            writers = st.multiselect("Select Writers", get_unique_values('writers'))
            production_countries = st.multiselect("Select Production Countries", get_unique_values('production_countries'))
        min_date, max_date = st.slider("Select Release Date Interval", 
                   value=(df['release_date'].min().date(), df['release_date'].max().date()), 
                   format="YYYY-MM-DD")
        
        def sort_movies(df, rating='weighted_rating_tmdb', rate_min=0, rate_max=10, genres=None, banned_genres=None, production_companies=None, directors=None, writers=None, production_countries=None, date_min=None, date_max=None):
            # Filter by rating interval
            df_filtered = df[(df[rating] >= rate_min) & (df[rating] <= rate_max)]
            
            # Filter by release date interval
            if date_min and date_max:
                df_filtered = df_filtered[(df_filtered['release_date'] >= pd.to_datetime(date_min)) & (df_filtered['release_date'] <= pd.to_datetime(date_max))]
            
            # Filter by genres
            if genres and 'All' not in genres:
                genres_set = set(genres)
                df_filtered = df_filtered[df_filtered['genres'].apply(lambda x: any(genre in x for genre in genres_set) if pd.notnull(x) else False)]
            
            # Filter by banned genres
            if banned_genres and 'All' not in banned_genres:
                banned_genres_set = set(banned_genres)
                df_filtered = df_filtered[~df_filtered['genres'].apply(lambda x: any(genre in x for genre in banned_genres_set) if pd.notnull(x) else False)]
            
            # Filter by production companies
            if production_companies and 'All' not in production_companies:
                production_companies_set = set(production_companies)
                df_filtered = df_filtered[df_filtered['production_companies'].apply(lambda x: any(company in x for company in production_companies_set) if pd.notnull(x) else False)]
            
            # Filter by directors
            if directors and 'All' not in directors:
                directors_set = set(directors)
                df_filtered = df_filtered[df_filtered['directors'].apply(lambda x: any(director in x for director in directors_set) if pd.notnull(x) else False)]
            
            # Filter by writers
            if writers and 'All' not in writers:
                writers_set = set(writers)
                df_filtered = df_filtered[df_filtered['writers'].apply(lambda x: any(writer in x for writer in writers_set) if pd.notnull(x) else False)]
            
            # Filter by production countries
            if production_countries and 'All' not in production_countries:
                production_countries_set = set(production_countries)
                df_filtered = df_filtered[df_filtered['production_countries'].apply(lambda x: any(country in x for country in production_countries_set) if pd.notnull(x) else False)]
            
            # Sort by rating
            df_sorted = df_filtered.sort_values(by=rating, ascending=False)
            
            return df_sorted
        
        if st.button("Sort Movies"):
            sorted_df = sort_movies(df, rating=rating, rate_min=min_score, rate_max=max_score, genres=genres, banned_genres=banned_genres, production_companies=production_companies, directors=directors, writers=writers, production_countries=production_countries, date_min=min_date, date_max=max_date)
            titles = sorted_df['title'].head(num_movies)
            posters = ['https://image.tmdb.org/t/p/w500' + sorted_df[sorted_df['title'] == title]['poster_path'].values[0] for title in titles]
            overviews = sorted_df['overview'].head(num_movies)
            scores = sorted_df[rating].head(num_movies)

            for i, (title, poster, overview, score) in enumerate(zip(titles, posters, overviews, scores)):
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    st.markdown(f"<h1 style='color: #ff0000;'>{i+1}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #ff0000;'><strong>Score:</strong> {score:.2f}</p>", unsafe_allow_html=True)
                with col2:
                    st.image(poster, width=150)
                with col3:
                    st.markdown(f"<h3 style='color: #ff0000;'>{title}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: black; margin-bottom: 0.2em;'><strong>Production Company:</strong> {sorted_df[sorted_df['title'] == title]['production_companies'].values[0]}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: black; margin-bottom: 0.2em;'><strong>Writers:</strong> {sorted_df[sorted_df['title'] == title]['writers'].values[0]}</p>", unsafe_allow_html=True)
                    primary_names = []
                    for i in range(1, 6):
                        name = sorted_df[sorted_df['title'] == title][f'primaryName_{i}'].values[0]
                        if pd.notnull(name):
                            primary_names.append(name)
                    
                    if primary_names:
                        st.markdown(f"<p style='color: black; margin-bottom: 0.2em;'><strong>Top 5 Actors:</strong> {', '.join(primary_names)}</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p style='color: black; margin-bottom: 0.2em;'><strong>Top 5 Actors:</strong> None</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: black; font-style: italic;'>{overview}</p>", unsafe_allow_html=True)
                st.markdown("<hr style='border: 1px solid #D3D3D3;'>", unsafe_allow_html=True)

    elif method == 'Find similar movies to':
        def get_unique_values(column):
            return ['All'] + sorted(df[column].dropna().str.split(', ').explode().unique())
        
        genre = st.selectbox("Select a Genre", get_unique_values('genres'))
        movie_title = st.selectbox("Select a Movie", df[df['genres'].str.contains(genre, na=False)]['title'])
        n_similar = st.number_input("Number of Similar Movies", 1, 100, 10)

        if st.button("Find Similar Movies"):
            start_time = time.time()
            @st.cache_data
            def compute_similarity_matrix(df):
                # Load a Faster BERT-like Model
                model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

                # Combine relevant text features into a single column
                df['combined_text'] = df[['title', 'overview', 'primaryName_1', 'primaryName_2', 'primaryName_3', 'primaryName_4', 'primaryName_5', 'director_1', 'director_2', 'writer_1', 'writer_2']].fillna('').agg(' '.join, axis=1)

                # Generate Embeddings for combined text and overview separately
                st.write('Generating embeddings for combined text...')
                try:
                    embeddings_combined = model.encode(df['combined_text'].tolist(), batch_size=64, show_progress_bar=True)
                except Exception as e:
                    st.write(f"Error during embedding generation: {e}")
                    return None, None

                # Compute cosine similarity matrices
                st.write('Computing cosine similarity matrices...')
                cosine_sim = cosine_similarity(embeddings_combined)

                # Constructing a reverse map of indices and movie titles
                indices = pd.Series(df.index, index=df['title']).drop_duplicates()

                return cosine_sim, indices

            if last_genre != genre:
                filtered_df = df[(df['genres'].str.contains(genre, na=False))].reset_index(drop=True)
                cosine_sim, indices = compute_similarity_matrix(filtered_df)
                last_genre = genre


            def find_similar_movies(movie_title, top_n=5):
                if movie_title not in indices:
                    return f"Movie '{movie_title}' not found in the dataset."
            
                # Get the index of the movie that matches the title
                idx = indices[movie_title]

                # Get the list of cosine similarity scores for that particular movie with all movies
                sim_scores = list(enumerate(cosine_sim[idx]))

                # Sort the movies based on the similarity scores
                sim_scores = sorted(sim_scores, key=lambda x: float(x[1]), reverse=True)

                # Get the scores of the top_n most similar movies
                sim_scores = sim_scores[1:top_n + 1]

                # Get the movie indices
                movie_indices = [i[0] for i in sim_scores]

                # Return the top_n most similar movies
                return filtered_df['title'].iloc[movie_indices]
                
            similar_movies = find_similar_movies(movie_title, n_similar)
            titles = similar_movies
            posters = ['https://image.tmdb.org/t/p/w500' + df[df['title'] == title]['poster_path'].values[0] for title in titles]
            overviews = df[df['title'].isin(titles)]['overview']
            scores = df[df['title'].isin(titles)]['weighted_rating_tmdb']
            runtimes = df[df['title'].isin(titles)]['runtime']

            selected_movie = df[df['title'] == movie_title].iloc[0]
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Elapsed time: {elapsed_time:.2f} seconds")

            st.markdown("<hr style='border: 1px solid #D3D3D3;'>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: #ff0000; text-align: center;'>Selected Movie: {selected_movie['title']}</h2>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(f"https://image.tmdb.org/t/p/w500{selected_movie['poster_path']}", width=200)
            with col2:
                st.markdown(f"<p style='color: black;'><strong>Overview:</strong> {selected_movie['overview']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #ff0000;'><strong>Score:</strong> {selected_movie['weighted_rating_tmdb']:.2f}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #ff0000;'><strong>Runtime:</strong> {selected_movie['runtime']} min</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: black;'><strong>Production Company:</strong> {selected_movie['production_companies']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: black;'><strong>Writers:</strong> {selected_movie['writers']}</p>", unsafe_allow_html=True)
                primary_names = [selected_movie[f'primaryName_{i}'] for i in range(1, 6) if pd.notnull(selected_movie[f'primaryName_{i}'])]
                if primary_names:
                    st.markdown(f"<p style='color: black;'><strong>Top 5 Actors:</strong> {', '.join(primary_names)}</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='color: black;'><strong>Top 5 Actors:</strong> None</p>", unsafe_allow_html=True)
            st.markdown("<hr style='border: 1px solid #D3D3D3;'>", unsafe_allow_html=True)
            # Your existing code here

            st.markdown(f"<p style='color: black;'><strong>Writers:</strong> {selected_movie['writers']}</p>", unsafe_allow_html=True)
            primary_names = [selected_movie[f'primaryName_{i}'] for i in range(1, 6) if pd.notnull(selected_movie[f'primaryName_{i}'])]
            if primary_names:
                st.markdown(f"<p style='color: black;'><strong>Top 5 Actors:</strong> {', '.join(primary_names)}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color: black;'><strong>Top 5 Actors:</strong> None</p>", unsafe_allow_html=True)
            st.markdown("<hr style='border: 1px solid #D3D3D3;'>", unsafe_allow_html=True)
            st.markdown("<h2 style='color: #ff0000;'>Suggestions:</h2>", unsafe_allow_html=True)
            for i, (title, poster, overview, score, runtime) in enumerate(zip(titles, posters, overviews, scores, runtimes)):
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    st.markdown(f"<h1 style='color: #ff0000;'>{i+1}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #ff0000;'><strong>Score:</strong> {score:.2f}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #ff0000;'><strong>Runtime:</strong> {runtime} min</p>", unsafe_allow_html=True)
                with col2:
                    st.image(poster, width=150)
                with col3:
                    st.markdown(f"<h3 style='color: #ff0000;'>{title}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: black; margin-bottom: 0.2em;'><strong>Production Company:</strong> {df[df['title'] == title]['production_companies'].values[0]}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: black; margin-bottom: 0.2em;'><strong>Writers:</strong> {df[df['title'] == title]['writers'].values[0]}</p>", unsafe_allow_html=True)
                    primary_names = []
                    for i in range(1, 6):
                        name = df[df['title'] == title][f'primaryName_{i}'].values[0]
                        if pd.notnull(name):
                            primary_names.append(name)
                    
                    if primary_names:
                        st.markdown(f"<p style='color: black; margin-bottom: 0.2em;'><strong>Top 5 Actors:</strong> {', '.join(primary_names)}</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p style='color: black; margin-bottom: 0.2em;'><strong>Top 5 Actors:</strong> None</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: black; font-style: italic;'>{overview}</p>", unsafe_allow_html=True)
                st.markdown("<hr style='border: 1px solid #D3D3D3;'>", unsafe_allow_html=True)


# TODO
    # elif method == 'LLM RAG':
    #     prompt = st.text_input("Enter your preferences (e.g., 'thriller movies with high ratings')")
    #     if st.button("Get Recommendations"):
    #         if prompt:
    #             # Initialize the vector store
    #             embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    #             documents = [Document(page_content=overview) for overview in df['overview'].tolist()]
    #             vectorstore = FAISS.from_documents(documents, embeddings)

    #             # Initialize the LLM
    #             model_id = "meta-llama/Meta-Llama-3-8B"
    #             llm = pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

    #             # Create the RetrievalQA chain
    #             qa = RetrievalQA.from_chain_type(
    #                 llm=llm,
    #                 chain_type="stuff",
    #                 retriever=vectorstore.as_retriever(),
    #                 return_source_documents=True,
    #             )

    #             # Get recommendations
    #             results = qa.invoke(prompt)
    #             recommendations = results['result']
    #             st.write(recommendations)

    #             # Placeholder for LLM model call
    #             recommendations = [{"title": "Example Movie", "poster_path": "/example.jpg", "description": "Example description", "score": 8.5}]
    #             st.markdown("<h2 style='color: #ff0000;'>Recommendations:</h2>", unsafe_allow_html=True)
    #             for rec in recommendations:
    #                 col1, col2 = st.columns([1, 3])
    #                 with col1:
    #                     st.image(f"https://image.tmdb.org/t/p/w500{rec['poster_path']}", width=150)
    #                 with col2:
    #                     st.markdown(f"<h3 style='color: #ff0000;'>{rec['title']}</h3>", unsafe_allow_html=True)
    #                     st.markdown(f"<p style='color: black;'><strong>Description:</strong> {rec['description']}</p>", unsafe_allow_html=True)
    #                     st.markdown(f"<p style='color: #ff0000;'><strong>Score:</strong> {rec['score']}</p>", unsafe_allow_html=True)
    #                 st.markdown("<hr style='border: 1px solid #D3D3D3;'>", unsafe_allow_html=True)
