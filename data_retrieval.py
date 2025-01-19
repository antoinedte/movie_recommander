# packages
import pandas as pd
import numpy as np
import kagglehub
import os


def load_data():

    # check if the file "df_clean.csv" exists in the current directory
    if os.path.exists("df_clean.csv"):
        df_clean = pd.read_csv("df_clean.csv")
    # if the file does not exist, download the TMDB dataset and process it
    else:
        print("Downloading and processing the TMDB dataset...")
        # Download and load the TMDB dataset
        path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")
        tmdbDF = pd.read_csv(os.path.join(path, "TMDB_movie_dataset_v11.csv"))

        # Drop adult content and unnecessary columns
        tmdbDF = tmdbDF[tmdbDF['adult'] == False]
        COLUMNS_WITH_NO_INTEREST = ['backdrop_path', 'homepage', 'tagline', 'adult']
        tmdbDF.drop(columns=COLUMNS_WITH_NO_INTEREST, inplace=True)

        # Drop rows with missing values in important columns
        COLUMNS_TO_BE_FILLED = ['release_date', 'imdb_id', 'overview', 'poster_path']
        tmdbDF.dropna(subset=COLUMNS_TO_BE_FILLED, inplace=True)

        # Load and process IMDb name data
        name = pd.read_csv("https://datasets.imdbws.com/name.basics.tsv.gz", compression='gzip', sep='\t', header=0)
        name_dict = dict(zip(name['nconst'], name['primaryName']))

        # Load and process IMDb principals data
        principals = pd.read_csv("https://datasets.imdbws.com/title.principals.tsv.gz", compression='gzip', sep='\t', header=0)
        principal_actors = principals[principals['category'].isin(['actor', 'actress', 'self'])].drop_duplicates(subset=['tconst', 'nconst'])
        principal_actors['nconst_count'] = principal_actors.groupby('tconst').cumcount() + 1
        principal_actors = principal_actors[principal_actors['nconst_count'] <= 5]

        # Merge actor names with principal actors
        actor_names = pd.merge(principal_actors[['tconst', 'nconst_count', 'nconst']], name[['nconst', 'primaryName']], on='nconst', how='left')
        actor_pivot = actor_names.pivot(index='tconst', columns='nconst_count', values='primaryName')
        actor_pivot.columns = [f'primaryName_{col}' for col in actor_pivot.columns]
        actor_pivot.reset_index(inplace=True)

        # Merge actor data with TMDB dataset
        tmdbDF.rename(columns={'imdb_id': 'tconst'}, inplace=True)
        df = pd.merge(tmdbDF, actor_pivot, on='tconst', how='left')

        # Load and process IMDb crew data
        crew = pd.read_csv("https://datasets.imdbws.com/title.crew.tsv.gz", compression='gzip', sep='\t', header=0)
        crew['writers'] = crew['writers'].apply(lambda x: x.split(','))
        crew['writers'] = crew['writers'].apply(lambda writers: [np.nan if w.strip() == '\\N' else w.strip() for w in writers])
        crew['writers'] = crew['writers'].apply(lambda writers: [name_dict.get(w, np.nan) for w in writers])
        crew['writers'] = crew['writers'].apply(lambda writers: ', '.join([str(w) if pd.notna(w) else '' for w in writers]) if writers else np.nan)

        crew['directors'] = crew['directors'].apply(lambda x: x.split(','))
        crew['directors'] = crew['directors'].apply(lambda directors: [np.nan if d.strip() == '\\N' else d.strip() for d in directors])
        crew['directors'] = crew['directors'].apply(lambda directors: [name_dict.get(d, np.nan) for d in directors])
        crew['directors'] = crew['directors'].apply(lambda directors: ', '.join([str(d) if pd.notna(d) else '' for d in directors]) if directors else np.nan)

        # Merge crew data with the main dataframe
        df = pd.merge(df, crew, on='tconst', how='left')

        # Load and merge IMDb ratings data
        rating = pd.read_csv("https://datasets.imdbws.com/title.ratings.tsv.gz", compression='gzip', sep='\t', header=0)
        df = pd.merge(df, rating, on="tconst", how="left")

        # Clean the dataframe by dropping rows with missing ratings or directors
        df_clean = df.dropna(subset=['averageRating', 'directors'])

        # Split writers and directors into separate columns
        df_clean_writers = df_clean['writers'].str.split(', ', expand=True).rename(columns=lambda x: f'writer_{x+1}')
        df_clean = pd.concat([df_clean, df_clean_writers[[f'writer_{x+1}' for x in range(3)]]], axis=1)
        df_clean_directors = df_clean['directors'].str.split(', ', expand=True).rename(columns=lambda x: f'director_{x+1}')
        df_clean = pd.concat([df_clean, df_clean_directors[[f'director_{x+1}' for x in range(3)]]], axis=1)

        def weighted_rating(df):
            m = 10
            v = df['vote_count']
            r = df['vote_average']
            c = df['vote_average'].mean()
            # Calculation based on the IMDB formula
            return (v/(v+m) * r) + (m/(m+v) * c)

        def weighted_rating_2(df):
            m = 10
            v = df['numVotes']
            r = df['averageRating']
            c = df['averageRating'].mean()
            # Calculation based on the IMDB formula
            return (v/(v+m) * r) + (m/(m+v) * c)

        df_clean['weighted_rating'] = weighted_rating(df_clean)
        df_clean['weighted_rating_2'] = weighted_rating_2(df_clean)

        col_to_be_renammed = {
            # tmdb
            'vote_average': 'average_rate_tmdb',
            'vote_count': 'num_votes_tmdb',
            'weighted_rating': 'weighted_rating_tmdb',

            # imdbws
            'averageRating': 'average_rate_imdb',
            'numVotes': 'num_votes_imdb',
            'weighted_rating_2': 'weighted_rating_imdb',
        }

        df_clean.rename(columns=col_to_be_renammed, inplace=True)

        df_clean = df_clean[df_clean['status'] == 'Released'].copy()
        df_clean.drop('status', axis=1, inplace=True)


        # Save the cleaned dataframe to a CSV file
        df_clean.to_csv("df_clean.csv", index=False)

    return df_clean