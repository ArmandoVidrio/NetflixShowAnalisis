import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

### Funciones que necesitaremos

def aplanar_json(dataset, columna, llave):
    ## Verificamos si es un tipo de datos str para convertirlo a un tipo de dato de python
    dataset[columna] = dataset[columna].apply(lambda x: json.loads(x) if isinstance(x,str) else x)

    ### Extraemos que necesitamos
    dataset[columna] = dataset[columna].apply(lambda x: [y[llave] for y in x] if x else [])

def createCleanDataset():
    # Combinamos nuestros 2 cvs
    df_credits = pd.read_csv(r'../data/tmdb_5000_credits.csv')
    df_movies = pd.read_csv(r'../data/tmdb_5000_movies.csv')

    # Hacemos las preparaciones para el merge
    df_movies.rename(columns={'id':'movie_id'}, inplace=True)
    df_credits.drop(columns='title', inplace=True)

    # Juntamos los 2 dataframes
    dataset = pd.merge(df_movies, df_credits, on='movie_id', how='inner')

    dataset.set_index('movie_id', inplace=True)
    dataset = dataset.sort_index()

    #borrar NaN de release_date
    dataset = dataset.dropna(subset=['release_date'])

    #borrar NaN de runtime
    dataset = dataset.dropna(subset=['runtime'])

    # Reemplazar NaN por 'No disponble"
    columnas_noDisponibles = {'homepage': 'No disponible', 'tagline': 'No disponible', 'overview': 'No disponible'}
    dataset = dataset.fillna(columnas_noDisponibles)

    # Aplanamos
    aplanar_json(dataset, 'genres', 'name')
    aplanar_json(dataset, 'keywords', 'name')
    aplanar_json(dataset, 'production_companies', 'name')
    aplanar_json(dataset, 'production_countries', 'name')
    aplanar_json(dataset, 'spoken_languages', 'name')
    aplanar_json(dataset, 'cast', 'character')
    aplanar_json(dataset, 'crew', 'name')

    # Renderizamos
    st.dataframe(dataset)

    return dataset

def obtener_indices_peliculas_similares(dataset, movie_index, top_n=5):
    # Convertir listas en vectores binarios usando MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    features = mlb.fit_transform(dataset['genres'] + dataset['keywords'])

    # Calcular la similitud de coseno entre películas basándote en sus características numéricas
    similarity_matrix = cosine_similarity(features)
    movie_similarities = list(enumerate(similarity_matrix[movie_index]))
    movie_similarities_sorted = sorted(movie_similarities, key=lambda x: x[1], reverse=True)
    similar_movies_indices = [movie[0] for movie in movie_similarities_sorted[1:top_n+1]]

    return similar_movies_indices

if __name__ == "__main__":
    movies_data = createCleanDataset()
    indicesSimilares = obtener_indices_peliculas_similares(movies_data, 0)
    peliculasSimilares = movies_data.iloc[indicesSimilares]
    st.write(peliculasSimilares)

