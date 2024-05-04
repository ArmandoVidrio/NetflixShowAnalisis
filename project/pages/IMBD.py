import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import requests

### Funciones que necesitaremos
PLACEHOLDER = "https://placekitten.com/200/300"

def fetch_img(movie):
    url = f'http://www.omdbapi.com/?apikey=1a9586fd&t={movie}'
    response = requests.get(url)
    data = response.json()
    if 'Poster' in data:
        return data['Poster']
    else:
        return PLACEHOLDER
    
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
    dataset.reset_index(drop=True, inplace=True)
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

    return dataset

def obtener_peliculas_similares(dataset, movie, top_n=5):
    movie_index = dataset.index[dataset['title']==movie].tolist()[0]
    # Convertir listas en vectores binarios usando MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    features = mlb.fit_transform(dataset['genres'] + dataset['keywords'])

    # Calcular la similitud de coseno entre películas basándote en sus características numéricas
    similarity_matrix = cosine_similarity(features)
    movie_similarities = list(enumerate(similarity_matrix[movie_index]))
    movie_similarities_sorted = sorted(movie_similarities, key=lambda x: x[1], reverse=True)
    similar_movies_indices = [movie[0] for movie in movie_similarities_sorted[1:top_n+1]]

    _peliculasRecomendadas = dataset.iloc[similar_movies_indices]

    return _peliculasRecomendadas

if __name__ == "__main__":
    ## Creamos nuestro dataframe
    movies_data = createCleanDataset()

    ## Creamos la aplicacion de streamlit
    st.header('Recomendador de peliculas IMBD')
    ## peliculas disponibles
    peliculas_disponibles = set(movies_data['title'])
    pelicula_base = st.selectbox("Seleccionar pelicula para hacer las recomendaciones", peliculas_disponibles)

    if st.button('Buscar'):
        ## Obtenemos las recomendaciones, sus nombres, sus imagenes y el link a la pagina
        peliculasRecomendadas = obtener_peliculas_similares(movies_data, pelicula_base)
        nombresPeliculas = peliculasRecomendadas['title'].tolist()
        imagesPeliculas = [fetch_img(pelicula) for pelicula in nombresPeliculas]
        linkPaginaIMDB = peliculasRecomendadas['homepage'].tolist()

        ## Imprimimos las recomendaciones en la aplicacion
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.write(nombresPeliculas[0])
            st.image(imagesPeliculas[0])
            st.write(f'Mas información: {linkPaginaIMDB[0]}')
        with col2:
            st.write(nombresPeliculas[1])
            st.image(imagesPeliculas[1])
            st.write(f'Mas información: {linkPaginaIMDB[1]}')
        with col3:
            st.write(nombresPeliculas[2])
            st.image(imagesPeliculas[2])
            st.write(f'Mas información: {linkPaginaIMDB[2]}')
        with col4:
            st.write(nombresPeliculas[3])
            st.image(imagesPeliculas[3])
            st.write(f'Mas información: {linkPaginaIMDB[3]}')
        with col5:
            st.write(nombresPeliculas[4])
            st.image(imagesPeliculas[4])
            st.write(f'Mas información: {linkPaginaIMDB[4]}')
    st.header("Curiosidades sobre las peliculas")

