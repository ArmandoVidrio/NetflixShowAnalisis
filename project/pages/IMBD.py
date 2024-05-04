import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import json
from pandas import json_normalize

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
    st.write(dataset)

def loadWidgets() -> None:
    ## Description button
    with st.expander("What's this app about?"):

        """
        This app has the objective to give custom Netflix series and/or movies recomendations 
        based on parameters given by the user.
        """
    
    ## We calculate and show how many differents movies are in the dataset
    different_movies = data.loc[data['type'] == 'Movie', 'title'].unique()
    number_different_movies = len(different_movies)
    st.metric(label="Number of movies",value=number_different_movies)

    ## We calculate and show how many differents tv shows are in the dataset
    different_tv_shows = data.loc[data['type'] == 'TV Show', 'title'].unique()
    number_different_tv_shows = len(different_tv_shows)
    st.metric(label="Number of series", value=number_different_tv_shows)

    ## Select columns to be shown 
    df_filtered = data[['type', 'country','rating']]

    gb = GridOptionsBuilder.from_dataframe(df_filtered)

    ## We create the sidebar menu
    user_input = st.sidebar.text_input('Enter your name', 'Your Name')
    selected_option = st.sidebar.selectbox('Select an option', ['Option 1', 'Option 2', 'Option 3'])

    ## We create our dynamic table
    gb = GridOptionsBuilder

if __name__ == "__main__":
    createCleanDataset()

