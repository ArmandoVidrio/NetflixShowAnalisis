import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
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

    #cambiar tipo de fecha a datetime
    dataset['release_date'] = pd.to_datetime(dataset['release_date'])

    #cambiar genres lrg
    dataset['production_companies'] = dataset['production_companies'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

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

    # Definir la función para extraer los géneros
def extraer_generos(json_list) -> list:
    nombres = []
    for json in json_list:
        nombres.append(json['name'])
    return nombres

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

""" # GENRE BY TITLE

def extract_genres(data, movie_title):
    #data['genres'] = data['genres'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    # Ensure the 'genres' column is a list of dictionaries if it is stored as a string
    if isinstance(data['genres'].iloc[0], str):
        data['genres'] = data['genres'].apply(lambda x: json.loads(x))
    
    # Filter the DataFrame for the given movie title
    movie_row = data[data['original_title'] == movie_title]
    
    # Check if any movie was found
    if not movie_row.empty:
        # Extract genre list from the first (and should be only) row
        genres_list = movie_row.iloc[0]['genres']
        # Collect all genre names into a list
        
        genre_names = [genre['name'] for genre in genres_list]
        return genre_names
    else:
        return "Movie not found"
    
# Define functions to filter and retrieve top movies by genre
def filter_by_genre(data, genre_name):
    return data[data['genres'].apply(lambda g: any(genre['name'] == genre_name for genre in g))]

def get_top_movies_by_rating(data, genre_name):
    genre_movies = filter_by_genre(data, genre_name)
    top_movies = genre_movies.sort_values(by='vote_average', ascending=False).head(5)
    return top_movies

def top_genre_votes(data, genre_name):
    top_movies = get_top_movies_by_rating(data, genre_name)
    plt.figure(figsize=(15, 6))  # Wider plot for better layout
    colors = list(mcolors.TABLEAU_COLORS)  # Get a set of color names

    # Create a scatter plot
    for i, (index, row) in enumerate(top_movies.iterrows()):
        plt.scatter(row['vote_count'], row['vote_average'], color=colors[i % len(colors)], s=100, label=row['original_title'])

    plt.xlabel('Vote Count')
    plt.ylabel('Average Vote')
    plt.title(f'Top 5 {genre_name} Movies by Average Vote')
    plt.legend(title="Movie Titles", loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside of the plot
    plt.grid(True)
    st.pyplot(plt) """

    # YEAR BY TITLE

     # Filter movies by year
def filter_movies_by_year(data, year):
    return data[data['release_date'].dt.year == year]

# Get top 5 movies by vote average
def get_top_movies_by_vote(data, year):
    year_movies = filter_movies_by_year(data, year)
    top_movies = year_movies.sort_values(by='vote_average', ascending=False).head(5)
    return top_movies

# Get top 5 movies by popularity
def get_top_movies_by_popularity(data, year):
    year_movies = filter_movies_by_year(data, year)
    top_movies = year_movies.sort_values(by='popularity', ascending=False).head(5)
    return top_movies

    # Function to extract the release year of a movie by its title
def extract_release_year(data, movie_title):
    # Filter the DataFrame for the given movie title
    movie_row = data[data['original_title'] == movie_title]

    # Check if the movie was found
    if not movie_row.empty:
        # Extract the release year from the 'release_date' column
        release_year = movie_row['release_date'].dt.year.iloc[0]
        return release_year
    else:
        return "Movie not found"
    
# Plot top movies in a vertical bar chart
def top_year_popularity(data, year):
    top_movies = get_top_movies_by_popularity(data, year)
    plt.figure(figsize=(10, 8))  # Adjusted for better display of vertical bars
    plt.bar(top_movies['original_title'], top_movies['popularity'], color='teal')
    plt.ylabel('Popularity')
    plt.xlabel('Movie Title')
    plt.title(f'Top 5 Movies of {year} by Popularity')
    plt.xticks(rotation=45)  # Rotate movie titles for better readability

    # Adjust the y-axis scale dynamically based on the range of popularity values
    popularity_min = top_movies['popularity'].min() - 1  # slightly lower to add breathing room
    popularity_max = top_movies['popularity'].max() + 1  # slightly higher to add breathing room
    plt.yticks(ticks=np.arange(int(popularity_min), int(popularity_max) + 10, 10))  # adjust ticks on y-axis for clarity

    st.pyplot(plt)

 
def top_by_year(data, year):
    top_movies = get_top_movies_by_vote(data, year)
    plt.figure(figsize=(15, 8))  # Adjusted for better display of horizontal bars
    plt.barh(top_movies['original_title'], top_movies['vote_average'], color='skyblue')
    plt.ylabel('Movie Title')
    plt.xlabel('Average Vote')
    plt.title(f'Top 5 Movies of {year} by Average Vote')

    # Setting y-axis scale to have a tick every 0.25 points
    vote_min = top_movies['vote_average'].min() - 0.50  # slightly lower to add breathing room
    vote_max = top_movies['vote_average'].max() + 0.50  # slightly higher to add breathing room
    plt.xticks(ticks=np.arange(int(vote_min), int(vote_max) + 0.50, 0.50))  # adjust ticks on x-axis for clarity

    st.pyplot(plt)

"""     # PRODUCTION COMPANY BY TITLE
    
def extract_production_company(data, movie_title):
     # Find the movie row based on its title
    movie_row = data[data['original_title'] == movie_title]
    
    # Check if any movie was found
    if movie_row.empty:
        return "Movie not found"
    
    # Extract the production company list from the first (should be only) row
    production_companies_list = movie_row.iloc[0]['production_companies']
    
    # Debug print to check what the actual data looks like
    #print("Debug - Production Companies Data:", production_companies_list)

    return production_companies_list
    
    
# Filter movies by production company

def filter_movies_by_company(data, company_name):
    def valid_company(companies):
        if isinstance(companies, list):
            return any(isinstance(company, dict) and company.get('name') == company_name for company in companies)
        return False

    return data[data['production_companies'].apply(valid_company)]


# Get top 5 movies by vote average
def get_top_movies_by_company(data, company_name):
    company_movies = filter_movies_by_company(data, company_name)
    top_movies = company_movies.sort_values(by='vote_average', ascending=False).head(5)
    return top_movies[['original_title', 'vote_average', 'vote_count']]  # Including vote_count for the scatter plot


def top_by_company(data, company_name):
    top_movies = get_top_movies_by_company(data, company_name)
    plt.figure(figsize=(15, 6))  # Wider plot
    colors = list(mcolors.TABLEAU_COLORS)  # Get a set of color names from Matplotlib's tableau color set

    # Create a scatter plot
    for i, (index, row) in enumerate(top_movies.iterrows()):
        plt.scatter(row['vote_count'], row['vote_average'], color=colors[i % len(colors)], s=100, label=row['original_title'])

    plt.xlabel('Vote Count')
    plt.ylabel('Average Vote')
    plt.title(f'Top 5 Movies by {company_name} by Average Vote')
    plt.legend(title="Movie Titles", loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside of the plot
    plt.grid(True)
    st.pyplot(plt) """


def extract_movie_language(df, title):
        # Buscar la película por título
    movie_row = df[df['original_title'] == title]
    
    # Extraer el lenguaje
    if not movie_row.empty:
        return movie_row['original_language'].iloc[0]
    else:
        return "Película no encontrada"
    
def top_5_movies_by_language(data, language):
    # Filtrar las películas por idioma
    filtered_data = data[data['original_language'] == language]
    
    # Ordenar por popularidad de mayor a menor
    sorted_data = filtered_data.sort_values(by='popularity', ascending=False)
    
    # Seleccionar las top 5
    top_5_movies = sorted_data.head(5)
    
    # Crear la gráfica de barras
    plt.figure(figsize=(10, 6))
    plt.bar(top_5_movies['original_title'], top_5_movies['popularity'], color='green')
    plt.xlabel('Movie Title')
    plt.ylabel('Popularity')
    plt.title(f'Top 5 Most Popular Movies in {language} Language')
    plt.xticks(rotation=45)
    st.pyplot(plt)

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
        
    # lrg genre = extract_genres(movies_data, pelicula_base)

    release_year = extract_release_year(movies_data, pelicula_base)
    language = extract_movie_language(movies_data, pelicula_base)

    st.header("Curiosidades Relacionadas a tu Pelicula")

    top_year_popularity(movies_data, release_year)

    top_by_year(movies_data, release_year)

    top_5_movies_by_language(movies_data, language)
    # top 5 peliculas del mismo genero por vote_avergae
    #  top 5 peliculas del mismo genero por popularidad
    #  top 5 peliculas del misma productora por vote_avergre








    