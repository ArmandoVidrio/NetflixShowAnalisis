import streamlit as st
import pandas as pd
import pickle
import requests

dataframes_options = ['-','Amazon', 'Disney', 'Hulu', 'Netflix']
PLACEHOLDER = "https://placekitten.com/200/300"

def fetch_img(movie):
    url = f'http://www.omdbapi.com/?apikey=1a9586fd&t={movie}'
    response = requests.get(url)
    data = response.json()
    if 'Poster' in data:
        return data['Poster']
    else:
        return PLACEHOLDER

def recommend(name):
    index = movies[movies['title'] == name].index[0]
    distance = sorted(list(enumerate(sim[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = []
    posters = []
    for i in distance[1:6]:
        recommended_movies.append(movies.iloc[i[0]].title)
        posters.append(fetch_img(movies.iloc[i[0]].title))
        
    return recommended_movies, posters

st.header('Recomendador de series y películas')

selected_platform = st.selectbox("Selecciona una opción de plataforma", dataframes_options)

if selected_platform != dataframes_options[0]:
    movies = pickle.load(open(f'./saved/df_{selected_platform.lower()}.pkl', 'rb'))
    sim = pickle.load(open(f'./saved/sim_{selected_platform.lower()}.pkl', 'rb'))
    movies_list = movies['title'].values
    selected_movie = st.selectbox('Selecciona una serie o película', movies_list)
    

if st.button('Recomendar'):
    nombres, posters = recommend(selected_movie)
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.text(nombres[0])
        st.image(posters[0])
    with col2:
        st.text(nombres[1])
        st.image(posters[1])
    with col3:
        st.text(nombres[2])
        st.image(posters[2])
    with col4:
        st.text(nombres[3])
        st.image(posters[3])
    with col5:
        st.text(nombres[4])
        st.image(posters[4])