import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from streamlit_dynamic_filters import DynamicFilters

data = pd.read_csv(r'../data/netflix_titles.csv')

def cleanDataset(dataset) -> None:
    ## We drop the unnamed columns
    dataset.drop(columns=dataset.columns[12:], inplace=True)
    ## Set our index
    dataset.set_index('show_id', inplace=True)
    ## Change to datetime type our date
    dataset['date_added'] = pd.to_datetime(dataset['date_added'], format='mixed')
    ## We change the nan values
    dataset.fillna('Not avaible', inplace=True)
    ## We cahnge the release_year to string
    dataset['release_year'] = dataset['release_year'].astype('string')

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
    ##user_input = st.sidebar.text_input('Enter your name', 'Your Name')
    ##selected_option = st.sidebar.selectbox('Select an option', ['Option 1', 'Option 2', 'Option 3'])

    ## We create our dynamic table
    gb = GridOptionsBuilder

def main() -> None:
    st.title("Netflix Series Recommender")
    ## We call the function in charge of cleaning the dataset
    cleanDataset(data)
    ## We display our widgets
    loadWidgets()
    ## We display our dataset in the app
    st.write(data)

if __name__ == "__main__":
    main()