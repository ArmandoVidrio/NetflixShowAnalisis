import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def cleanDataset(dataset):
    ## We drop the unnamed columns
    dataset.drop(columns=dataset.columns[12:], inplace=True)
    ## Set our index
    dataset.set_index('show_id', inplace=True)
    ## Change to datetime type our date
    dataset['date_added'] = pd.to_datetime(dataset['date_added'], format='mixed')

def main():
    st.title("Netflix Shows Information")
    data = pd.read_csv(r'../data/netflix_titles.csv')
    ## We call the function in charge of cleaning the dataset
    cleanDataset(data)
    
    st.write(data)

if __name__ == "__main__":
    main()