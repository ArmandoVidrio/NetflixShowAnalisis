# importamos las librerias necesarias
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import the data
amazon = pd.read_csv('./data/amazon_prime_titles.csv')
disney = pd.read_csv('./data/disney_plus_titles.csv')
hulu = pd.read_csv('./data/hulu_titles.csv')
netflix = pd.read_csv('./data/netflix_titles.csv')

# Drop the columns we don't need in order the dataframes are similar
netflix.drop(columns=netflix.columns[12:], inplace=True)
names = ['amazon', 'disney', 'hulu', 'netflix']
dataframes = [amazon, disney, hulu, netflix]

# Drop null values
for dataframe in dataframes:
    dataframe.dropna(subset=['title', 'listed_in', 'description'], inplace=True)

# Create a tags column (key words)
for dataframe in dataframes:
    dataframe['tags'] = dataframe['description'] + ' ' + dataframe['listed_in']

cv = CountVectorizer(max_features=1000, stop_words='english')

# Create a vector list with the tags in each movie
vectores = []
for dataframe in dataframes:
    vectores.append(cv.fit_transform(dataframe['tags'].values.astype('U')).toarray())

# Find the similarity between the movie's vectors
similarities = [cosine_similarity(vectores[i]) for i in range(4)]

# Save the results in pickle files
for i in range(4):
    with open(f'./saved/sim_{names[i]}.pkl', 'wb') as f:
        pickle.dump(similarities[i], f)
    
    df = dataframes[i][['title', 'tags']]
    with open(f'./saved/df_{names[i]}.pkl', 'wb') as f:
        pickle.dump(df, f)

df = pickle.load(open('./saved/df_netflix.pkl', 'rb'))
sim = pickle.load(open('./saved/sim_netflix.pkl', 'rb'))

def recommend(movies):
    index = df[df['title'] == movies].index[0]
    distance = sorted(list(enumerate(sim[index])), reverse=True, key=lambda x: x[1])
    for i in distance[1:6]:
        print(df.iloc[i[0]].title)


recommend('Pokémon the Movie: I Choose You!')