import joblib
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import requests
import urllib.parse

# models = joblib.load('redrakor.joblib')
df = pd.read_csv('tb_redrakor.csv', index_col=None)

def get_index_from_name(judul):
  search_result = df[df['judul']==judul].index.tolist()
  if not search_result:
    return -1
  return search_result[0]

def fetch_poster(judul):
  url = "https://www.omdbapi.com/?i=tt3896198&apikey=cf1b6d9f&t={}".format(judul)
  data = requests.get(url).json()

  if data['Response'] == 'False':
    return None

  return data['Poster']

def run():
    st.title("Rekomendasi Drama Korea")

    title = st.text_input("Judul", "Move to Heaven")
    
    df_features = pd.concat([df['genre'].str.strip().str.get_dummies(sep=','), df[['rating']],df[['tahun']],df[['episode']]],axis=1)
    df['judul'] = df['judul'].map(lambda judul:re.sub('[^A-Za-z0-9]+',' ', judul))

    min_max_scaler = MinMaxScaler()
    df_features = min_max_scaler.fit_transform(df_features)

    models = joblib.load('redrakor.joblib')

    distances, indices = models.kneighbors(df_features)

    prediction = []
    found_rank = get_index_from_name(title)
    for rank in indices[found_rank][1:]:
      prediction.append(df.iloc[rank]['judul'])
        
    pred = df[df['judul'].isin(pd.Series(prediction))]
    print(pred)

    if st.button("Berikan saya rekomendasi drakor"):
        for index, data in pred.iterrows():
          poster = fetch_poster(data['judul'])
          link = "https://www.themoviedb.org/search?query={}".format(urllib.parse.quote_plus(data['judul']))
          if poster is None:
            st.image("https://skydomepictures.com/wp-content/uploads/2018/08/movie-poster-coming-soon-2.png")
            st.text(data['judul'])
            st.write('Lihat selengkapnya {}'.format(link))
          else:
            st.image(poster)
            st.text(data['judul'])
            st.write('Lihat selengkapnya {}'.format(link))

if __name__ == "__main__":
    run()