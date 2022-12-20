from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

models = joblib.load('redrakor.joblib')
df = pd.read_csv('../data/input/tb_redrakor.csv', index_col=None)

def get_index_from_name(judul):
  search_result = df[df['judul']==judul].index.tolist()
  if not search_result:
    return -1
  return search_result[0]

@app.route("/recommendation", methods=['POST'])
def print_similar_drama():
    title = request.form.get('judul')
    if not title:
        return "No title provided"

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
    
    return jsonify(pred.to_dict('records'))

if __name__ == "__main__":
    app.run()