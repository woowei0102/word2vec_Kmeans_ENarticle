from gensim.models import word2vec
from gensim import models
from sklearn.cluster import KMeans
import numpy as np
import joblib
import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
  #  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # 讀取Model
    Kmeans = joblib.load('save/kmeans_2.pkl')
    w2v = models.Word2Vec.load('word2vec.model')
    # print(Kmeans.predict(w2v['win']))

   # labels=Kmeans.labels_
    print('woman: ' + str(Kmeans.predict([w2v['woman']]))+', ' + 'man: ' + str(Kmeans.predict([w2v['man']])))
    print('math: ' + str(Kmeans.predict([w2v['math']]))+', ' + 'add: ' + str(Kmeans.predict([w2v['add']])))
    print('animal: ' + str(Kmeans.predict([w2v['animal']]))+', ' + 'dog: ' + str(Kmeans.predict([w2v['dog']])))

if __name__ == "__main__":
    main() 