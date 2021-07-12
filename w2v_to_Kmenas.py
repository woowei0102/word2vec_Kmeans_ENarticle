from gensim.models import word2vec
from gensim import models
from sklearn.cluster import KMeans
import numpy as np
import joblib
import logging

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = models.Word2Vec.load('word2vec.model')
    # y1 = model.most_similar(positive=['ate', 'speak'], negative=['eat'], topn=5
    # print("y1:", y1)
    #print(len(model.wv.index2word))
    #print('詞表長度：', len(model.wv.vocab))
    ''' word2vec 關鍵字'''
    keys = model.wv.vocab.keys()
    ''' 每個關鍵字的詞向量'''
    wordvector = []
    for key in keys:
        wordvector.append(model[key])
    
    ''' 分類 '''
    classCount = 4 #分類数
    clf = KMeans(n_clusters=classCount)
    s = clf.fit(wordvector)

    ''' 儲存模型 '''
    joblib.dump(s, 'save/kmeans_4.pkl')

    ''' 類別輸出 '''
    labels=clf.labels_
    print('類别：',labels)
    print(type(labels))
    

if __name__ == "__main__":
    main() 