from gensim.models import word2vec
from gensim import models
import logging

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = models.Word2Vec.load('word2vec.model')
    a1 = model.similarity('woman', 'man')
    a2 = model.similarity('math', 'add')
    a3 = model.similarity('animal', 'dog')
    print('woman和man的相似度:' + ' ' + str(a1))
    print('math和add的相似度:'+ ' ' + str(a2))
    print('animal和dog的相似度:' + ' ' + str(a3))

if __name__ == "__main__":
    main() 