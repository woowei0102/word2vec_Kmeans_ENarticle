# 透過K-means訓練Word2Vec(訓練英文新聞文章語句)單字向量之分類模型

## 介紹
自己想作一個如何提升程式碼變數命名，而這部分並沒有太多的相關研究，反而自然語言已經有相對成熟的工具，所以想先透過此word2vec+k-means去作測試，透過word2vec輸出的詞向量再透過K-means模型訓練是否有好的分類效果。

## gensim套件安裝

~~~python
pip install -U gensim
~~~

## 訓練過程

### word2vec 訓練過程
我透過英文新聞的數據透過word2vec去進行訓練，有產出不錯的模型。
1. 取得英文維基數據，本實驗檔案名稱為`enwiki-20200220-pages-articles.xml.bz2`
2. 將下載後的維基數據置於與專案同個目錄，再使用wiki_to_txt.py從 xml 中提取出維基文章
    ~~~python
    py wiki_to_txt.py enwiki-20200220-pages-articles.xml.bz2
    ~~~

3. 使用`gensim` 的 word2vec 模型進行訓練
    ~~~python
    py train.py
    ~~~

4. 測試訓練出的word2vec模型
    ~~~python
    py word2vec_demo.py
    ~~~

    其中，可透過similarity屬性找到兩個字詞之間的相似度。
    ~~~python
    print('woman和man的相似度:' + ' ' + str(model.similarity('woman', 'man')))
    ~~~
### K-means訓練過程
拿word2vec訓練出的向量拿去K-means作訓練，是希望可以透過非監督訓練方式分類出各個種類，但效果極差，不推薦以下訓練方法。

5. 透過word2vec的關鍵詞去訓練出K-means模型

    ~~~
    py w2v_to_Kmenas
    ~~~
    其中，w2v_to_Kmenas程式中有個classCount是需要進行多少分類。
    注意: 請在該檔案夾裡面新增一個`save資料夾`，不然恐會有問題。
6. 測試訓練出的K-means模型
    ~~~
    py demo.py
    ~~~
    下方例子是將woman和man判斷是否有同一分類。
    ~~~Python
    print('woman: ' + str(Kmeans.predict([w2v['woman']]))+', ' + 'man: ' + str(Kmeans.predict([w2v['man']])))
    ~~~

## 結論
word2vec+k-means這方法我覺得是無法得到較好的分類效果，原因可能是k-means方式所導致的，並無法往自己想像的去作分類，所以非監督方式很難達到較好的效果，建議將每個單詞都要Label並透過其它模型訓練才有可能有較好的效果。

## 相關文件
### word2vec
* [以 gensim 訓練中文詞向量](http://zake7749.github.io/2016/08/28/word2vec-with-gensim/)
* [使用自己的语料训练word2vec模型](https://www.jianshu.com/p/0425bfe619c3)
* [NL2Bash](https://github.com/TellinaTool/nl2bash?fbclid=IwAR2DKk4-qRGEJKUOkcnbK1L8fWeLIKJBTiedyV-aQl7fh7q7OAbCwOKw734)
* [深入淺出Word2Vec原理解析](https://www.jishuwen.com/d/pVET/zh-tw)

### K-means
* [尝试word2vec结合k-means实现关键字聚类](https://www.cnblogs.com/birdmmxx/p/12532751.html)
* [Python-计算word2vec向量的分层聚类并将结果绘制为树状图](https://www.coder.work/article/385557)



