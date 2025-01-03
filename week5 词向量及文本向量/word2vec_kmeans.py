#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型

#01:38  例子：对文章标题训练。对句子训练
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算


    ###################作业代码：获取center和label，计算距离，排序##########################################
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    #构建存储数据结构
    distanceDict = {}
    for i, center in enumerate(centers):
        distance_sum = 0
        for label, value in zip(labels[labels == i], vectors[labels == i]):
            distance_sum += np.linalg.norm(center - value)  #计算与中心的距离，越小说明聚类效果越好
        #存储距离相加后，求平均
        distanceDict[i] = distance_sum/labels[labels == i].size

    #排序，最小的聚类中心距离越小，效果越好
    sorted_distanceDict = sorted(distanceDict.items(), key=lambda x: x[1])
    print("聚类中心距离排序：", sorted_distanceDict)
    top_ten_keys = [item[0] for item in sorted_distanceDict[:10]]
    #######################################################################
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for label, sentences in sentence_label_dict.items():
        if label not in top_ten_keys: #只打印前10个聚类中心的句子
            continue
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

