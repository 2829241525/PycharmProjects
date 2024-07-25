import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from pylab import mpl
import pandas as pd
import numpy as np
import matplotlib

import matplotlib.font_manager as fm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 定义文档集合
documents = [
    "疫情 影响 全球 经济",
    "科技 创新 人工智能",
    "环保 全球 气候变化",
    "教育 改革 质量 公平",
    "新能源汽车 电动汽车 交通"
]

# 计算TF-IDF
vectorizer = TfidfVectorizer(use_idf=True)
tfidf_matrix = vectorizer.fit_transform(documents)

# 获取词汇和IDF值
words = vectorizer.get_feature_names_out()
idf_values = vectorizer.idf_

# IDF值图表
plt.figure(figsize=(10, 6))
plt.bar(range(len(idf_values)), idf_values, tick_label=words)
plt.xticks(rotation=90)
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['axes.unicode_minus'] = False

plt.title('IDF Values for Different Words')
plt.xlabel('Words')
plt.ylabel('IDF Value')
plt.show()

# 计算TF值
tf_matrix = tfidf_matrix.toarray()
# 将TF-IDF矩阵转换为DataFrame，方便作图
df_tf = pd.DataFrame(tf_matrix, columns=words)
# 计算每个词在所有文档中的平均TF值
avg_tf = df_tf.mean()
# TF值图表
plt.figure(figsize=(10, 6))
plt.bar(range(len(avg_tf)), avg_tf, tick_label=words)
plt.xticks(rotation=90)
plt.title('Average TF Values for Different Words')
plt.xlabel('Words')
plt.ylabel('Average TF Value')
plt.show()
