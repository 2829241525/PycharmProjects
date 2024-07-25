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
    "COVID-19疫情对全球经济造成重大影响，各国政府采取措施应对疫情带来的挑战。",
    "科技行业持续创新，人工智能技术在医疗、金融等领域展现出巨大潜力。",
    "环保议题引起全球关注，各国加强环境保护措施应对气候变化带来的影响。",
    "教育改革成为社会热议话题，各国致力于提升教育质量和公平性。",
    "新能源汽车市场蓬勃发展，电动汽车成为未来交通发展的重要方向。"
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
