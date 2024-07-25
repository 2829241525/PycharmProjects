# 导入所需的库
from sklearn.feature_extraction.text import TfidfVectorizer

# 示例文本数据
documents = [
    "COVID-19疫情对全球经济造成重大影响，各国政府采取措施应对疫情带来的挑战。",
    "科技行业持续创新，人工智能技术在医疗、金融等领域展现出巨大潜力。",
    "环保议题引起全球关注，各国加强环境保护措施应对气候变化带来的影响。"
]

# 创建TF-IDF向量化器
tfidf_vectorizer = TfidfVectorizer()

# 对文本数据进行特征提取
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# 打印特征矩阵的形状
print("TF-IDF特征矩阵的形状：", tfidf_matrix.shape)

# 打印特征词汇及其索引
feature_names = tfidf_vectorizer.get_feature_names_out()
print("特征词汇及其索引：")
for idx, feature in enumerate(feature_names):
    print(f"特征词汇索引 {idx}: {feature}")

# 打印每个文档中的特征词汇及对应的TF-IDF值
print("\n每个文档中的特征词汇及对应的TF-IDF值：")
for doc_idx, doc in enumerate(documents):
    print(f"文档 {doc_idx + 1}: {doc}")
    feature_index = tfidf_matrix[doc_idx].nonzero()[1]
    tfidf_scores = tfidf_matrix[doc_idx, feature_index].toarray()[0]
    for i, idx in enumerate(feature_index):
        print(f" - 特征词汇: {feature_names[idx]}, TF-IDF值: {tfidf_scores[i]}")
