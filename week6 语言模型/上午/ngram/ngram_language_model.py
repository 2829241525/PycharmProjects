import math
from collections import defaultdict

##上午1:30，ngram语言模型

# 语言模型的基本思想是，给定一个句子，计算其出现的概率。
# 语言模型的训练过程就是统计语言模型所需要的训练数据，包括训练文本、词表、ngram概率等。
# 语言模型的应用就是根据给定的句子，计算其出现的概率，并根据概率给出相应的句子。

# ngram语言模型是一种统计语言模型，它假设句子是由一系列的词组成，每个词都由前面若干个词决定。
# 它用n-gram模型来建模语言，n表示词的个数，n-gram模型就是用n个词来描述一个词。
# 例如，在一元语言模型中，一个词只依赖于前一个词，而在二元语言模型中，一个词还依赖于前两个词。
# 语言模型的训练过程就是统计语言模型所需要的训练数据，包括训练文本、词表、ngram概率等。

# 语言模型的应用就是根据给定的句子，计算其出现的概率，并根据概率给出相应的句子。
# 语言模型的应用有很多，如文本生成、信息检索、机器翻译、语音识别等。

# 语言模型的训练过程包括：
# 1. 收集训练数据：从语料库中收集大量的句子，并将句子分词。
# 2. 统计词频：统计训练数据中每个词的出现频率。
# 3. 计算ngram概率：根据词频计算ngram概率。
# 4. 利用语言模型：根据语言模型计算句子的概率。

# 语言模型的评估指标有：
# 1. 困惑度（Perplexity）：困惑度是语言模型的评估指标，它表示语言模型预测的句子的困难程度。
# 2. 语言模型的准确率：准确率是语言模型的评估指标，它表示语言模型预测的句子与实际句子的相似程度。
# 3. 语言模型的召回率：召回率是语言模型的评估指标，它表示语言模型预测的句子中有多少是正确的。


# 语言模型的实现
# 语言模型的实现可以分为两步：
# 1. 统计ngram的数量：统计训练数据中每个ngram的出现频率。
# 2. 计算ngram概率：根据词频计算ngram概率。


# 1. 统计ngram的数量
# 统计ngram的数量可以用字典来实现，其中每个键对应一个ngram，值对应出现的次数。
# 例如，给定一个句子"the quick brown fox jumps over the lazy dog"，
# 我们可以统计它的ngram数量，如：
# 1-gram: {"the": 1, "quick": 1, "brown": 1, "fox": 1, "jumps": 1, "over": 1, "lazy": 1, "dog": 1}
# 2-gram: {"the quick": 1, "quick brown": 1, "brown fox": 1, "fox jumps": 1, "jumps over": 1, "over the": 1, "the lazy": 1, "lazy dog": 1}
# 3-gram: {"the quick brown": 1, "quick brown fox": 1, "brown fox jumps": 1, "fox jumps over": 1, "jumps over the": 1, "over the lazy": 1, "the lazy dog": 1}


# 2. 计算ngram概率
# 计算ngram概率可以用贝叶斯公式来实现。
# 贝叶斯公式：P(A|B) = P(B|A) * P(A) / P(B)
# 其中，P(A|B)表示事件A发生的条件下事件B发生的概率，P(B|A)表示事件B发生的条件下事件A发生的概率。
# P(A)表示事件A发生的概率，P(B)表示事件B发生的概率。
# 语言模型的训练数据中，每一个ngram都对应一个词，我们可以用词频来计算ngram的概率。
# 例如，在一元语言模型中，一个词只依赖于前一个词，那么我们可以用词频来计算ngram的概率。
# 例如，在一元语言模型中，"the"的出现频率是1/10，那么"the"的出现的概率就是1/10。
# 例如，在二元语言模型中，"the quick"的出现频率是1/10，那么"the quick"的出现的概率就是1/10。
# 例如，在三元语言模型中，"the quick brown"的出现频率是1/10，那么"the quick brown"的出现的概率就是1/10。


# 3. 利用语言模型
# 利用语言模型可以计算句子的概率。
# 例如，给定一个句子"the quick brown fox jumps over the lazy dog"，
# 我们可以计算它的概率，如：
# 1-gram: P("the"|SOS) * P("quick"|the) * P("brown"|quick) * P("fox"|brown) * P("jumps"|fox) * P("over"|jumps) * P("the"|over) * P("lazy"|the) * P("dog"|lazy) * P(EOS|dog)
# 2-gram: P("the quick"|SOS) * P("brown"|the quick) * P("fox"|brown) * P("jumps"|fox) * P("over"|jumps) * P("the"|over) * P("lazy"|the) * P("dog"|lazy) * P(EOS|dog)
# 3-gram: P("the quick brown"|SOS) * P("fox"|the quick brown) * P("jumps"|fox) * P("over"|jumps) * P("the"|over) * P("lazy"|the) * P("dog"|lazy) * P(EOS|dog)

# 其中，SOS表示句子开始的标识符，EOS表示句子结束的标识符。
# 利用语言模型计算句子的概率时，我们可以用前面的ngram概率来计算后面的ngram概率。
# 例如，在一元语言模型中，"the"的出现的概率是1/10，那么"the"的出现的概率就是1/10。
# 例如，在二元语言模型中，"the quick"的出现的概率是1/10，那么"the quick"的出现的概率就是1/10。
# 例如，在三元语言模型中，"the quick brown"的出现的概率是1/10，那么"the quick brown"的出现的概率就是1/10。

# 最后，我们可以将这些概率相乘，得到整个句子的概率。
# 例如，在一元语言模型中，"the"的出现的概率是1/10，"quick"的出现的概率是1/10，"brown"的出现的概率是1/10，"fox"的出现的概率是1/10，"jumps"的出现的概率是1/10，"over"的出现的概率是1/10，"the"的出现的概率是1/10，"lazy"的出现的概率是1/10，"dog"的出现的概率是1/10，那么"the quick brown fox jumps over the lazy dog"的出现的概率就是1/10。
# 例如，在二元语言模型中，"the quick"的出现的概率是1/10，"brown"的出现的概率是1/10，"fox"的出现的概率是1/10，"jumps"的出现的概率是1/10，"over"的出现的概率是1/10，"the"的出现的概率是1/10，"lazy"的出现的概率是1/10，"dog"的出现的概率是1/10，那么"the quick brown fox jumps over the lazy dog"的出现的概率就是1/10。
# 例如，在三元语言模型中，"the quick brown"的出现的概率是1/10，"fox"的出现的概率是1/10，"jumps"的出现的概率是1/10，"over"的出现的概率是1/10，"the"的出现的概率是1/10，"lazy"的出现的概率是1/10，"dog"的出现的概率是1/10，那么"the quick brown fox jumps over the lazy dog"的出现的概率就是1/10。

# 因此，语言模型可以用来计算一个句子的概率，即给定一个句子，计算其出现的概率。


class NgramLanguageModel:
    def __init__(self, corpus=None, n=3):
        self.n = n
        self.sep = "_"     # 用来分割两个词，没有实际含义，只要是字典里不存在的符号都可以
        self.sos = "<sos>"    #start of sentence，句子开始的标识符
        self.eos = "<eos>"    #end of sentence，句子结束的标识符
        self.unk_prob = 1e-5  #给unk分配一个比较小的概率值，避免集外词概率为0
        self.fix_backoff_prob = 0.4  #使用固定的回退概率
        self.ngram_count_dict = dict((x + 1, defaultdict(int)) for x in range(n))
        self.ngram_count_prob_dict = dict((x + 1, defaultdict(int)) for x in range(n))
        self.ngram_count(corpus)
        self.calc_ngram_prob()

    #将文本切分成词或字或token
    def sentence_segment(self, sentence):
        return sentence.split()
        #return jieba.lcut(sentence)

    #统计ngram的数量
    def ngram_count(self, corpus):
        for sentence in corpus:
            word_lists = self.sentence_segment(sentence)
            word_lists = [self.sos] + word_lists + [self.eos]  #前后补充开始符和结尾符
            for window_size in range(1, self.n + 1):           #按不同窗长扫描文本
                for index, word in enumerate(word_lists):
                    #取到末尾时窗口长度会小于指定的gram，跳过那几个
                    if len(word_lists[index:index + window_size]) != window_size:
                        continue
                    #用分隔符连接word形成一个ngram用于存储
                    ngram = self.sep.join(word_lists[index:index + window_size])
                    self.ngram_count_dict[window_size][ngram] += 1
        #计算总词数，后续用于计算一阶ngram概率
        self.ngram_count_dict[0] = sum(self.ngram_count_dict[1].values())
        return

    #计算ngram概率
    def calc_ngram_prob(self):
        for window_size in range(1, self.n + 1):
            for ngram, count in self.ngram_count_dict[window_size].items():
                if window_size > 1:
                    ngram_splits = ngram.split(self.sep)              #ngram        :a b c
                    ngram_prefix = self.sep.join(ngram_splits[:-1])   #ngram_prefix :a b
                    ngram_prefix_count = self.ngram_count_dict[window_size - 1][ngram_prefix] #Count(a,b)
                else:
                    ngram_prefix_count = self.ngram_count_dict[0]     #count(total word)
                # word = ngram_splits[-1]
                # self.ngram_count_prob_dict[word + "|" + ngram_prefix] = count / ngram_prefix_count
                self.ngram_count_prob_dict[window_size][ngram] = count / ngram_prefix_count
        return

    #获取ngram概率，其中用到了回退平滑，回退概率采取固定值
    def get_ngram_prob(self, ngram):
        n = len(ngram.split(self.sep))
        if ngram in self.ngram_count_prob_dict[n]:
            #尝试直接取出概率
            return self.ngram_count_prob_dict[n][ngram]
        elif n == 1:
            #一阶gram查找不到，说明是集外词，不做回退
            return self.unk_prob
        else:
            #高于一阶的可以回退
            ngram = self.sep.join(ngram.split(self.sep)[1:])
            return self.fix_backoff_prob * self.get_ngram_prob(ngram)


    #回退法预测句子概率
    def calc_sentence_ppl(self, sentence):
        word_list = self.sentence_segment(sentence)
        word_list = [self.sos] + word_list + [self.eos]
        sentence_prob = 0
        for index, word in enumerate(word_list):
            ngram = self.sep.join(word_list[max(0, index - self.n + 1):index + 1])
            prob = self.get_ngram_prob(ngram)
            # print(ngram, prob)
            sentence_prob += math.log(prob)
        return 2 ** (sentence_prob * (-1 / len(word_list)))



if __name__ == "__main__":
    corpus = open("sample.txt", encoding="utf8").readlines()
    lm = NgramLanguageModel(corpus, 3)
    print("词总数:", lm.ngram_count_dict[0])
    print(lm.ngram_count_prob_dict)
    print(lm.calc_sentence_ppl("c d b d b"))
