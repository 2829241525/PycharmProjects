#词典，每个词后方存储的是其词频，仅为示例，也可自行添加
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

max_len = len(max(Dict, key=len))
min_len = len(min(Dict, key=len))
resp=[]

def calc_dag(sentence,temp):
    if not sentence:  # 如果句子已经处理完毕
        resp.append(temp[:])  # 将当前的分词方式加入到全局结果列表中
        return

    for i in range(min_len,max_len+1):
        if i > len(sentence):
            break
        word = sentence[0:i]
        if word in Dict.keys():
            temp.append(word)
            calc_dag(sentence[i:], temp)  # 递归处理剩余的句子部分
            temp.pop()  # 回溯：撤销最近一次加入的词，尝试下一个可能的词
    return temp


# def calc_dag(sentence, temp):
#     if not sentence:  # 如果句子已经处理完毕
#         resp.append(temp[:])  # 将当前的分词方式加入到全局结果列表中
#         return
#
#     for i in range(min_len, max_len + 1):
#         if i > len(sentence):
#             break  # 如果当前词的长度超过剩余句子长度，停止此轮循环
#         word = sentence[0:i]
#         if word in Dict:
#             temp.append(word)
#             calc_dag(sentence[i:], temp)  # 递归处理剩余的句子部分
#             temp.pop()  # 回溯：撤销最近一次加入的词，尝试下一个可能的词


def segment_sentence(sentence):
    calc_dag(sentence, [])
    return resp

sentence = "经常有意见分歧"
#sentence = "经常有意见分歧"
test = []
calc_dag(sentence,test)
print(resp)