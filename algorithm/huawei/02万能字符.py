from collections import Counter


def count_words(words, chars):
    # 计算chars中每个字母的频次，包括万能字符'?'的频次
    char_counter = Counter(chars)

    # 计算能够掌握的单词数量
    mastered_count = 0

    for word in words:
        word_counter = Counter(word)  # 当前单词字母频次

        # 检查能否用chars拼写该单词
        # 对每个字母，如果chars中没有足够的数量，或者万能字符也不能补足，返回False
        remaining_question_marks = char_counter['?']  # 存储剩余的万能字符数量

        can_master = True
        for letter, count in word_counter.items():
            if char_counter[letter] < count:
                required_question_marks = count - char_counter[letter]
                if remaining_question_marks >= required_question_marks:
                    remaining_question_marks -= required_question_marks
                else:
                    can_master = False
                    break

        if can_master:
            mastered_count += 1

    return mastered_count


# 示例
words = ["abc", "de", "fgh"]
chars = "abc??d"
result = count_words(words, chars)
print(result)  # 输出掌握的单词数
