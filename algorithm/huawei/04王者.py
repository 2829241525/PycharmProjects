from itertools import combinations


def min_difference(scores):
    # 计算总评分和
    total_sum = sum(scores)

    # 使用 combinations 生成所有可能的 5 个人的组合
    min_diff = float('inf')  # 初始化为一个很大的数

    # 枚举所有可能的5人组合
    for comb in combinations(scores, 5):
        sum1 = sum(comb)  # 当前组合的评分和
        sum2 = total_sum - sum1  # 剩下的5人组合的评分和
        min_diff = min(min_diff, abs(sum1 - sum2))  # 更新最小差值

    return min_diff


# 示例输入
scores = [5, 1, 8, 3, 4, 6, 7, 10, 9, 2]
result = min_difference(scores)
print(result)  # 输出两组的实力差的最小值
