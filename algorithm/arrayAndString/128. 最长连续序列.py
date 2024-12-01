from typing import List


class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        res = 0     # 记录最长连续序列的长度
        num_set = set(nums)     # 记录nums中的所有数值
        for num in num_set:
            # 如果当前的数是一个连续序列的起点，统计这个连续序列的长度
            if (num - 1) not in num_set:
                seq_len = 1     # 连续序列的长度，初始为1
                while (num + 1) in num_set:
                    seq_len += 1
                    num += 1    # 不断查找连续序列，直到num的下一个数不存在于数组中
                res = max(res, seq_len)     # 更新最长连续序列长度
        return res

if __name__ == '__main__':
    nums = [100, 4, 200, 1, 3, 2]
    print(Solution().longestConsecutive(nums))
