from typing import List


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) <= 2:
            return len(nums)

        cnt = 2  # 先计入 nums[0], nums[1]
        v1 = nums[0]  # 前面第二个元素
        v2 = nums[1]  # 前面第一个元素

        for i in range(2, len(nums)):
            # nums[i] != v2, 说明不是重复两次以上的后续元素。
            # 因为数组有序，v1 <= v2。nums[i] 如果 == v1，那么也 == v2
            if nums[i] != v1:
                cnt += 1  # 计数
                nums[cnt - 1] = nums[i]  # 放入 cnt - 1 处
                # 以上两行可以调换简化，但个人更倾向于可读性

            v1 = v2
            v2 = nums[i]

        return cnt

if __name__ == '__main__':
    nums = [1, 1, 2]
    print(Solution().removeDuplicates(nums))  # Output: 3