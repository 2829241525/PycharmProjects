class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """

        n = len(nums)
        ans = n + 1  # 也可以写 inf
        s = left = 0
        for right, x in enumerate(nums):  # 枚举子数组右端点
            s += x
            while s - nums[left] >= target:  # 尽量缩小子数组长度
                s -= nums[left]
                left += 1  # 左端点右移
            if s >= target:
                ans = min(ans, right - left + 1)
        return ans if ans <= n else 0

if __name__ == '__main__':
    s = Solution()
    print(s.minSubArrayLen(9, [1, 2, 3, 4, 5]))
