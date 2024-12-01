class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # answer = [0]*len(nums)

        # for i in range(len(nums)):
        #     temp = 1
        #     for j in range(len(nums)):
        #         if i != j:
        #             if nums[j] == 0:
        #                 temp = 0
        #                 break
        #             temp *=nums[j]
        #     answer[i] = temp
        # return answer
        n = len(nums)
        l = [1] * n
        for i in range(1, n):
            l[i] = l[i - 1] * nums[i - 1]
        r = 1
        for i in range(n - 1, -1, -1):
            l[i] *= r
            r *= nums[i]
        return l

if __name__ == '__main__':
    nums = [1, 2, 3, 4]
    print(Solution().productExceptSelf(nums))