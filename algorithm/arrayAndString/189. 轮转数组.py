class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        a = len(nums)
        res = [-1]*a
        if k > len(nums):
            k = k/len(nums)

        for i in range(0,len(nums)):
            if i+k<=len(nums)-1:
                res[i+k] = nums[i]
            else:
                res[i+k-len(nums)] = nums[i]
        return res

if __name__ == '__main__':
    nums = [1,2,3,4,5,7]
    k = 3
    print(Solution().rotate(nums,k))