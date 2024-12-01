class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        n, rightmost = len(nums), 0

        for i in range(n):
            if i <= rightmost:
                rightmost = max(rightmost, i + nums[i])
                if rightmost >= n - 1:
                    return True
        return False

if __name__ == '__main__':
    nums = [2,2,0,0,3,0,0,4]
    print(Solution().canJump(nums))