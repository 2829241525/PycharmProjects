class Solution:
    def jump(self, nums) -> int:
        end, max_pos = 0, 0
        steps = 0
        for i in range(len(nums) - 1):
            max_pos = max(max_pos, nums[i] + i)
            if i == end:
                end = max_pos
                steps += 1
        return steps


if __name__ == '__main__':
    nums = [2,3,1,1,4]
    print(Solution().jump(nums))