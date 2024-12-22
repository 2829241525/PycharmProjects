class Solution:
    def maxArea(self, height) -> int:
        start = 0
        end = len(height) - 1

        for i in range(len(height) - 1):
            if height[i] < height[i + 1] and (end - start) * min(height[start], height[end]) < (end - start - 1) * min(
                    height[i + 1], height[end]):
                start += 1
            else:
                break

        for j in range(len(height)-1, 0,-1):
            temp1 = (end - start) * min(height[start], height[j])
            temp2 = (end - start - 1) * min(height[start], height[j - 1])
            print(temp1)
            print(temp2)
            if height[j] < height[j - 1] and (end - start) * min(height[start], height[j]) < (end - start - 1) * min(
                    height[start], height[j - 1]):
                end -= 1
            else:
                break

        return (end - start) * min(height[start], height[end])

if __name__ == '__main__':
    height = [8,7,2,1]

    for j in range(len(height) - 1, 0, -1):
        print(j)
    print(Solution().maxArea(height))