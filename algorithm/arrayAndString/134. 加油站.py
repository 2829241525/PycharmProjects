class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """

        # for i in range(len(gas)):
        #     temp = 0
        #     resign = 1
        #     for j in range(i,i+len(gas)):
        #         ji = j
        #         if j>=len(gas)-1:
        #             ji = j-len(gas)
        #         temp = temp+gas[ji]-cost[ji]
        #         if temp <0:
        #             resign = 0
        #             break
        #     if resign == 1:
        #         return i
        # return -1
        n = len(gas)
        i = 0
        while i < n:
            sum_of_gas = sum_of_cost = 0
            cnt = 0
            while cnt < n:
                j = (i + cnt) % n
                sum_of_gas += gas[j]
                sum_of_cost += cost[j]
                if sum_of_cost > sum_of_gas:
                    break
                cnt += 1
            if cnt == n:
                return i
            else:
                i += cnt + 1
        return -1


if __name__ == '__main__':
    gas = [1,2,3,4,5]
    cost = [3,4,5,1,2]
    print(Solution().canCompleteCircuit(gas, cost))