from collections import deque


def find_contacts(n, confirmed, matrix):
    # 用集合来记录需要核酸检测的人
    to_test = set()
    visited = [False] * n  # 用来标记每个人是否被访问过

    # for i in confirmed:
    #     visited = [False] * n
    #     j = matrix[i]
    #     for ik, vk in enumerate(j):
    #         if ik not in confirmed and vk == 1 and not visited[ik]:
    #             to_test.add(ik)
    #             confirmed.add(ik)
    #             visited[ik] = True
    #
    # return len(to_test)

    #
    #
    # 广度优先搜索 BFS
    def bfs(start):
        queue = deque([start])
        visited[start] = True
        while queue:
            person = queue.popleft()
            for i in range(n):
                if matrix[person][i] == 1 and not visited[i]:
                    visited[i] = True
                    to_test.add(i)  # 这个人需要做核酸检测
                    queue.append(i)

    # 对每个确诊病例进行BFS遍历
    for person in confirmed:
        if not visited[person]:  # 如果这个人还没有被访问过
            bfs(person)

    return len(to_test)

#
# # 输入数据
# n = int(input())  # 总人数
# confirmed = list(map(int, input().split(',')))  # 确诊病例的编号，输入格式：X1,X2,X3,...
# matrix = [list(map(int, input().split())) for _ in range(n)]  # 接触矩阵
#
# # 计算需要核酸检测的人数
# result = find_contacts(n, confirmed, matrix)
# print(result)

if __name__ == '__main__':
    n = 5
    confirmed = [1, 2]
    matrix = [[1, 1, 0, 1, 0], [1, 1, 0, 0, 0], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0], [0, 0, 1, 0, 1]]
    result = find_contacts(n, confirmed, matrix)
    print(result)