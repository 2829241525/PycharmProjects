class Solution:
    def simplifyPath(self, path: str) -> str:
        stack = []
        path = path.split("/")

        for item in path:
            if item == "..":
                if stack : stack.pop()
            elif item and item != ".":
                stack.append(item)
        return "/" + "/".join(stack)

if __name__ == '__main__':
    path = "/home/user/Documents/../Pictures"
    print(Solution().simplifyPath(path))