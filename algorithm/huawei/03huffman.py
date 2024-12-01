import heapq

# 定义哈夫曼树的节点结构
class HuffmanNode:
    def __init__(self, weight, left=None, right=None):
        self.weight = weight  # 权值
        self.left = left  # 左子节点
        self.right = right  # 右子节点
        self.height = 0  # 高度，默认是0

    # 定义节点的比较方式，根据权值和高度进行比较
    def __lt__(self, other):
        if self.weight == other.weight:
            return self.height <= other.height  # 当权值相同，高度较小的节点优先
        return self.weight < other.weight  # 按照权值排序

# 生成哈夫曼树并返回根节点
def build_huffman_tree(weights):
    # 使用最小堆来生成哈夫曼树
    # 初始时，每个权值都作为一个独立的叶子节点
    heap = [HuffmanNode(weight=w) for w in weights]
    heapq.heapify(heap)

    # 不断合并最小的两个节点，直到剩下一个节点
    while len(heap) > 1:
        # 取出权值最小的两个节点
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        # 创建一个新的父节点，其权值为左右子节点的权值之和
        new_node = HuffmanNode(weight=left.weight + right.weight, left=left, right=right)
        new_node.height = max(left.height, right.height) + 1  # 父节点的高度是两个子节点中较大高度+1

        # 将新的父节点插回堆中
        heapq.heappush(heap, new_node)

    # 最终堆中只有一个节点，它就是哈夫曼树的根节点
    return heap[0]

# 中序遍历哈夫曼树
def inorder_traversal(root, result):
    if root is not None:
        inorder_traversal(root.left, result)
        result.append(root.weight)
        inorder_traversal(root.right, result)

# 主函数
def huffman_tree(weights):
    # 1. 构建哈夫曼树
    root = build_huffman_tree(weights)

    # 2. 中序遍历哈夫曼树
    result = []
    inorder_traversal(root, result)

    return result

# 示例输入
weights = [5, 3, 8, 10, 2, 6]
result = huffman_tree(weights)
print(result)  # 输出哈夫曼树的中序遍历结果
