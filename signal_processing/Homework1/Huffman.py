from queue import PriorityQueue, Queue
from BitStream import BitBuffer

class HuffmanNode:
    def __init__(self, left, right):
        self.left = left
        self.right = right

class HuffmanLeaf:
    def __init__(self, val):
        self.val = val

class Huffman:
    def __init__(self, rle_stats):
        self.tree_root = self._build_tree(rle_stats)
        self.table = self._build_table()

    # Builds the huffman tree from rle_stats
    # Returns the root
    def _build_tree(self, rle_stats: dict):
        pq = PriorityQueue()

        for k, v in rle_stats.items():
            pq.put((v, HuffmanLeaf(k)))

        for _ in range(len(rle_stats) - 1):
            prob_left, node_left = pq.get()
            prob_right, node_right = pq.get()

            new_prob = prob_left + prob_right
            new_node = HuffmanNode(node_left, node_right)
            
            pq.put((new_prob, new_node))

        return pq.get()[1]

    # Build the huffman table from the huffman tree
    # Returns the table as a dict
    def _build_table(self):
        table = {}
        
        q = Queue()
        q.put((self.tree_root,  BitBuffer()))

        while not q.empty():
            node, code = q.get()
            
            if isinstance(node, HuffmanNode):
                q.put((node.left, code.append('0')))
                q.put((node.right, code.append('1')))
            elif isinstance(node, HuffmanLeaf):
                table[node.val] = code
        return table
