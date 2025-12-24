from queue import PriorityQueue, Queue
from BitStream import BitBuffer
import itertools

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
        if len(rle_stats) == 0:
            raise RuntimeWarning("Huffman._build_tree: rle_stats empty")
        
        # Needed in case frequencies match, just assigns consecutive unique numbers
        counter = itertools.count()
        pq = PriorityQueue()

        # Initial queue only has the elements of rle stats
        for k, v in rle_stats.items():
            pq.put((v, next(counter), HuffmanLeaf(k)))

        # Extract the most improbable events and merge them, also putting them back in the pq
        while pq.qsize() > 1:
            freq_left, _, node_left = pq.get()
            freq_right, _, node_right = pq.get()

            new_freq = freq_left + freq_right
            new_node = HuffmanNode(node_left, node_right)
            
            pq.put((new_freq, next(counter), new_node))

        # They said the function is unreliable!
        assert(pq.qsize() == 1)
    
        print("\nHuffman Tree Build!\n")
        return pq.get()[2]

    # Build the huffman table from the huffman tree
    # Returns the table as a dict
    def _build_table(self):
        table = {}
        
        q = Queue()
        q.put((self.tree_root,  BitBuffer()))

        # Extract curr node and it's code and append 0/1 for left/right neighbours
        # Or add the entry to the table for leafs
        while not q.empty():
            node, code = q.get()
            
            if isinstance(node, HuffmanNode):
                q.put((node.left, code.append('0')))
                q.put((node.right, code.append('1')))
            elif isinstance(node, HuffmanLeaf):
                table[node.val] = code
        
        # Making sure all encodings are distinct!
        codes = table.values()
        assert(len(codes) == len(set(codes)))

        print("\nHuffman Table build\n")
        return table
