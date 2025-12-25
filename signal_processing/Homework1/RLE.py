import numpy as np
from BitStream import BitBuffer, bit_length

# Run-length encoding
class RLE:
    EOB = -1
    freq_dict = {} # Needed for huffman encoding

    def __init__(self, zero_count: int = 0, non_zero_val: int = 0):
        self.zero_count = zero_count
        self.non_zero_val = non_zero_val
    
    # Converts the zig-zag vector to a vector of rle
    # Optimization ideea: if zero_count is 0 only store the non_zero_val
    @staticmethod
    def from_zig_zag_vec(vec: np.ndarray):
        rle = []
        zero_count = 0

        for elem in vec:
            if elem == 0:
                zero_count += 1
            else:
                rle.append(RLE(zero_count, elem))
                zero_count = 0

        if zero_count > 0:
            rle.append(RLE.EOB)

        return rle
    
    # Converts a vector of RLE back to a zig-zag vector
    @staticmethod
    def to_zig_zag_vec(rle_vec: list):
        res = []
        trailing_zero_count = rle_vec[-1]

        for i in range(len(rle_vec)):
            z_count, non_z_v = rle_vec[i]
            res.extend([0] * z_count)
            res.append(non_z_v)

        res.extend([0] * trailing_zero_count)
        return np.array(res, np.float64)
    
    # Updates the frequency vector of the RLE class
    @staticmethod
    def update_rle_freq(rle_vec: list):
        for i in range(len(rle_vec) - 1):
            rle = rle_vec[i]
            huff_key = rle.get_huff_key()

            if huff_key in RLE.freq_dict:
                RLE.freq_dict[huff_key] += 1
            else:
                RLE.freq_dict[huff_key] = 1
        
        last = rle_vec[-1]
        # Don't forget about the last elem
        if last != RLE.EOB:
            last = last.get_huff_key()
        
        if last in RLE.freq_dict:
            RLE.freq_dict[last] += 1
        else:
            RLE.freq_dict[last] = 1

    def get_huff_key(self):
        return RLEHuffKey(self.zero_count, bit_length(self.non_zero_val))
    
    def __eq__(self, other):
        if not isinstance(other, RLE):
            return False
        
        return self.zero_count == other.zero_count and self.non_zero_val == other.non_zero_val
    
# Key for entries in huffman table
# Only keeps the relevant part of the RLE
class RLEHuffKey:
    def __init__(self, zero_count: int, bit_count: int):
        self.zero_count = zero_count
        self.bit_count = bit_count
    
    def __eq__(self, other):
        if not isinstance(other, RLEHuffKey):
            return False
        return self.zero_count == other.zero_count and self.bit_count == other.bit_count
    
    def __hash__(self):
        return hash((self.zero_count, self.bit_count))