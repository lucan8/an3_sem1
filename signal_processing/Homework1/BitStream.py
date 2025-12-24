# int.bit_length but returns 1 if val is 0!
def bit_length(val: int):
    if val == 0:
        return 1
    return int(val).bit_length()
    
class BitBuffer:
    BYTE_SIZE = 8
    BYTE_MAX_VAL = 255
    CHUNK_SIZE = 4

    def __init__(self, val: int = 0, bit_count: int = 0):
        if val < 0:
            self.set_neg(val)
        else:
            self.val = val
            self.bit_count = bit_count
            if self.bit_count == 0:
                self.bit_count = bit_length(self.val)
            elif self.bit_count < bit_length(self.val):
                raise RuntimeError("BitBuffer: Invalid argument: bit_count should be < val.bit_length()")
            
    
    # Sets the bitbuffer when val is negative(please pass it only when negative!)
    def set_neg(self, val):
        abs_val = abs(val)
        self.bit_count = bit_length(abs_val)

        val = (1 << self.bit_count) - 1 + val
        self.val = val
    
    # Returns a copy with the added bit
    def append(self, bit: str):
        cp = BitBuffer(self.val, self.bit_count)
        bit = int(bit)

        cp.val = (cp.val << 1) + bit
        cp.bit_count += 1

        return cp
    
    # Returns a copy with the added bits from bit_buffer
    def extend(self, bit_buffer):
        # Convert to bitbuffer if the arg is a number
        if isinstance(bit_buffer, int):
            bit_buffer = BitBuffer(bit_buffer)
        
        cp = BitBuffer(self.val, self.bit_count)
        cp.val = (cp.val << bit_buffer.bit_count) + bit_buffer.val
        cp.bit_count += bit_buffer.bit_count
        
        return cp
    
    # Merge a list of BitBuffers into a big one and return the result
    @staticmethod
    def merge(chunk):
        res = chunk[0]
        
        for i in range(1, len(chunk)):
            res = res.extend(chunk[i])
        
        return res

    # Converts a list of BitBuffer to a list of bytes
    # Also returns the remaining BitBuffer when not divisible by BYTE_SIZE
    @staticmethod
    def from_chunk_to_bytes(chunk):
        if not isinstance(chunk[0], BitBuffer):
            raise RuntimeError(f"BitBuffer: from_chunk_to_bytes: chunks should be a list of BitBuffer but is {type(chunk[0])}")
        curr = chunk[0]
        result = bytearray()

        for i in range(1, len(chunk)):
            curr = curr.extend(chunk[i])
            result.extend(curr.pop_bytes())
        
        return result, curr

    # Takes a list of BitBuffers splits it into chunks, converts each chunk to a bytearray
    # Returns the whole result and the remaining BitBuffer when not divisible by BYTE_SIZE
    @staticmethod
    def from_list_to_bytes(bit_buff_list, chunk_size = CHUNK_SIZE):
        result, rem = BitBuffer.from_chunk_to_bytes(bit_buff_list[0:chunk_size])
        
        # In case we don't get inside the for loop, it will be needed for the last chunk
        i = 0
        for i in range(chunk_size, len(bit_buff_list) - chunk_size, chunk_size):
            arr, rem = BitBuffer.from_chunk_to_bytes([rem] + bit_buff_list[i:i + chunk_size])
            result.extend(arr)
        
        # Last chunk
        arr, rem = BitBuffer.from_chunk_to_bytes([rem] + bit_buff_list[i + chunk_size:])
        result.extend(arr)

        return result, rem

    # Removes the first BYTE_SIZE from the buffer, left to right and returns the corresponding byte
    def pop_byte(self):
        rem_bits = self.bit_count - BitBuffer.BYTE_SIZE

        byte = (self.val & (BitBuffer.BYTE_MAX_VAL << (rem_bits))) >> (rem_bits)
        self.val = self.val & ((1 << (rem_bits)) - 1)
        self.bit_count = rem_bits

        return byte
    
    # Calls pop_bytes until it can't and returns the resulting bytearray
    def pop_bytes(self):
        byte_chunk = bytearray()
        
        while self.bit_count > BitBuffer.BYTE_SIZE:
            byte_chunk.append(self.pop_byte())
        
        return byte_chunk

    def __str__(self):
        actual_bit_count = bit_length(self.val)
        return '0' * (self.bit_count - actual_bit_count) + bin(self.val)
    
# THIS WILL BECOME RELEVANT WHEN(IF) I DECIDE TO ACTUALLY USE FILES
#TODO: Look into byte stuffing
#TODO: Add functionality for cummulation
# class BitStreamWriter:
#     def __init__(self, filename: str):
#         self.bit_buffer = BitBuffer()
#         self.file = open(filename, 'wb')
    
#     # Extend inner bit buffer, extract and write bytes
#     def write(self, bit_buffer: BitBuffer):
#         self.bit_buffer = self.bit_buffer.extend(bit_buffer)
        
#         self.file.write(self.bit_buffer.pop_bytes())