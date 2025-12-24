class BitBuffer:
    BYTE_SIZE = 8
    BYTE_MAX_VAL = 255

    def __init__(self, val: int = 0, bit_count: int = 0):
        self.val = val
        self.bit_count = bit_count
    
    # Returns a copy with the added bit
    def append(self, bit: str):
        cp = BitBuffer(self.val, self.bit_count)
        bit = int(bit)

        cp.val = cp.val << 1 + bit
        cp.bit_count += 1

        return cp
    
    # Returns a copy with the added bits from bit_buffer
    def extend(self, bit_buffer):
        # Convert to bitbuffer if the arg is a number
        if isinstance(bit_buffer, int):
            bit_buffer = BitBuffer(bit_buffer, bit_buffer.bit_length())
        
        cp = BitBuffer(self.val, self.bit_count)
        cp.val = (cp.val << bit_buffer.bit_count) + bit_buffer.val
        cp.bit_count += bit_buffer.bit_count
        
        return cp
    
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
        actual_bit_count = self.val.bit_length()
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