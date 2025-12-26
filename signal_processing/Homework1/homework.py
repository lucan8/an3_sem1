import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets
from scipy.fft import dctn, idctn, idct
import cv2
import os
from queue import PriorityQueue, Queue
from Huffman import Huffman
from BitStream import BitBuffer
from RLE import RLE, RLEHuffKey


# I WILL ASSUME IMAGES ARE SQUARE SHAPED(NXN)
# FOR NOW EVERYTHING IS IN MEMORY

#TODO: Ask chat why using 1 padding is better than 0 padding or any padding really
#CRITICAL: 
# YOU MIGHT LOSE SOME 0 PADDING BITS ON THE LAST BYTE WHEN COMPRESSING
# FAILS WHEN TAKING THE 64X64 IMAGE ON THE LAST BLOCK

#SOLUTIONS:
# ADD A 1 BIT TO THE LAST BYTE TO THE LEFT THEN REMOVE IT BEFORE CONVERTING TO BIT BUFFER

#TODO: Handle EOB better as it does not always occur when the block ends
# It might also create problems when hashing

#TODO: Look whether you need to process the last block separately

# The max size, in bits of a RLE after it has been encoded
RLE_MAX_BIT_COUNT = 0

huffman = None
img_h = None # image height, it's expected to be equal to width
block_h = 8 # blcok height, it's expected to be equal to width
block_size = block_h * block_h

# Quantization matrix
Q_JPEG = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 28, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]])

def get_mse(initial, compressed):
    return (np.linalg.norm(initial - compressed) ** 2) / initial.size

# Encodes a 8x8 block into a rle vector
# Stats mode also tracks the appearance frequency of rle elements
def encode_block(block: np.ndarray, compr_scale: float = 1, stats_mode: bool=False):
    global Q_JPEG
    h, w = block.shape

    # Scale quantization matrix
    Q_jpeg_scaled = compr_scale * Q_JPEG
    
    # Apply cosine transform and devide by quantization matrix
    block_dctn = dctn(block)
    block_dctn_jpeg = np.round(block_dctn/Q_jpeg_scaled)

    # Convert to zig-zag vec
    zig_zag = to_zig_zag_vec(block_dctn_jpeg)
    
    # Rule length encoding
    rle_vec = RLE.from_zig_zag_vec(zig_zag)

    if stats_mode:
        RLE.update_rle_freq(rle_vec)

    return rle_vec

# def decode_block(rle: list):
#     return idctn(from_zig_zag_vec(from_rle(rle)))

# Returns a BitBuffer that contains both the huffman encoding for the huff table key
# and the non-zero value
# Also updates rle_max_bit_count if needed
def rle_to_bit_buffer(huffman: Huffman, rle: RLE|int):
    global RLE_MAX_BIT_COUNT
    
    # EOB only has huffman encoding
    if rle == RLE.EOB:
        return huffman.table[RLE.EOB]
    
    bb = BitBuffer(rle.non_zero_val)
    huff_encoding = huffman.table[RLEHuffKey(rle.zero_count, bb.bit_count)]
    huff_encoding = huff_encoding.extend(bb)
    
    RLE_MAX_BIT_COUNT = max(RLE_MAX_BIT_COUNT, huff_encoding.bit_count)

    return huff_encoding

def to_zig_zag_vec(mat: np.ndarray):
    mat_size = mat.shape[0]
    diag_count = 2 * mat_size - 2

    row, col = 0, 0
    res = [int(mat[row][col])]
    mov_func = None

    # Take every secondary diagonals
    for it_count in range(diag_count):
        if it_count % 2 == 0:
            if col + 1 < mat_size:
                col += 1
            else:
                row += 1
            mov_func = go_left_down_fill_vec
        else:
            if row + 1 < mat_size:
                row += 1
            else:
                col += 1
            mov_func = go_right_up_fill_vec
        
        res.append(int(mat[row][col]))
        row, col = mov_func(row, col, min(it_count + 1, diag_count - it_count - 1), mat, res)

    return res

def from_zig_zag_vec(vec: list[int]):
    mat_size = int(np.sqrt(len(vec)))
    diag_count = 2 * mat_size - 2
    mat = np.zeros((mat_size, mat_size), np.int64)

    row, col = 0, 0
    vec_i = 0
    mat[row][col] = vec[vec_i]
    mov_func = None

    # Take every secondary diagonals
    for it_count in range(diag_count):
        if it_count % 2 == 0:
            if col + 1 < mat_size:
                col += 1
            else:
                row += 1
            mov_func = go_left_down_fill_mat
        else:
            if row + 1 < mat_size:
                row += 1
            else:
                col += 1
            mov_func = go_right_up_fill_mat
        
        vec_i += 1
        mat[row][col] = vec[vec_i]
        row, col, vec_i = mov_func(row, col, min(it_count + 1, diag_count - it_count - 1), mat, vec, vec_i)

    return mat

# Moves right and up from row, col to col, row, setting the values in mat to the corresponding ones from res
# Returns the new row and col
def go_left_down_fill_mat(row:int, col:int, it_count:int, mat:np.ndarray, vec:list[int], vec_i: int):
    for _ in range(it_count):
        vec_i += 1
        row, col = row + 1, col - 1
        mat[row][col] = vec[vec_i]
    return row, col, vec_i

# Moves right and up from row, col to col, row, setting the values in mat to the corvecponding ones from vec
# Returns the new row and col
def go_right_up_fill_mat(row:int, col:int, it_count:int, mat:np.ndarray, vec:list[int], vec_i: int):
    for _ in range(it_count):
        vec_i += 1
        row, col = row - 1, col + 1
        mat[row][col] = vec[vec_i]
    return row, col, vec_i

# Moves left and down from row, col to col, row, adding the values in mat to vec
# Returns the new row and col
def go_left_down_fill_vec(row:int, col:int, it_count:int, mat:np.ndarray, vec:list[int]):
    for _ in range(it_count):
        row, col = row + 1, col - 1
        vec.append(int(mat[row][col]))
    return row, col

# Moves right and up from row, col to col, row, adding the values in mat to vec
# Returns the new row and col
def go_right_up_fill_vec(row:int, col:int, it_count:int, mat:np.ndarray, vec:list[int]):
    for _ in range(it_count):
        row, col = row - 1, col + 1
        vec.append(int(mat[row][col]))
    return row, col

# img should be have one channel
# returns the image as series of bytes, compressed
def compress_img(img: np.ndarray, compr_scale: float = 1):
    global block_h, huffman

    print(f"\nInitial image size: {img.__sizeof__()} bytes!\n")
    # Transform the image into a list of blocks
    # Each block is partially compressed just before huffman encoding
    h, w = img.shape[0], img.shape[1]
    block_list = []
    for i in range(0, h, block_h):
        for j in range(0, w, block_h):
            block = img[i:i + block_h, j:j+block_h]

            block_list.append(encode_block(block, compr_scale, True))
            
            
    print(f"\nEncoded all blocks to RLE\n")

    # Build huffman tree and table
    huffman = Huffman(RLE.freq_dict)
    print("\nConstructed huffman table and tree!\n")
    
    # Transform RLEs into a bytearray
    # First block
    compr_image, rem = BitBuffer.from_list_to_bytes([rle_to_bit_buffer(huffman, rle) for rle in block_list[0]])

    for i in range(1, len(block_list)):
        # Transform every rle into a bit buffer
        bit_buffers = [rem] + [rle_to_bit_buffer(huffman, rle) for rle in block_list[i]] 
        
        # Transform the list of bit buffers into a list of bytes
        compr_block, rem = BitBuffer.from_list_to_bytes(bit_buffers)

        # Add it to the result
        compr_image.extend(compr_block)

    assert(rem.bit_count <= BitBuffer.BYTE_SIZE)

    # Remaining bits
    if rem.bit_count > 0:
        compr_image.append(rem.pad_with_ones_right())
    
    print(f"\nConverted list of bit buffers to bytearray: {compr_image.__sizeof__()} bytes")
    return compr_image

# Returns a rle object and the remaining bb
def from_bb_to_rle(bit_buffer: BitBuffer) -> tuple[RLE|int, BitBuffer]:
    # Process huffman encoding to get the zero count and bit length of the following non-zero val
    rle_huff_key = huffman.get_rle_huff_key(bit_buffer)

    # Early return for EOB
    if rle_huff_key == RLE.EOB:
        return RLE.EOB, bit_buffer
    
    # Read the non-zero value
    nz_val = bit_buffer.pop_bits(rle_huff_key.bit_count)

    # Convert to int
    nz_val = BitBuffer(nz_val, rle_huff_key.bit_count).to_int()

    return RLE(rle_huff_key.zero_count, nz_val), bit_buffer

# Converts a bytearray to a list of blocks, each block being a list of rles
# Would benefit from nicer code!
def from_bytes_to_rle_blocks(bytes: bytearray) -> list[list[RLE|int]]:
    global RLE_MAX_BIT_COUNT, img_h

    # Determine the expected number of blocks of the initial image
    block_count = (img_h * img_h) // (block_h * block_h) 
    
    # Convert byte array to rles

    # Prepare stuff
    rem = BitBuffer()
    rle_blocks = [[] for _ in range(block_count)]
    rle_block_i = 0
    byte_i = 0

    # Process until all the blocks were recreated
    while rle_block_i < block_count:
        # Merge bytes until we are sure we have enough bits to process a RLE
        while byte_i < len(bytes) and rem.bit_count < RLE_MAX_BIT_COUNT:
            rem = rem.extend(BitBuffer(bytes[byte_i], BitBuffer.BYTE_SIZE))
            byte_i += 1
        
        # Tranform the bitstream into a RLE
        rle, rem = from_bb_to_rle(rem)
        rle_blocks[rle_block_i].append(rle)
        
        # Go to next block
        if rle == RLE.EOB:
            rle_block_i += 1
        
    return rle_blocks

# Converts a rle vector to the image patch it represents
def decode_block(block: list[RLE|int]):
    global block_size

    # Convert rle vectors to zig-zag and back matrix block
    block = RLE.to_zig_zag_vec(block, block_size)
    block = from_zig_zag_vec(block)

    # Multiply with quantization matrix and apply cosine tranform inverse
    block *= Q_JPEG
    block = np.astype(idctn(block), np.uint8)

    return block


def decompress_img(bytes: bytearray):
    global block_size
    block_list = from_bytes_to_rle_blocks(bytes)

    block_list = [decode_block(block) for block in block_list]

    # Create the image from the blocks
    img = np.zeros((img_h, img_h))

    h, w = img.shape[0], img.shape[1]
    block_i = 0

    for i in range(0, h, block_h):
        for j in range(0, w, block_h):
            img[i:i + block_h, j:j+block_h] = block_list[block_i]
            block_i += 1

    return img

def task1(img: np.ndarray):
    global img_h

    img = img.copy()
    img_h = img.shape[0]

    img_jpeg = compress_img(img)

    img_dec = decompress_img(img_jpeg)
    
    fig, axs = plt.subplots(2)
    axs[0].imshow(img, cmap=plt.cm.gray)
    axs[1].imshow(img_dec, cmap=plt.cm.gray)
    plt.show()

    print(f"err={get_mse(img, img_dec)}")

def task2(img: np.ndarray):
    # Convert image to ycrcb
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)

    # Split by channel
    img_y, img_cr, img_cb = img_ycrcb[:, :, 0], img_ycrcb[:, :, 1], img_ycrcb[:, :, 2]

    # Apply compression on every channel
    img_y_jpeg, img_cr_jpeg, img_cb_jpeg = compress_img(img_y), compress_img(img_cr), compress_img(img_cb)

    # Put them back together
    img_ycrc_jpeg = img_ycrcb.copy()
    img_ycrc_jpeg[:, :, 0], img_ycrc_jpeg[:, :, 1], img_ycrc_jpeg[:, :, 2] = img_y_jpeg, img_cr_jpeg, img_cb_jpeg

    fig, axs = plt.subplots(2)
    axs[0].imshow(img_ycrcb)
    axs[1].imshow(img_ycrc_jpeg)
    plt.show()

    print(f"err={get_mse(img_ycrcb, img_ycrc_jpeg)}")

img = datasets.ascent()
# task1(img)
task1(img)