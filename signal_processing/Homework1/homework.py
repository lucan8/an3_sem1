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
from ZigZagVec import to_zig_zag_vec, from_zig_zag_vec

#TODO: REFACTOR+OPTIMIZATION: FOR BOTH COMPRESS AND DECOMPRESS
# ELIMINATE ENCODE/DECODE AND KEEP ALL OPERATIONS IN THE SAME FUNCTION, USE VECTORIZATION FOR EACH STEP
# I WILL ASSUME IMAGES ARE SQUARE SHAPED(NXN)
# FOR NOW EVERYTHING IS IN MEMORY

# TODO: WHY DOES DOING CHANNEL WISE MSE AVERAGE DIFFER FROM THE GLOBAL ONE? 

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

# Splits the image into blocks, calls encode_block and returns the resulting list of blocks
def from_image_to_rle_vec_blocks(img: np.ndarray, compr_scale: float = 1) -> list[list[RLE|int]]:
    # Transform the image into a list of blocks
    # Each block is partially compressed just before huffman encoding
    h, w = img.shape[0], img.shape[1]
    Q_JPEG_scaled = Q_JPEG * compr_scale
    block_list = []

    for i in range(0, h, block_h):
        for j in range(0, w, block_h):
            block = img[i:i + block_h, j:j+block_h]

            block_list.append(encode_block(block, Q_JPEG_scaled, True))

    return block_list

# Tranform the block list into a byte array, using huffman encoding for RLE
def from_rle_vec_blocks_to_bytes(block_list: list[list[RLE|int]]):
    global huffman

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
    
    return compr_image
    

# img should be have one channel
# returns the image as series of bytes, compressed
def compress_img(img: np.ndarray, compr_scale: float = 1):
    global block_h, huffman
    print(f"\nInitial image size: {img.__sizeof__()} bytes!\n")

    block_list = from_image_to_rle_vec_blocks(img, compr_scale)
    print(f"\nEncoded all blocks to RLE\n")

    # Build huffman tree and table
    huffman = Huffman(RLE.freq_dict)
    print("\nConstructed huffman table and tree!\n")
    
    compr_img = from_rle_vec_blocks_to_bytes(block_list)
    print(f"\nConverted list of bit buffers to bytearray: {compr_img.__sizeof__()} bytes")
    
    return compr_img

# bytes: compressed image
# Returns the initial image
def decompress_img(bytes: bytearray, compr_scale: float = 1):
    global block_size
    
    block_list = from_bytes_to_rle_blocks(bytes)

    Q_JPEG_scaled = Q_JPEG * compr_scale

    # Create the image from the blocks
    img = np.zeros((img_h, img_h))

    h, w = img.shape[0], img.shape[1]
    block_i = 0

    for i in range(0, h, block_h):
        for j in range(0, w, block_h):
            img[i:i + block_h, j:j+block_h] = decode_block(block_list[block_i], Q_JPEG_scaled)
            block_i += 1

    return img

# Encodes a 8x8 block into a rle vector
# Stats mode also tracks the appearance frequency of rle elements
def encode_block(block: np.ndarray, Q_jpeg_scaled: np.ndarray, stats_mode: bool=False):
    # Apply cosine transform and devide by quantization matrix
    block_dctn = dctn(block)
    block_dctn_jpeg = np.round(block_dctn/Q_jpeg_scaled)

    # Convert to zig-zag vec and then to run length encodings vec
    zig_zag = to_zig_zag_vec(block_dctn_jpeg)
    rle_vec = RLE.from_zig_zag_vec(zig_zag)

    # Stats needed to build huffman table
    if stats_mode:
        RLE.update_rle_freq(rle_vec)

    return rle_vec

# Converts a rle vector to the image patch it represents
def decode_block(block: list[RLE|int], Q_jpeg_scaled: np.ndarray):
    global block_size

    # Convert rle vectors to zig-zag and back matrix block
    block = RLE.to_zig_zag_vec(block, block_size)
    block = from_zig_zag_vec(block)

    # Multiply with quantization matrix and apply cosine tranform inverse
    block = np.array(block, np.float64) * Q_jpeg_scaled
    block = np.astype(idctn(block), np.uint8)

    return block

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

def get_mse(initial, compressed):
    return (np.linalg.norm(initial - compressed) ** 2) / initial.size

def compr_decompr_for_one_ch(img: np.ndarray, compr_scale: float = 1):
    global img_h

    img = img.copy()
    img_h = img.shape[0]

    img_jpeg = decompress_img(compress_img(img, compr_scale), compr_scale)
    mse = get_mse(img, img_jpeg)

    return img_jpeg, mse

def task1(img: np.ndarray):
    img_jpeg, mse = compr_decompr_for_one_ch(img)
    
    fig, axs = plt.subplots(2)
    axs[0].imshow(img, cmap=plt.cm.gray)
    axs[1].imshow(img_jpeg, cmap=plt.cm.gray)
    plt.show()

    print(f"mse={mse}")

def compr_decompr_for_colored(img_ycrcb: np.ndarray, compr_scale: float = 1):
    # Split by channel
    img_y, img_cr, img_cb = img_ycrcb[:, :, 0], img_ycrcb[:, :, 1], img_ycrcb[:, :, 2]

    # Apply compression and decompression on every channel
    img_y_jpeg, mse_y = compr_decompr_for_one_ch(img_y, compr_scale)
    img_cr_jpeg, mse_cr = compr_decompr_for_one_ch(img_cr, compr_scale)
    img_cb_jpeg, mse_cb = compr_decompr_for_one_ch(img_cb, compr_scale)

    print(mse_y, mse_cr, mse_cb)

    # Put them back together
    img_ycrcb_jpeg = img_ycrcb.copy()
    img_ycrcb_jpeg[:, :, 0], img_ycrcb_jpeg[:, :, 1], img_ycrcb_jpeg[:, :, 2] = img_y_jpeg, img_cr_jpeg, img_cb_jpeg

    return img_ycrcb_jpeg, get_mse(img_ycrcb, img_ycrcb_jpeg)

def task2(img: np.ndarray):
    # Convert image to ycrcb
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
    
    img_ycrcb_jpeg, mse = compr_decompr_for_colored(img_ycrcb)

    fig, axs = plt.subplots(2)
    axs[0].imshow(img_ycrcb)
    axs[1].imshow(img_ycrcb_jpeg)
    plt.show()

    print(f"mse={mse}")
    
    return img_ycrcb_jpeg, mse

# Search for the quatization factor that gives the closest mse to the threshold that is smaller than it
def compr_decompr_until_thresh(img: np.ndarray, mse_thresh: np.float64):
    if len(img.shape) == 2:
        compr_func = compr_decompr_for_one_ch
    elif img.shape[2] == 3:
        compr_func = compr_decompr_for_colored
    
    # Binary search the quantization factor
    # decompress + compress, check mse, update factor
    lower_b, upper_b = 0, 100
    chosen_img_jpeg, chosen_mse = None, np.inf
    mse_range = mse_thresh / 10
    while lower_b <= upper_b:
        mid = (lower_b + upper_b) / 2
        
        img_jpeg, mse = compr_func(img, mid)
        print(f"MSE: {mse}, scale: {mid}")

        if mse <= mse_thresh: # Below threshold, go right and keep track of it
            lower_b = mid
            chosen_img_jpeg, chosen_mse = img_jpeg, mse
            if mse_thresh - mse < mse_range:
                break
        else: # Above threshold, go left
            upper_b = mid
    
    return chosen_img_jpeg, chosen_mse

def task3(img: np.ndarray, mse_thresh: np.float64, colored: bool = True):
    # If colored desired, convert to ycrcb
    if colored:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
    
    img_jpeg, mse = compr_decompr_until_thresh(img, mse_thresh)
    fig, axs = plt.subplots(2)
    axs[0].imshow(img)
    axs[1].imshow(img_jpeg)
    plt.show()

    print(f"mse={mse}")

img = datasets.ascent()
# task1(img)
task3(img, 4, False)