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

#TODO: Handle EOB better as it does not always occur when the block ends
# It might also create problems when hashing

def get_mse(initial, compressed):
    return (np.linalg.norm(initial - compressed) ** 2) / initial.size

# Encodes a 8x8 block into a rle vector
# Stats mode also tracks the appearance frequency of rle elements
def encode_block(block: np.ndarray, compr_scale: float = 1, stats_mode: bool=False):
    h, w = block.shape
    Q_jpeg = compr_scale * np.array(
            [[16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 28, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]])
    
    block_dctn = dctn(block)
    block_dctn_jpeg = np.round(block_dctn/Q_jpeg)

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
def rle_to_bit_buffer(huffman: Huffman, rle: RLE):
    # print(f"\nRLE size before encoding: {rle.__sizeof__()} bytes!\n")
    
    huff_encoding = huffman.table[rle.get_huff_key()]
    huff_encoding = huff_encoding.extend(rle.non_zero_val)

    # print(f"\nRLE size after encoding: {huff_encoding.bit_count / BitBuffer.BYTE_SIZE} bytes!\n")
    return huff_encoding

def to_zig_zag_vec(mat: np.ndarray):
    mat_size = mat.shape[0]
    diag_count = 2 * mat_size - 2

    row, col = 0, 0
    res = [mat[row][col]]
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
        
        res.append(mat[row][col])
        row, col = mov_func(row, col, min(it_count + 1, diag_count - it_count - 1), mat, res)

    return np.array(res, int)

def from_zig_zag_vec(vec: np.ndarray):
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
def go_left_down_fill_mat(row:int, col:int, it_count:int, mat:np.ndarray, vec:np.ndarray, vec_i: int):
    for _ in range(it_count):
        vec_i += 1
        row, col = row + 1, col - 1
        mat[row][col] = vec[vec_i]
    return row, col, vec_i

# Moves right and up from row, col to col, row, setting the values in mat to the corvecponding ones from vec
# Returns the new row and col
def go_right_up_fill_mat(row:int, col:int, it_count:int, mat:np.ndarray, vec:np.ndarray, vec_i: int):
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
        vec.append(mat[row][col])
    return row, col

# Moves right and up from row, col to col, row, adding the values in mat to vec
# Returns the new row and col
def go_right_up_fill_vec(row:int, col:int, it_count:int, mat:np.ndarray, vec:list[int]):
    for _ in range(it_count):
        row, col = row - 1, col + 1
        vec.append(mat[row][col])
    return row, col

# img should be have one channel
# returns the image as series of bytes, compressed
def compress_img(img: np.ndarray, compr_scale: float = 1):
    print(f"\nInitial image size: {img.__sizeof__()} bytes!\n")
    # Transform the image into a list of blocks
    # Each block is partially compressed just before huffman encoding
    h, w = img.shape[0], img.shape[1]
    block_size = 8
    block_list = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i + block_size, j:j+block_size]

            block_list.append(encode_block(block, compr_scale, True))
            
            
    print(f"\nEncoded all blocks to RLE\n")

    # Build huffman tree and table
    huff = Huffman(RLE.freq_dict)
    print("\nConstructed huffman table and tree!\n")
    
    # First block
    compr_image, rem = BitBuffer.from_list_to_bytes([rle_to_bit_buffer(huff, rle) for rle in block_list[0]])

    for i in range(1, len(block_list)):
        # Transform every rle into a bit buffer
        bit_buffers = [rem] + [rle_to_bit_buffer(huff, rle) for rle in block_list[i]] 
        
        # Transform the list of bit buffers into a list of bytes
        compr_block, rem = BitBuffer.from_list_to_bytes(bit_buffers)

        # Add it to the result
        compr_image.extend(compr_block)

    # Remaining bits
    if rem.bit_count > 0:
        compr_image.extend(rem.val)
    
    print(f"\nConverted list of bit buffers to bytearray: {compr_image.__sizeof__()} bytes")
    return compr_image

# def decompress_img(block_list: np.ndarray):
#     for block in block_list

def task1(img: np.ndarray):
    img_jpeg = compress_img(img)

    fig, axs = plt.subplots(2)
    axs[0].imshow(img, cmap=plt.cm.gray)
    axs[1].imshow(img_jpeg, cmap=plt.cm.gray)
    plt.show()

    print(f"err={get_mse(img, img_jpeg)}")

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