import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets
from scipy.fft import dctn, idctn, idct
import cv2
import os
from queue import PriorityQueue, Queue
from Huffman import Huffman

# I WILL ASSUME IMAGES ARE SQUARE SHAPED(NXN)
# FOR NOW EVERYTHING IS IN MEMORY

EOB = -1
rle_freq = {} # Needed for huffman encoding
    

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
    block_dctn_jpeg = Q_jpeg*np.round(block_dctn/Q_jpeg)

    # Convert to zig-zag vec
    zig_zag = to_zig_zag_vec(block_dctn_jpeg)
    
    # Rule length encoding
    rle = to_rle(zig_zag)

    if stats_mode:
        update_rle_freq(rle)

    return rle

def decode_block(rle: list):
    return idctn(from_zig_zag_vec(from_rle(rle)))

def update_rle_freq(rle_vec: list):
    global rle_freq
    last = rle_vec.pop()
    
    for zero_count, bit_count, _ in rle_vec:
        p = (zero_count, bit_count)
        if p in rle_freq:
            rle_freq[p] += 1
        else:
            rle_freq[p] = 1
    
    # Don't forget about the last elem
    if last != EOB:
        last = (last[0], last[1])
    
    if last in rle_freq:
        rle_freq[last] += 1
    else:
        rle_freq[last] = 1


# returns a vector of pairs in the form (zeros_count, bit_count_non_zero_value, non_zero_value)
# negative values become (1 << val.bit_len) - 1 + abs(val)
def to_rle(vec: np.ndarray):
    rle = []
    zero_count = 0

    for elem in vec:
        if elem == 0:
            zero_count += 1
        else:
            elem = abs(elem)
            bit_len = elem.bit_length()

            if elem < 0:
                elem = (1 << bit_len) - 1 + elem

            rle.append((zero_count, bit_len, elem))
            zero_count = 0

    if zero_count > 0:
        rle.append(EOB)
    return rle

def from_rle(rle: list):
    res = []
    trailing_zero_count = rle[-1]
    rle.pop()

    for z_count, non_z_v in rle:
        res.extend([0] * z_count)
        res.append(non_z_v)

    res.extend([0] * trailing_zero_count)
    return np.array(res, np.float64)

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

    return np.array(res, np.float64)

def from_zig_zag_vec(vec: np.ndarray):
    mat_size = int(np.sqrt(len(vec)))
    diag_count = 2 * mat_size - 2
    mat = np.zeros((mat_size, mat_size), np.float64)

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
# returns a list of compressed blocks representing the image
def compress_img(img: np.ndarray, compr_scale: float = 1, stats_mode: bool = False):
    # Transform the image into a list of blocks
    # Each block is partially compressed just before huffman encoding
    h, w = img.shape[0], img.shape[1]
    block_size = 8
    block_list = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i + block_size, j:j+block_size]
            block_list.append(encode_block(block, compr_scale, stats_mode))

    # Build huffman tree and table
    huff = Huffman(rle_freq)
    
    block_list = np.squeeze(np.asarray([[huff.table[elem] for elem in block] for block in block_list]))

    return block_list

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
task2(img)

freq_of_freq = {}
for k, v in rle_freq.items():
    if v in freq_of_freq:
        freq_of_freq[v] += 1
    else:
        freq_of_freq[v] = 1

sorted_freq = sorted(freq_of_freq.items(), key=lambda x: -x[0])
for k, v in sorted_freq:
    print(k, v)