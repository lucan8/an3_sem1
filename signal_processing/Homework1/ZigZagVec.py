import numpy as np

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