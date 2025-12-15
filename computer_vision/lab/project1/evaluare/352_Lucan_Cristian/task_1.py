import cv2 as cv
import numpy as np
import os
import shutil
from copy import deepcopy

# TODO: Replace thr_and_clean with simple thresholding

parent_dir = os.path.dirname(__file__)
parent_parent_dir = os.path.dirname(parent_dir)
parent_parent_parent_dir = os.path.dirname(parent_parent_dir)
circ_dic = {}
vert_dic = {}

EMPTY_CELL = "0"
BONUS_CELL_1 = "-1"
BONUS_CELL_2 = "-2"
CIRCLE_PIECE_D = "1"
CLOVER_PIECE_D = "2"
RHOMB_PIECE_D = "3"
SQUARE_PIECE_D = "4"
STAR_4_PIECE_D = "5"
STAR_8_PIECE_D = "6"


WHITE_PIECE_D = "W"
BLUE_PIECE_D = "B"
RED_PIECE_D = "R"
YELLOW_PIECE_D = "Y"
ORANGE_PIECE_D = "O"
GREEN_PIECE_D = "G"

QWIRKLE_BONUS = 6

BOARD_SHAPE = (16, 16)


BLACK_VAL = 0
WHITE_VAL = 255


class Template:
    def __init__(self, image_path):
        filename = image_path.split('\\')[-1]
        
        self.filename = filename
        self.nr = filename[:filename.find('.')]
        self.img = thr_and_clean(cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2HSV))
        self.img, _ = keep_strongest_conn_comp(self.img)
        self.cnt = max(cv.findContours(self.img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0], key=cv.contourArea)

    def match_cell_by_cnt(self, cell_cs):
        return cv.matchShapes(cell_cs, self.cnt, cv.CONTOURS_MATCH_I1, 0)
    
    # Counts the number of matching pixels
    def match_cell_by_img(self, cell):
        template = cv.resize(self.img, (cell.shape[1], cell.shape[0]))
        res = 0
        for row in range(cell.shape[0]):
            for col in range(cell.shape[1]):
                res += cell[row][col] == template[row][col]
        return res


def show_image(title,image):
    image=cv.resize(image,(600,600),fx=0.3,fy=0.3)
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def get_board_coords(image):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_m_blur = cv.medianBlur(image_gray, 9)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5) 
    image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)
   
    # show_image('image_sharpened',image_sharpened)
    _, thresh = cv.threshold(image_sharpened, 33, 255, cv.THRESH_BINARY)
    # show_image("image_threshold", thresh)

    kernel = np.ones((7, 7), np.uint8)
    thresh = cv.erode(thresh, kernel)

    # show_image('image_thresholded_eroded',thresh)
    thresh = cv.erode(thresh, kernel)
    # show_image('image_thresholded_eroded',thresh)

    edges = cv.Canny(thresh , 60, 200, apertureSize=5, L2gradient=True)
    # show_image('edges with appertureSize=5 and L2gradient=true',edges)
    contours, _ = cv.findContours(edges,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0

    for i in range(len(contours)):
        if(len(contours[i]) >3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis = 1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    # image_copy = cv.cvtColor(image.copy(),cv.COLOR_GRAY2BGR)
    # cv.circle(image,tuple(top_left),20,(0,0,255),-1)
    # cv.circle(image,tuple(top_right),20,(0,0,255),-1)
    # cv.circle(image,tuple(bottom_left),20,(0,0,255),-1)
    # cv.circle(image,tuple(bottom_right),20,(0,0,255),-1)
    # show_image("detected corners",image)

    return top_left, top_right, bottom_left, bottom_right

def extract_board(image, pts):
    # pts = four corner points in order: TL, TR, BR, BL
    pts = np.array(pts, dtype="float32")

    # compute width
    width_top = np.linalg.norm(pts[1] - pts[0])
    width_bottom = np.linalg.norm(pts[2] - pts[3])
    board_width = int(max(width_top, width_bottom))

    # compute height
    height_left = np.linalg.norm(pts[3] - pts[0])
    height_right = np.linalg.norm(pts[2] - pts[1])
    board_height = int(max(height_left, height_right))

    # destination points = perfect rectangle
    dst = np.array([
        [0, 0],
        [board_width - 1, 0],
        [board_width - 1, board_height - 1],
        [0, board_height - 1]
    ], dtype="float32")

    # compute perspective transform
    M = cv.getPerspectiveTransform(pts, dst)

    # warp
    warped = cv.warpPerspective(image, M, (board_width, board_height))

    return warped

# Returns the image, keeping only the stronges connected component
def keep_strongest_conn_comp (image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image)
    sizes = stats[:, -1]

    if sizes.shape[0] <= 1:
        return image, (image.shape[0] // 2, image.shape[1] // 2)
    
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape, dtype=np.uint8)
    img2[output == max_label] = 255
    return img2, centroids[max_label]
    

# Note: Expects hsv and does thr based on V
def thr_and_clean(cell):
    # Extract Value
    V = cell[:, :, 2]

    # Threshold
    _, V_thresh = cv.threshold(V, 80, 255, cv.THRESH_BINARY)

    # Clean-up
    # kernel = np.ones((3, 3))
    # V_clean = cv.erode(V_thresh, kernel)
    # V_clean = cv.erode(V_clean, kernel)

    # V_clean = cv.dilate(V_clean, kernel)
    # V_clean = cv.dilate(V_clean, kernel)

    # V_clean = cv.erode(V_clean, kernel)
    # V_clean = cv.erode(V_clean, kernel)

    return V_thresh

def extract_cells(board, N=BOARD_SHAPE[0]):
    h, w = board.shape[:2]
    cell_h = h // N
    cell_w = w // N
    
    cells = []
    for row in range(N):
        for col in range(N):
            # Construct a guess bounding box for the cell
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w

            # show_image("Without centroid calculation", board[y1:y2, x1:x2])
            # Extract guess cell
            cell = cv.cvtColor(board[y1:y2, x1:x2], cv.COLOR_BGR2HSV)

            # Clean and keep only the most important comp
            cell_V_clean = thr_and_clean(cell)
            cell_V_important, centroid_local = keep_strongest_conn_comp(cell_V_clean)
            # show_image("Thresh and strongest comp", cell_V_important)

            x_centroid_local, y_centroid_local = int(centroid_local[0]), int(centroid_local[1])

            # Reconstruct the cell based on centroid
            x_centroid_global, y_centroid_global = x1 + x_centroid_local, y1 + y_centroid_local
            y1 = max(0, y_centroid_global - cell_h // 2)
            y2 = min(y_centroid_global + cell_h // 2, h)
            x1 = max(0, x_centroid_global - cell_w // 2)
            x2 = min(x_centroid_global + cell_w // 2, w)

            cell = board[y1:y2, x1:x2]
            # cv.circle(cell,(x_centroid_local, y_centroid_local),10,(0,0,0),-1)
            # show_image("With centroid calculation", cell)
            cell = cv.cvtColor(board[y1:y2, x1:x2], cv.COLOR_BGR2HSV)
            cells.append(((row, col), cell))

    return cells


# Soft empty means either EMPTY_CELL, BONUS_CELL_1, BONUS_CELL_2
def is_cell_soft_empty(cell_d):
    return cell_d in [EMPTY_CELL, BONUS_CELL_1, BONUS_CELL_2]

# Classify cells and return updated board with all the cells, the added cells and the score
def classify_cells(cells, board: list[list[str]], templates: list[Template], check_bonus:bool):
    new_board = deepcopy(board)
    added_cells = []
    score = 0
    for i, ((row, col), cell) in enumerate(cells):
        if is_cell_soft_empty(new_board[row][col]):
            new_cell = classify_cell(cell, templates, check_bonus)
            # Only update if needed
            if new_cell != EMPTY_CELL:
                new_board[row][col] = new_cell
                added_cells.append([row, col, new_cell])

    if not check_bonus:
        score = calc_score_for_added_cells(board, new_board, added_cells)
    return new_board, added_cells, score

def calc_score_for_added_cells(old_board: list[list[str]], new_board: list[list[str]], added_cells: list[tuple]):
    score = 0
    visited = set()
    h, w = len(old_board), len(old_board[0])
    for row, col, cell in added_cells:
        # Update score for special cells
        if old_board[row][col] == BONUS_CELL_1:
            score += 1
        elif old_board[row][col] == BONUS_CELL_2:
            score += 2

        visited.add((row, col))
        vert_p = 0
        hor_p = 0

        # Go left
        col_cp = col - 1
        while 0 <= col_cp and not is_cell_soft_empty(new_board[row][col_cp]) and (row, col_cp) not in visited:
            visited.add((row, col_cp))
            col_cp -= 1
        
        hor_p += col - col_cp - 1
        
        # Go right
        col_cp = col + 1
        while col_cp < w and not is_cell_soft_empty(new_board[row][col_cp]) and (row, col_cp) not in visited:
            visited.add((row, col_cp))
            col_cp += 1
        
        # -1 cause we added the current piece twice
        hor_p += col_cp - col - 1

        # Only add the piece if a valid row was formed
        if hor_p > 0:
            hor_p += 1

        # Add horizonta; line to score and bonus if QWIRKLE
        score += hor_p + QWIRKLE_BONUS * (QWIRKLE_BONUS == hor_p)

        # Go down
        row_cp = row - 1
        while 0 <= row_cp and not is_cell_soft_empty(new_board[row_cp][col]) and (row_cp, col) not in visited:
            visited.add((row_cp, col))
            row_cp -= 1
        
        vert_p += row - row_cp - 1
        
        # Go up
        row_cp = row + 1
        while row_cp < h and not is_cell_soft_empty(new_board[row_cp][col]) and (row_cp, col) not in visited:
            visited.add((row_cp, col))
            row_cp += 1
        
        # -1 cause we added the current piece twice
        vert_p += row_cp - row - 1

        # Only add the piece if a valid col was formed
        if vert_p > 0:
            vert_p += 1

        # Add vertical line to score and bonus if QWIRKLE
        score += vert_p + QWIRKLE_BONUS * (QWIRKLE_BONUS == vert_p)
    
    return score

# Not used at the moment, but I used once to observe their differences
def track_circularity_and_vert_count(cs, temp):
    area = cv.contourArea(cs)
    perim = cv.arcLength(cs, True)
    circularity = 4 * np.pi * area / (perim * perim)

    if temp in circ_dic:
        circ_dic[temp].append(circularity)
    else:
        circ_dic[temp] = []

    epsilon = 0.02 * perim
    approx_points = cv.approxPolyDP(cs, epsilon, True)

    if temp in vert_dic:
        vert_dic[temp].append(approx_points.shape[0])
    else:
        vert_dic[temp] = []

# Not used at the moment but part of the documentaiton
dic_empty_class_mean_V_max = {0 : 0, 1 : 0}
dic_empty_class_mean_V_min = {0: np.inf, 1 : np.inf}

def has_piece(V_thresh):
    # Compute the mean of the value in the thresholded image
    V_mean = np.mean(V_thresh)

    # Empty cell/bonus points cell
    res = V_mean < 170

    return res

# Returns either EMPTY, BONUS_CELL_1, BONUS_CELL_2
def classify_bonus(S):
    _, S_thresh = cv.threshold(S, 60, 255, cv.THRESH_BINARY)

    # Cut equally from all directions to get rid of noise
    shape = np.array(S_thresh.shape)
    w1, h1 = ((0.1) * shape).astype(np.uint8)
    w2, h2 = ((0.9) * shape).astype(np.uint8)

    S_clean = np.zeros(shape)
    S_clean[h1:h2, w1:w2] = S_thresh[h1:h2, w1:w2]
    S_important, centroid = keep_strongest_conn_comp(S_clean)

    S_mean = np.mean(S_important)

    # Low mean -> empty cell
    if S_mean < 20:
        return EMPTY_CELL
    else:
        # Classify bonus points based circularity
        cell_cs = max(cv.findContours(S_important, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0], key=cv.contourArea)
        area = cv.contourArea(cell_cs)
        perim = cv.arcLength(cell_cs, True)
        circularity = 4 * np.pi * area / (perim * perim)
        
        # I was expecting 2 to be more circular, but it seems like not
        if circularity < 0.4:
            return BONUS_CELL_2
        else:
            return BONUS_CELL_1

# Returns one of the shape constants
def classify_shape(V_thresh: np.ndarray, templates: list[Template]):
    cell_cs = max(cv.findContours(V_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0], key=cv.contourArea)

    # Choose the best template
    matches = np.array([cv.matchShapes(cell_cs, t.cnt, cv.CONTOURS_MATCH_I1, 0) for t in templates])
    chosen_template = str(np.argmin(matches) + 1)
    # print(f"Initial template:{chosen_template}")

    # Usual missclassification happens between rhombs and square, with wild appearnces of circles
    if chosen_template in [RHOMB_PIECE_D, SQUARE_PIECE_D]:
        epsilon = 0.02 * cv.arcLength(cell_cs, True)
        approx_points = cv.approxPolyDP(cell_cs, epsilon, True)

        # Choose from the other best candidates
        if approx_points.shape[0] > 6:
            matches[int(RHOMB_PIECE_D) - 1] = 1
            matches[int(SQUARE_PIECE_D) - 1] = 1
            chosen_template = str(np.argmin(matches) + 1)
        else:
            # Compute diff between neighbouring points
            diff = approx_points[1][0] - approx_points[0][0]

            # Neighbour edges on the same axis -> square
            if abs(diff[0]) < 20 or abs(diff[1]) < 20:
                chosen_template = SQUARE_PIECE_D
            else: # Rhomb
                chosen_template = RHOMB_PIECE_D
    # Comparing circularity for misclassifications between clovers, 8-stars and circles
    if chosen_template in [CLOVER_PIECE_D, STAR_8_PIECE_D, CIRCLE_PIECE_D]:
        area = cv.contourArea(cell_cs)
        perim = cv.arcLength(cell_cs, True)
        circularity = 4 * np.pi * area / (perim * perim)

        if circularity > 0.6: chosen_template = CIRCLE_PIECE_D
        elif circularity > 0.3: chosen_template = CLOVER_PIECE_D
        else: chosen_template = STAR_8_PIECE_D

    track_circularity_and_vert_count(cell_cs, chosen_template)
    return chosen_template

# Not used at the moment, but I used them for the documentation!
color_dic_V_min = {}
color_dic_V_max = {}

color_dic_S_min = {}
color_dic_S_max = {}

color_dic_H_min = {}
color_dic_H_max = {}

# Returns one of the color constants
def classify_color(H: np.ndarray, V: np.ndarray, S: np.ndarray):
    # Compute mean for each axis
    shape_mean_h = np.mean(H)
    shape_mean_s = np.mean(S)
    shape_mean_v = np.mean(V)
    
    # Classify piece color
    chosen_color = WHITE_PIECE_D
    if shape_mean_s < 50: chosen_color =  WHITE_PIECE_D
    elif 0 <= shape_mean_h < 18: chosen_color =  ORANGE_PIECE_D
    elif 18 <= shape_mean_h < 35: chosen_color =  YELLOW_PIECE_D
    elif 35 <= shape_mean_h < 85: chosen_color =  GREEN_PIECE_D
    elif 85 <= shape_mean_h < 140: chosen_color =  BLUE_PIECE_D
    elif shape_mean_h >= 140: chosen_color = RED_PIECE_D

    return chosen_color


def classify_cell(cell, templates: list[Template], check_bonus: bool):
    H, S, V = cell[:, :, 0], cell[:, :, 1], cell[:, :, 2]

    # Threshold
    _, V_thresh = cv.threshold(V, 80, 255, cv.THRESH_BINARY)

    # Clean-up
    kernel = np.ones((5, 5))
    V_clean = cv.erode(V_thresh, kernel)
    V_clean = cv.dilate(V_clean, kernel)

    # Check empty/bonus cell
    if not has_piece(V_thresh):
        if check_bonus:
            return classify_bonus(S)
        else:
            return EMPTY_CELL
    
    # Keep only the strongest connected component
    V_important, centroid = keep_strongest_conn_comp(V_clean)

    # Get shape descriptor
    shape_d = classify_shape(V_important, templates)

    # Extract shape
    shape_mask = V_important == WHITE_VAL

    # Get color descriptor
    color_d = classify_color(H[shape_mask], V[shape_mask], S[shape_mask])

    # print(f"Final template: {chosen_template}")
    return shape_d + color_d 


def print_matrix(mat):
    for row in mat:
        print(*row)

def save_output(output_file, added_cells, score):
    output_file = open(output_file, 'w')
    for row, col, piece_d in added_cells:
        output_file.write(f"{row + 1}{chr(ord('A') + col)} {piece_d}\n")
    output_file.write(str(score))

# Moves files from train dir to input dir based on game_nr 
def copy_train_to_input(train_dir, input_dir, game_nr):
    # Clear input dir
    shutil.rmtree(input_dir)
    os.mkdir(input_dir)

    images_dir = input_dir + "images\\"
    ground_truth_dir = input_dir + "ground_truth\\"
    os.mkdir(images_dir)
    os.mkdir(ground_truth_dir)

    # Copy the images and ground truths to input
    for name in os.listdir(train_dir):
        if f"{game_nr}_" not in name:
            continue
        if "jpg" in name:
            shutil.copy2(train_dir + name, images_dir)
        else:
            shutil.copy2(train_dir + name, ground_truth_dir)

# Returns the added cells, the new board and the score
def run_on_img(last_board: list[list[str]], img: np.ndarray, templates: list[Template], first_image:bool):
    top_left, top_right, bottom_left, bottom_right = get_board_coords(img)

    image_board = extract_board(img, [top_left, top_right, bottom_right, bottom_left])
    # show_image("Board", image_board)

    cells = extract_cells(image_board)

    return classify_cells(cells, last_board, templates, first_image)

# Returns a list of templates from templates_dir
def get_templates(template_dir):
    return [Template(template_dir + name) for name in os.listdir(template_dir)]

# Only called once to apply the same op to the templates as to the pieces on the board
def save_temp(templates: list[Template]):
    for temp in templates:
        cv.imwrite(temp.filename, temp.img)


def run(input_dir):
    output_dir = parent_dir + "\\output\\"
    template_dir = parent_dir + "\\templates\\"
    templates = get_templates(template_dir)
   
    # Prepare output
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    for game_nr in range (1, 6):
        last_board = [[EMPTY_CELL for _ in range(BOARD_SHAPE[0])] for _ in range(BOARD_SHAPE[1])]

        # For each image, play the game
        for i in range(0, 21):
            # Determine filename
            move = str(i)
            if i < 10:
                move = "0" + move
            filename = f"{game_nr}_{move}.jpg"

            # Run on the current image
            print(filename)
            test_img = cv.imread(input_dir + filename)
            new_board, added_cells, score = run_on_img(last_board, test_img, templates, i == 0)

            save_output(output_dir + filename[:filename.find('.')] + '.txt', added_cells, score)
            last_board = new_board

run(parent_parent_parent_dir + "\\testare\\")