import numpy as np
import math
from scipy.optimize import linear_sum_assignment

def print_sudoku(board_2d_int_arr):
    X = board_2d_int_arr
    r,c = X.shape
    a = int(math.sqrt(r))
    assert r == c
    assert type(X) == np.ndarray
    assert a**2 == r 
    assert isinstance(X[0][0].item(),int)
    
    # Convert array elements to strings
    board_str = X.astype(str)
    
    # Our row separator
    row_sep = '-'*25

    # Loop through 9 rows
    for i in range(r):
        
        # At each multiple of 3, print row separator
        if i % a == 0:
            print(row_sep)

        # Get row data
        row = board_str[i]

        # Format row of data with pipe separators at each end, and between each sub grid
        print('| '+' '.join(row[0:a])+' | '+' '.join(row[a:2*a])+' | '+' '.join(row[2*a:])+' |')

    # Print final row separator at bottom after loops finish
    print(row_sep)

def block_to_coordinate(id,k):
    # Takes in the block number id, and the kth element in that block,
    # Output the coordinate of that cell.
    assert(isinstance(id,int))
    assert(isinstance(k,int))

    a = id // 3
    b = id % 3

    i = 3*a + (k//3)
    j = 3*b + (k%3)

    return (i,j)

def check_board(input_board):
    """Check if the current board is a valid finished sudoku"""
    X = input_board
    r,c = X.shape
    n = r
    a = int(math.sqrt(r))
    assert r == c
    assert type(X) == np.ndarray
    assert a**2 == r
    assert isinstance(X[0][0].item(),int)


    flag = True
    for row in range(r):
        if len(np.unique(X[row])) != len(X[row]):
            flag = False
            print("row is wrong:",row)

    for col in range(c):
        if len(np.unique(X[:,col])) != len(X[:,col]):
            flag = False
            print("col is wrong:",col)
    
    for block_id in range(n):
        temp_arr = np.zeros(n,dtype=int)
        for k in range(n):
            r_idx, c_idx = block_to_coordinate(block_id,k)
            temp_arr[k] = X[r_idx][c_idx]
        if len(np.unique(temp_arr)) != len(temp_arr):
            flag = False
            print("block is wrong:",block_id)

    return flag

def to_onehot(int_x,n=9):
    """ Convert integer x between 1 and 9 to its one-hot probablity vector encoding. """

    assert 1 <= int_x and int_x <= n
    onehot_vec = np.zeros(n)
    onehot_vec[int_x-1] = 1
    return onehot_vec

def convert_to_3d_repr(nonzero_board):
    """This function converts board with all NONZERO entries to 3d probability representation. """

    r,c = nonzero_board.shape
    n = r
    a = int(math.sqrt(r))
    assert r == c
    assert type(nonzero_board) == np.ndarray
    assert a**2 == r 

    prob_3d_arr = np.zeros(shape=(r,c,n))

    for i in range(r):
        for j in range(c):
            prob_3d_arr[i][j] = to_onehot(nonzero_board[i][j])

    return prob_3d_arr

def convert_to_board(prob_3d_arr):
    """This function converts a 3d probability vector representation back to the normal 2d board representation"""
    
    X = prob_3d_arr
    assert type(X) == np.ndarray
    r,c,n = X.shape
    assert r == c and c == n

    board = np.zeros(shape=(r,c), dtype=int)

    for i in range(r):
        for j in range(c):
            prob_vec = X[i][j]
            num = np.argmax(prob_vec)+1
            board[i][j] = num

    return board

def init_from_board(start_board):
    """Initialize probability 3d representation from a given board (2d).
    Given 'clue' cells are converted to one-hot encoding.
    Empty 'unkown' cells are converted to a uniformly random encoding over possible values, for example,
    an empty cell that can take values in {1,4,5,8} is converted to [0.25,0,0,0.25,0.25,0,0,0.25,0]
    
    """

    X = start_board
    r,c = X.shape
    n = r
    a = int(math.sqrt(r))
    assert r == c
    assert type(X) == np.ndarray
    assert a**2 == r 

    temp_3d_arr = np.ones(shape=(r,c,n))
    prob_3d_arr = np.ones(shape=(r,c,n))

    # First pass converts the nonzero entries to one-hot encodings, and
    # Set the correpsonding entries of temp_arr in same row/column/block to 0
    for i in range(r):
        for j in range(c):
            val = X[i][j]
            if val != 0:
                prob_3d_arr[i][j] = to_onehot(val)
                for k in range(n):
                    assert(isinstance(k,int))
                    temp_3d_arr[i][k][val-1] = 0  # set the entry in same row to have zero corresponding element
                    temp_3d_arr[k][j][val-1] = 0  # set the entry in same column to have zero corresponding element

                    # for same box
                    a = i // 3
                    b = j // 3
                    row_start_idx = 3*a
                    col_start_idx = 3*b

                    temp_3d_arr[row_start_idx+(k//3)][col_start_idx+(k%3)][val-1] = 0

    # Second pass normalizes the zero entries' corresponding vector to probability vector and assign it to prob_3d_arr
    for i in range(r):
        for j in range(c):
            val = X[i][j]
            if val == 0:
                prob_3d_arr[i][j] = temp_3d_arr[i][j]/np.sum(temp_3d_arr[i][j])

    return prob_3d_arr

def load_sudokus(filename):
    # Read list of sudokus from local file,
    # Filenames can be e.g. 'start_boards.npy', 'ground_truth_solutions.npy'
    list_sudokus = np.load(filename,allow_pickle=True)

    return list_sudokus

def save_sudokus(list_sudokus,filename):
    # Optional helper function, saves list of sudokus to file ('.npy').
    np.save(filename, list_sudokus, allow_pickle=True)
    return

def compare_sudokus(alg_output_sudokus,ground_truth_sudokus):
    """
    This function compares the list of final boards solved by your implementation with the list of ground truth solutions
    """
    n = len(alg_output_sudokus)
    m = len(ground_truth_sudokus)
    assert n == m

    for i in range(n):
        print("Sudoku number:",i)
        if np.array_equal(alg_output_sudokus[i],ground_truth_sudokus[i]):
            print("Correct!")
        else:
            print("Wrong")

def constraints_helper(X, Z):
    N, _ = X.shape
    for i in range(N):
        for j in range(N):
            if X[i][j] != 0 and X[i][j] != Z[i][j]:
                return 0
    
    return 1

def IsConstraints(X, Z):
    result = constraints_helper(X, Z)
    if result == 0:
        print("Not satisfied the constraints")
    else:
        print("Satisfied the constraints")