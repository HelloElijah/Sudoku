import sys
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from helper_funcs import *

# Ex 2.3 Algorithm 2: Douglas-Rachford for Sudoku with binary encoding

# Some important Note
# prob_3d[0] is the first row (9x9 matrix prob)
# prob_3d[:,0] is the first col (9x9 matrix prob)
# prob_3d[0:3,0:3,:].reshape(9,9) is the first block

def Hungarian_Projection(X_2d, W_2d):
    # Create an Z that can satisfied the constraint X(X-Z) = 0, by increasing the value where X!=0
    # Only be used to find the index
    # By adding the value in X, to make sure constraints have a very high priority
    W_2d_constraints = W_2d.copy()
    W_2d_constraints = W_2d_constraints + X_2d * 100

    row_ind, col_ind = linear_sum_assignment(W_2d_constraints, maximize=True)
    Z_2d = np.zeros_like(W_2d, dtype=float)
    for i in range(len(row_ind)):
        Z_2d[row_ind[i], col_ind[i]] = 1.0

    return Z_2d

def Douglas_Rachford_binary_encoding(board, maxiter=100):
    X = init_from_board(board)
    Z = init_from_board(board)

    # Modified X to only contain 0 or 1, if it is not 1, then set to 0
    # Use to hold the constraints X(X-Z) = 0
    X = np.where(X < 1, 0.0, X)

    N = Z.shape[0]
    n = int(math.sqrt(N))
    W = np.zeros_like(Z, dtype=float)

    for t in range(maxiter):
        for i in range(N):
            W[:,:,i] = Hungarian_Projection(X[:,:,i], Z[:,:,i])
        Z = Z - W
        W = W - Z

        # Block constraints
        for i in range(N):
            r = int(i % n + 1)
            c = int(i // n + 1)

            X_block = X[(r-1)*n:r*n, (c-1)*n:c*n].reshape(N,N)
            W_block = W[(r-1)*n:r*n, (c-1)*n:c*n].reshape(N,N)
            W[(r-1)*n:r*n, (c-1)*n:c*n] = Hungarian_Projection(X_block, W_block).reshape(n,n,N)

        Z = Z + W

    for i in range(N):
        Z[:,:,i] = Hungarian_Projection(X[:,:,i], Z[:,:,i])
        
    return convert_to_board(Z)

def DebugMode(start_boards, answer):
    print('---------------------------------------------------------------------')
    print("DebugMode")
    for i in range(len(answer)):
        print('---------------------------------------------------------------------')
        print('Sudoku number:', i)
        print_sudoku(start_boards[i])
        print_sudoku(answer[i])
        print(check_board(answer[i]))
        IsConstraints(start_boards[i], answer[i])
        print('---------------------------------------------------------------------')

def main(argv):
    np.random.seed(0)
    start_boards = load_sudokus(argv[0])
    my_solution = []
    for start_board in start_boards:
        answer = Douglas_Rachford_binary_encoding(start_board, maxiter=200)
        my_solution.append(answer)

    if (len(argv) > 1):
        ground_truth_solutions = load_sudokus(argv[1])
        compare_sudokus(my_solution,ground_truth_solutions)
    else:
        DebugMode(start_boards, my_solution)

if __name__ == "__main__":
   main(sys.argv[1:])