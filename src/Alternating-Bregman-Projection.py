import sys
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from helper_funcs import *
from alg3_optional import *

# Ex 3.3 Algorithm 5: Alternating Bregman projection for Sudoku with fractional encoding
# Did not 100% follow the psudocode
# Change Z = argmax(Z,3) to Alternating_Projection_binary_encoding(Z)(Algorithm 3)
# Better performace than alg5.py

# Some important Note
# prob_3d[0] is the first row (9x9 matrix prob)
# prob_3d[:,0] is the first col (9x9 matrix prob)
# prob_3d[0:3,0:3,:].reshape(9,9) is the first block

def Alternating_Projection(W, maxiter=100):
    N = W.shape[0]
    for t in range(maxiter):
        for k in range(N):
            W[k] = W[k] / np.sum(W[k])
        for k in range(N):
            W[:,k] = W[:,k] / np.sum(W[:,k])

    # We do need return most case, only useful in block constraints
    # Since reshape in block constraints use copy()
    return W

def Alternating_Bregman_Projection_fractional_encoding(board, maxiter=100):
    X = init_from_board(board)
    N = board.shape[0]
    n = int(math.sqrt(N))

    # initialize Z ∈ [0,1]N×N×N to preserve given values in X
    Z = init_from_board(board)

    for t in range(maxiter):
        # Row constraints
        for i in range(N):
            Alternating_Projection(Z[i], maxiter)

        # Col constraints
        for i in range(N):
            Alternating_Projection(Z[:,i], maxiter)

        # Block constraints
        for i in range(N):
            r = int(i % n + 1)
            c = int(i // n + 1)
            Z_block = Z[(r-1)*n:r*n, (c-1)*n:c*n, :].reshape(N,N)
            Z[(r-1)*n:r*n, (c-1)*n:c*n, :] = Alternating_Projection(Z_block, maxiter).reshape(n,n,N)


    for t in range(maxiter):
        # Row constraints
        for i in range(N):
            Z[i] = Hungarian_Projection(X[i], Z[i])

        # Col constraints
        for i in range(N):
            Z[:,i] = Hungarian_Projection(X[:,i], Z[:,i])

        # Block constraints
        for i in range(N):
            r = int(i % n + 1)
            c = int(i // n + 1)

            X_block = X[(r-1)*n:r*n, (c-1)*n:c*n].reshape(N,N)
            Z_block = Z[(r-1)*n:r*n, (c-1)*n:c*n].reshape(N,N)
            Z[(r-1)*n:r*n, (c-1)*n:c*n] = Hungarian_Projection(X_block, Z_block).reshape(n,n,N)
        
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
        answer = Alternating_Bregman_Projection_fractional_encoding(start_board, maxiter=100)
        my_solution.append(answer)

    if (len(argv) > 1):
        ground_truth_solutions = load_sudokus(argv[1])
        compare_sudokus(my_solution,ground_truth_solutions)
    else:
        DebugMode(start_boards, my_solution)

if __name__ == "__main__":
   main(sys.argv[1:])
