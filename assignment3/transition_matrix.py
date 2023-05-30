import numpy as np
import matplotlib.pyplot as plt

# Probabilities to choose
probability_same_room = 0.6 # 60% chance to stay in the same room
probability_next_room = 0.3 # 30% chance to go to an adjacent room
probability_next_next_room = 0.1 # 10% chance to go to an adjacent room of an adjacent room

matrix = np.identity(41)
matrix *= probability_same_room


# 1 = a connected room
matrix[1, 0] = 1
matrix[2, 1] = 1
matrix[3, 1] = 1
matrix[3, 2] = 1
matrix[4, 1] = 1
matrix[4, 3] = 1
matrix[6, 3] = 1
matrix[6, 4] = 1
matrix[6, 5] = 1
matrix[7, 3] = 1
matrix[7, 6] = 1
matrix[8, 5] = 1
matrix[9, 8] = 1
matrix[10, 9] = 1
matrix[11, 10] = 1
matrix[12, 7] = 1
matrix[13, 12] = 1
matrix[14, 13] = 1
matrix[15, 14] = 1
matrix[15, 13] = 1
matrix[15, 12] = 1
matrix[16, 12] = 1
matrix[16, 15] = 1
matrix[17, 16] = 1
matrix[18, 17] = 1
matrix[18, 16] = 1
matrix[18, 15] = 1
matrix[17, 15] = 1
matrix[18, 11] = 1
matrix[20, 19] = 1
matrix[21, 20] = 1
matrix[22, 20] = 1
matrix[23, 20] = 1
matrix[23, 22] = 1
matrix[25, 22] = 1
matrix[25, 23] = 1
matrix[25, 24] = 1
matrix[26, 22] = 1
matrix[26, 25] = 1
matrix[27, 25] = 1
matrix[27, 24] = 1
matrix[28, 27] = 1
matrix[29, 28] = 1
matrix[30, 29] = 1
matrix[30, 18] = 1
matrix[30, 11] = 1
matrix[31, 26] = 1
matrix[32, 31] = 1
matrix[33, 32] = 1
matrix[34, 33] = 1
matrix[34, 32] = 1
matrix[34, 31] = 1
matrix[35, 31] = 1
matrix[35, 34] = 1
matrix[36, 35] = 1
matrix[36, 34] = 1
matrix[37, 36] = 1
matrix[37, 35] = 1
matrix[37, 34] = 1
matrix[37, 30] = 1
matrix[37, 18] = 1
matrix[37, 11] = 1
matrix[39, 38] = 1
matrix[40, 39] = 1
matrix[40, 37] = 1
matrix[40, 30] = 1
matrix[40, 27] = 1
matrix[39, 24] = 1
matrix[39, 23] = 1
matrix[39, 19] = 1
matrix[40, 18] = 1
matrix[40, 11] = 1
matrix[40, 8] = 1
matrix[39, 5] = 1
matrix[39, 4] = 1
matrix[39, 0] = 1
matrixT = np.transpose(matrix)
matrix_final = matrix + matrixT - (np.identity(41) * probability_same_room)


# Buren van buren krijgen een score van 2

for i in range(len(matrix_final)):
    indices = np.where(matrix_final[i] == 1)
    for index in indices[0]:
        indices_neighbours = np.where(matrix_final[index] == 1)
        for index_neighbour in indices_neighbours[0]:
            if matrix_final[i, index_neighbour] == 0:
                matrix_final[i, index_neighbour] = 2


for i in range(len(matrix_final)):
    count = np.count_nonzero(matrix_final[i] == 1)
    matrix_final[i][matrix_final[i] == 1] = probability_next_room / count
    count = np.count_nonzero(matrix_final[i] == 2)
    matrix_final[i][matrix_final[i] == 2] = probability_next_next_room / count

transition_matrix = matrix_final