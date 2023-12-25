import matplotlib.pyplot as plt
import numpy as np


# A parameter for blocks
A_arr = np.array( [[0.5, 0], [0, 0.25]] )

# B parameter for blocks
B_arr = np.array( [[1, 0], [0, 1]] )

# connection matrix
H_arr = np.array( [[0, 1], [1, 0]] )


# generate noise Z
def generateNoise(size : int, noise_range : float) -> np.ndarray:
    rng = np.random.default_rng(50)
    single_noise =  rng.uniform(-noise_range / 2, noise_range / 2, size * 2)
    noise_arr = np.array([[single_noise[i], single_noise[i + 1]] for i in range(0, len(single_noise) - 1, 2)])

    return noise_arr


# generate input U
def generateInput(size : int) -> np.ndarray:
    rng = np.random.default_rng(50)
    single_input =  rng.normal(0, 1.5, size * 2)
    input_arr = np.array([[single_input[i], single_input[i + 1]] for i in range(0, len(single_input) - 1, 2)])

    return input_arr


def calculateOutput(input : np.ndarray, noise : np.ndarray) -> np.ndarray:
    input_vector = np.array([input]).T
    noise_vector = np.array([noise]).T

    # (I + AH)
    bracket = np.linalg.inv(np.identity(2) - A_arr @ H_arr)

    # (I + AH)Bu + (I + AH)z
    return (bracket @ B_arr @ input_vector + bracket @ noise_vector)


# return array with input and output data: (U_arr, Y_arr)
def dataSimulation(size : int, noise_range : float) -> np.ndarray:
    U_arr = generateInput(size)
    noise_arr = generateNoise(size, noise_range)

    # Y_arr = [[[u1, u2],
    #           [y1, y2]]]
    Y_arr = np.empty((size, 2, 2))
    
    for i in range(len(Y_arr)):
        Y_arr[i][0] = U_arr[i]
        Y_arr[i][1] = calculateOutput(U_arr[i], noise_arr[i]).T[0] # extract data
        
    return Y_arr


# calculate A and B parameter for given block
def calculateParameters(U : np.ndarray, Y : np.ndarray, X : np.ndarray) -> np.ndarray:
    Wi = np.array([U, X])
    return Y @ Wi.T @ np.linalg.inv(Wi @ Wi.T)


def main() -> None:
    size = 5
    noise_range = 0.1

    output = dataSimulation(size, noise_range)

    # get U and Y for first and second block - data format: [[u, y]]
    first_block_UY = np.array([[output[i][0][0], output[i][1][0]] for i in range(len(output))])
    second_block_UY = np.array([[output[i][0][1], output[i][1][1]] for i in range(len(output))])

    # get X for first and second block
    first_block_X = np.array([(H_arr @ output[i][1])[0] for i in range(len(output))])
    second_block_X = np.array([(H_arr @ output[i][1])[1] for i in range(len(output))])

    print(calculateParameters(first_block_UY[:, 0], first_block_UY[:, 1], first_block_X))
    print(calculateParameters(second_block_UY[:, 0], second_block_UY[:, 1], second_block_X))


if __name__ == '__main__':
    main()