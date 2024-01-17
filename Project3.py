import matplotlib.pyplot as plt
import numpy as np
import math
# A parameter for blocks
A_arr = np.array([
    [0.5, 0], 
    [0, 0.25]
])

# B parameter for blocks
B_arr = np.array([
    [1, 0], 
    [0, 1]
])

# connection matrix
H_arr = np.array([
    [0, 1], 
    [1, 0]
])

RADIUS = 1


# generate noise Z - data format:  [[z1, z2]]
def generateNoise(size : int, noise_range : float) -> np.ndarray:
    rng = np.random.default_rng(50)
    single_noise =  rng.uniform(-noise_range / 2, noise_range / 2, size * 2)
    noise_arr = np.array([[single_noise[i], single_noise[i + 1]] for i in range(0, len(single_noise) - 1, 2)])

    return noise_arr


# generate input U - data format: [[u1, u2]]
def generateInput(size : int) -> np.ndarray:
    rng = np.random.default_rng(50)
    single_input =  rng.normal(0, 1.5, size * 2)
    input_arr = np.array([[single_input[i], single_input[i + 1]] for i in range(0, len(single_input) - 1, 2)])

    return input_arr


def calculateOutput(input : np.ndarray, noise : np.ndarray) -> np.ndarray:
    input_vector = np.array([input]).T
    noise_vector = np.array([noise]).T

    # (I - AH)^-1
    bracket = np.linalg.inv(np.identity(2) - A_arr @ H_arr)

    # ((I - AH)^-1) * Bu + ((I - AH)^-1) * z
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


def parametersIdentification(output : np.ndarray) -> None:
    # get U and Y for first and second block - data format: [[u, y]]
    first_block_UY = np.array([[output[i][0][0], output[i][1][0]] for i in range(len(output))])
    second_block_UY = np.array([[output[i][0][1], output[i][1][1]] for i in range(len(output))])

    # get X for first and second block
    # x = H * y
    first_block_X = np.array([(H_arr @ output[i][1])[0] for i in range(len(output))])
    second_block_X = np.array([(H_arr @ output[i][1])[1] for i in range(len(output))])

    b1, a1 = calculateParameters(first_block_UY[:, 0], first_block_UY[:, 1], first_block_X)
    print(f"Parametry dla pierwszego bloku: a = {a1}, b = {b1}")

    b2, a2 = calculateParameters(second_block_UY[:, 0], second_block_UY[:, 1], second_block_X)
    print(f"Parametry dla drugiego bloku: a = {a2}, b = {b2}\n")


def plotCirclePoint(u1: float, u2 : float) -> None:
    point = (u1, u2)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    circle = plt.Circle((0, 0), RADIUS, edgecolor='b', fill=False, label="Ograniczenia")
    ax.add_patch(circle)

    ax.plot(*point, 'ro', label='Wartość optymalna')
    ax.set_aspect('equal')

    # draw the axis lines
    ax.axvline(0, color="black", alpha=0.3)
    ax.axhline(0, color="black", alpha=0.3)

    plt.legend(); plt.xlabel("$u_{1}$"); plt.ylabel("$u_{2}$"); plt.grid(True)
    plt.show()


def costFunction(U : np.ndarray, wanted : np.ndarray) -> float:
    y1, y2 = calculateOutput(U, [0, 0]).T[0]
    return (y1 - wanted[0]) ** 2 + (y2 - wanted[1]) ** 2

def calculateOptimalU1(wanted):
    l = -1
    p = 1
    epsilon = (abs(l) + abs(p)) / 100
    for i in range(10):
        u1 = (l + p) / 2
        leftLimit = u1 - epsilon
        rightLimit = u1 + epsilon
        leftLimitU2 = math.sqrt(1 - pow(leftLimit, 2))
        rightLimitU2 = math.sqrt(1 - pow(rightLimit, 2))
        leftU = [leftLimit, calculateOptimalU2(-leftLimitU2, leftLimitU2, leftLimit, wanted)]
        rightU = [rightLimit, calculateOptimalU2(-rightLimitU2, rightLimitU2, rightLimit, wanted)]
        leftY = np.linalg.inv(np.identity(2) - (A_arr @ H_arr)) @ B_arr @ leftU
        rightY = np.linalg.inv(np.identity(2) - (A_arr @ H_arr)) @ B_arr @ rightU
        leftQ = pow((leftY[0] - wanted[0]), 2) + pow((leftY[1] - wanted[1]), 2)
        rightQ = pow((rightY[0] - wanted[0]), 2) + pow((rightY[1] - wanted[1]), 2)
        if leftQ > rightQ:
            l = leftLimit
        else:
            p = rightLimit
        epsilon = epsilon / 2
    u1 = (l + p) / 2
    return u1

def calculateOptimalU2(L, P, U1, wanted):
    l = L
    p = P
    epsilon = (abs(l) + abs(p)) / 100
    for i in range(10):
        u2 = (l + p) / 2
        leftLimit = u2 - epsilon
        rightLimit = u2 + epsilon
        leftU = [U1, leftLimit]
        rightU = [U1, rightLimit]
        leftY = np.linalg.inv(np.identity(2) - (A_arr @ H_arr)) @ B_arr @ leftU
        rightY = np.linalg.inv(np.identity(2) - (A_arr @ H_arr)) @ B_arr @ rightU
        leftQ = pow((leftY[0] - wanted[0]), 2) + pow((leftY[1] - wanted[1]), 2)
        rightQ = pow((rightY[0] - wanted[0]), 2) + pow((rightY[1] - wanted[1]), 2)
        if leftQ > rightQ:
            l = leftLimit
        else:
            p = rightLimit
        epsilon = epsilon / 2
    u2 = (l + p) / 2
    return u2

def optimizeU(wanted : np.ndarray) -> None:
    # u_min = []
    # u1=0
    # optimal_u1 = calculateOptimalU1(u1,wanted)
    # optimal_u2 = calculateOptimalU2(optimal_u1,wanted)
    # cost = costFunction([optimal_u1,optimal_u2], wanted)
    # print(f"Optymalne u1: {optimal_u1}")
    # print(f"Optymalne u2: {optimal_u2}")
    # print(f"Minimalna wartość funkcji kosztu: {cost}")

    optU1 = calculateOptimalU1(wanted)
    l = math.sqrt(1 - pow(optU1, 2))
    optU2 = calculateOptimalU2(-l, l, optU1, wanted)

    print("Optymalna wartosc U1: ", optU1, "\nOptymalna wartosc U2: ", optU2)
    optU = [optU1, optU2]
    Y = np.linalg.inv(np.identity(2) - (A_arr @ H_arr)) @ B_arr @ optU
    quality = pow((Y[0] - 4), 2) + pow((Y[1] - 4), 2)
    print("Kryterium jakosci: ", quality)

    plotCirclePoint(optU1, optU2)




def main() -> None:
    size = 1000
    noise_range = 0.1
    wanted = np.array([4, 4])

    # output = dataSimulation(size, noise_range)
    # parametersIdentification(output)
    optimizeU(wanted)


def main() -> None:
    size = 1000
    noise_range = 0.1
    wanted = np.array([4, 4])

    # output = dataSimulation(size, noise_range)
    # parametersIdentification(output)
    optimizeU(wanted)


if __name__ == '__main__':
    main()