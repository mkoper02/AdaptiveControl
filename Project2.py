import matplotlib.pyplot as plt
import numpy as np

# Generate random samples from a normal distribution
def generateInput(size):
    return np.random.normal(0, 0.5, size)

# Output based on set parameters and noise
def calculateOutput(input_k, input_k1, input_k2, interferance):
    b0_asterisk, b1_asterisk, b2_asterisk = 1, 1, 1

    return input_k * b0_asterisk + input_k1 * b1_asterisk + input_k2 * b2_asterisk + interferance

# Generate data (input, output) based on set parameters and noise
def dataSimulation(time):
    input = generateInput(len(time))

    simulated_data = np.zeros((len(input), 2))
    interferance_range = 2

    for point in range(len(input)):
        if point < 2:
            simulated_data[point] = [input[point], 0]
            continue

        simulated_data[point] = [input[point], calculateOutput(input[point], input[point - 1], input[point - 2], np.random.uniform(-interferance_range / 2, interferance_range / 2))]

    return simulated_data

def createXnMatrix(input_data):
    row_size = 3

    x_n = np.zeros((len(input_data) - 2, row_size))  

    for row in range(len(x_n)):
        for i in range(row_size): x_n[row][i] = input_data[row + i]

    return x_n

def calculateAn(past_an, Pn, temp_xn, yn):
    xn = np.array([temp_xn])

    return np.array([past_an[:, 0] + (np.matmul(Pn, xn[0]) * (yn - (np.matmul(xn.T[:, 0],  past_an[:, 0]))))]).T

def calculatePn(past_Pn, temp_xn, _lambda):
    xn = np.array([temp_xn])

    # return past_Pn - (((np.matmul(np.matmul(past_Pn, xn[0]), xn.T) * past_Pn) / (_lambda + np.matmul(np.matmul(xn.T[:, 0], past_Pn), xn[0]))))
    return (1 / _lambda) * (past_Pn - (((np.matmul(np.matmul(past_Pn, xn[0]), xn.T) * past_Pn) / (_lambda + np.matmul(np.matmul(xn.T[:, 0], past_Pn), xn[0])))))

def ex1(time):
    simulated_data = dataSimulation(time)

    Xn = createXnMatrix(simulated_data[:, 0]).T
    Yn = simulated_data[2:, 1]
    an = np.array([np.zeros(3)]).T
    Pn = np.identity(3) * 10**5

    an_arr = np.empty((Xn.shape[1], 3))

    for point in range(Xn.shape[1]):
    # for point in range(4):
        Pn = calculatePn(Pn, Xn[:, point], _lambda=1)
        an = calculateAn(an, Pn, Xn[:, point], Yn[point])
        an_arr[point] = an[:, 0]

    # plt.plot(time[2:], simulated_data[2:, 0])
    # plt.plot(time[2:], an_arr[:, 0])
    plt.plot(time[2:], an_arr)
    plt.grid(); plt.show()

def main():
    plot_range = 200    
    numb_points = 5000

    time = np.linspace(0, plot_range, num=numb_points)

    ex1(time)

if __name__ == "__main__":
    main()