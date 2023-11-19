import matplotlib.pyplot as plt
import numpy as np

# Generate traingle wave
def generateInput(time):
    input_data = np.zeros(len(time))

    for i in range(len(time)):
        input_data[i] = np.random.normal(0, 1)

    return input_data

# Generate interferance array
def generateInterferances(size, interferance_range):
    interferance = np.zeros(size)
    for i in range(size):
        interferance[i] = np.random.uniform(-interferance_range / 2, interferance_range / 2)

    return interferance

# Output basen on set parameters and noise
def calculateOutput(input_k, input_k1, input_k2, interferance):
    b0_asterisk, b1_asterisk, b2_asterisk = 1, 1, 1

    return input_k * b0_asterisk + input_k1 * b1_asterisk + input_k2 * b2_asterisk + interferance

# Generate data (input, output) based on set parameters and noise
def dataSimulation(base_funtion):
    simulated_data = np.zeros((len(base_funtion), 2))
    interferance_arr = generateInterferances(len(base_funtion), interferance_range=2)

    for point in range(len(base_funtion)):
        if point < 2:
            simulated_data[point] = [base_funtion[point], 0]
            continue

        simulated_data[point] = [base_funtion[point], calculateOutput(base_funtion[point], base_funtion[point - 1], base_funtion[point - 2], interferance_arr[point])]

    return simulated_data

# access data in column from 2d array
# input_data = np.zeros(len(simulated_data))
# output_data = np.zeros(len(simulated_data))
# input_data = simulated_data[:, 0]
# output_data = simulated_data[:, 1]

def ex1(time):
    input_function = generateInput(time)

    plt.plot(time[2:], dataSimulation(input_function)[2:], ".", markersize=3)
    plt.grid(True); plt.show()

def main():
    plot_range = 10
    numb_points = 100

    time = np.linspace(0, plot_range, num=numb_points)

    # ex1(time)

    print(generateInput(time))
    plt.plot(time, generateInput(time))
    plt.show()

if __name__ == "__main__":
    main()