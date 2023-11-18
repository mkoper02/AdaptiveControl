import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Generate traingle wave
def generateTriangleWave(time):
    amp = 1           # amplitude
    freq = 0.5        # frequency

    return amp * signal.sawtooth(freq * time, width=0.5)

def calculateOutput(input_k, input_k1, input_k2):
    b0_astrid = 1
    b1_astrid = 1
    b2_astrid = 1
    interferances_range = 1

    return input_k * b0_astrid + input_k1 * b1_astrid + input_k2 * b2_astrid + np.random.uniform(-interferances_range / 2, interferances_range / 2)

# Generate data (input, output) based on set parameters and noise
def dataSimulation(base_funtion):
    simulated_data =np.zeros((len(base_funtion), 2))

    for point in range(len(base_funtion)):
        if point < 2:
            simulated_data[point] = [base_funtion[point], 0]
            continue

        simulated_data[point] = [base_funtion[point], calculateOutput(base_funtion[point], base_funtion[point - 1], base_funtion[point - 2])]

    return simulated_data

def main():
    time = np.linspace(0, 100, num=5000)

    triangle_wave = generateTriangleWave(time)

    plt.plot(time[2:], dataSimulation(triangle_wave)[2:], ".", markersize=3)
    plt.grid(True); plt.show()

if __name__ == "__main__":
    main()