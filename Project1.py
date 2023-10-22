import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def denoising(noise_points, horizon):
    denoised = np.zeros(len(noise_points))

    for segment in range(len(noise_points)):
        sum = 0

        # Use values "from the future"
        # for point in range(horizon):
        #     if horizon + segment > len(noise_points): break
        #     sum += noise_points[segment + point]
        
        # Use values "from the past"
        for point in range(horizon):
            if segment < horizon - 1: break
            sum += noise_points[segment - point]
        
        denoised[segment] = sum / horizon
        
    return denoised

def addNoise(triangle_wave, rand_range): 
    noise_points = np.copy(triangle_wave)

    for i in range(len(triangle_wave)):
        noise_points[i] += np.random.uniform(0, rand_range) - (rand_range / 2)

    return noise_points

def generateWaves(time, rand_range):
    amp = 1         # amplitude
    freq = 0.2        # frequency

    # Generate triangle wave, points with noise
    triangle_wave = amp * signal.sawtooth(np.pi * freq * time, width=0.5)
    generated_noise_points = addNoise(triangle_wave, rand_range)

    return triangle_wave, generated_noise_points

def MSE(traingle_wave, denoised_points, horizon):
    mse = 0

    for point in range(len(traingle_wave) - horizon):
        mse += pow(denoised_points[point] - traingle_wave[point], 2)

    return mse / (len(traingle_wave) - horizon)

def simulateHorizons(horizon_range, triangle_wave, generated_noise_points):
    MSE_values = np.zeros([horizon_range])

    for horizon in range(1, horizon_range + 1): 
        MSE_values[horizon - 1] = MSE(triangle_wave, denoising(generated_noise_points, horizon), horizon)

    # MSE values on plot
    plt.scatter(np.arange(0, horizon_range, 1), MSE_values, marker=".")
    # Highlight min value
    plt.plot(np.argmin(MSE_values), min(MSE_values), marker='.', color='r')
    plt.grid(); plt.xlabel("Horizon"); plt.ylabel("MSE")

    print(np.argmin(MSE_values) + 1)
    return np.argmin(MSE_values) + 1

def createWavePlots(triangle_wave, generated_noise_points, denoised_wave, horizon, numb_points, time):
    plt.figure(figsize=(12, 7))

    # Create plots with scope lowered by the horizon
    plt.plot(time, triangle_wave, label="Function without noise", color="r") 
    plt.plot(time, generated_noise_points, ".", markersize=5, label="Points with noise")
    plt.plot(time[horizon - 1:], denoised_wave[horizon - 1:], ".-", markersize=5, label="Denoised function")

    # Plot properties
    plt.grid(); plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16))

# MSE dependence on horizons
def ex1(plot_range, horizon_simulate, numb_points):
    time = np.linspace(0, plot_range, num=numb_points)
    triangle_wave, generated_noise_points = generateWaves(time, rand_range=1)

    best_horizon = simulateHorizons(horizon_simulate, triangle_wave, generated_noise_points)
    createWavePlots(triangle_wave, generated_noise_points, denoising(generated_noise_points, best_horizon), best_horizon, numb_points, time)

# MSE dependence on interference variance
# def ex2():


# Optimal horizon dependence on interference variance
# def ex3():


def main():
    # horizon = 3
    plot_range = 50
    rand_range = 1
    horizon_simulate = 50
    numb_points = 5000

    ex1(plot_range, horizon_simulate, numb_points)

    plt.show()

if __name__ == "__main__":
    main()