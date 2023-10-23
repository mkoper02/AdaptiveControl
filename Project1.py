import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def denoising(noise_points, horizon):
    denoised_wave = np.zeros(len(noise_points))

    for segment in range(len(noise_points)):
        sum = 0

        for point in range(horizon):
            if segment < horizon - 1: break
            sum += noise_points[segment - point]
        
        denoised_wave[segment] = sum / horizon
        
    return denoised_wave

# Generate traingle wave
def generateTriangleWave(time):
    amp = 1           # amplitude
    freq = 0.2        # frequency

    return amp * signal.sawtooth(np.pi * freq * time, width=0.5)

# Generate wave with noise
def generateNoiseWave(base_wave, rand_range): 
    noise_wave = np.copy(base_wave)

    for i in range(len(base_wave)):
        noise_wave[i] += np.random.uniform(-rand_range / 2, rand_range / 2)

    return noise_wave

def MSE(traingle_wave, denoised_points, horizon):
    mse = 0

    for point in range(len(traingle_wave) - horizon):
        mse += pow(denoised_points[point] - traingle_wave[point], 2)

    return mse / (len(traingle_wave) - horizon)

def simulateHorizons(horizon_range, triangle_wave, generated_noise_points):
    MSE_values = np.zeros([horizon_range])

    for horizon in range(1, horizon_range + 1): 
        MSE_values[horizon - 1] = MSE(triangle_wave, denoising(generated_noise_points, horizon), horizon)

    return MSE_values

def allWavesPlot(triangle_wave, generated_noise_points, denoised_wave, horizon, time):
    plt.figure(figsize=(12, 7))

    # Create plots with scope lowered by the horizon
    plt.plot(time, triangle_wave, label="Function without noise", color="r") 
    plt.plot(time, generated_noise_points, ".", markersize=5, label="Points with noise")
    plt.plot(time[horizon - 1:], denoised_wave[horizon - 1:], ".-", markersize=5, label="Denoised function")

    # Plot properties
    plt.grid(); plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16))

# Create plot with triangle wave and generated noise points
def noiseTrianglePlot(base_wave, noise_points, time):
    # Plot size
    plt.figure(figsize=(12, 6))

    # Create plot
    plt.plot(time, base_wave, label="Function without noise", color="r") 
    plt.plot(time, noise_points, ".", markersize=5, label="Points with noise")

    # Plot properties
    plt.grid(); plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16))

def noiseDenoisedPlot(denoised_wave, noise_points, time, horizon):
    # Plot size
    plt.figure(figsize=(12, 6))
    
    # Create plot with scope lowered by the horizon
    plt.plot(time, noise_points, ".", markersize=5, label="Points with noise")
    plt.plot(time[horizon - 1:], denoised_wave[horizon - 1:], ".-", markersize=5, label="Denoised function")

    # Plot properties
    plt.grid(); plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16))

# MSE dependence on horizons
def ex1(time, horizon_max_range):
    triangle_wave = generateTriangleWave(time)
    noise_wave = generateNoiseWave(triangle_wave, rand_range=1)

    calculated_harizons_mse = simulateHorizons(horizon_max_range, triangle_wave, noise_wave)

    # MSE to horizon values on plot
    plt.scatter(np.arange(0, horizon_max_range, 1), calculated_harizons_mse, marker=".")
    # Highlight min value
    plt.plot(np.argmin(calculated_harizons_mse), min(calculated_harizons_mse), marker='.', color='r')
    plt.grid(); plt.xlabel("Horizon"); plt.ylabel("MSE")

    noiseTrianglePlot(triangle_wave, noise_wave, time)
    noiseDenoisedPlot(denoising(noise_wave, np.argmin(calculated_harizons_mse) + 1), noise_wave, time, np.argmin(calculated_harizons_mse) + 1)

# MSE dependence on interference variance
# def ex2():


# Optimal horizon dependence on interference variance
# def ex3():


def main():
    # horizon = 3
    plot_range = 100
    rand_range = 1
    horizon_max = 50
    numb_points = 5000

    time = np.linspace(0, plot_range, num=numb_points)

    ex1(time, horizon_max)
    # ex2()
    # ex3()

    plt.show()

if __name__ == "__main__":
    main()