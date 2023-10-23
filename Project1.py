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
    freq = 0.5        # frequency

    return amp * signal.sawtooth(freq * time, width=0.5)

# Generate wave with noise
def generateNoiseWave(base_wave, interferances_range): 
    noise_wave = np.copy(base_wave)

    for i in range(len(base_wave)):
        noise_wave[i] += np.random.uniform(-interferances_range / 2, interferances_range / 2)

    return noise_wave

def MSE(traingle_wave, denoised_points, horizon):
    mse = 0

    for point in range(len(traingle_wave) - horizon):
        mse += pow(denoised_points[point] - traingle_wave[point], 2)

    return mse / (len(traingle_wave) - horizon)

def simulateHorizons(triangle_wave, generated_noise_points, horizon_range):
    MSE_values = np.zeros([horizon_range])

    for horizon in range(1, horizon_range + 1): 
        MSE_values[horizon - 1] = MSE(triangle_wave, denoising(generated_noise_points, horizon), horizon)

    return MSE_values

def allWavesPlot(time, horizon):
    triangle_wave = generateTriangleWave(time)
    noise_wave = generateNoiseWave(triangle_wave, interferances_range=1)

    noiseTrianglePlot(triangle_wave, noise_wave, time)
    noiseDenoisedPlot(denoising(noise_wave, horizon), noise_wave, time, horizon)

# Create plot with triangle wave and generated noise points
def noiseTrianglePlot(base_wave, noise_points, time):
    # Plot size
    plt.figure(figsize=(12, 6))

    # Create plot
    plt.plot(time, base_wave, label="Function without noise", color="r") 
    plt.plot(time, noise_points, ".", markersize=5, label="Points with noise")

    # Plot properties
    plt.grid(True); plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16))

def noiseDenoisedPlot(denoised_wave, noise_points, time, horizon):
    # Plot size
    plt.figure(figsize=(12, 6))
    
    # Create plot with scope lowered by the horizon
    plt.plot(time, noise_points, ".", markersize=5, label="Points with noise")
    plt.plot(time[horizon - 1:], denoised_wave[horizon - 1:], ".-", markersize=5, label="Denoised function")

    # Plot properties
    plt.grid(True); plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16))

# MSE dependence on horizons
def ex1(time, horizon_range):
    triangle_wave = generateTriangleWave(time)
    noise_wave = generateNoiseWave(triangle_wave, interferances_range=1)

    calculated_harizons_mse = simulateHorizons(triangle_wave, noise_wave, horizon_range)
    best_horizon = np.argmin(calculated_harizons_mse)

    # MSE to horizon values plot and highlight min value
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, horizon_range, 1), calculated_harizons_mse, marker='.', markersize=8, linestyle=' ')
    plt.plot(best_horizon, min(calculated_harizons_mse), marker='.', color='r', markersize=8)
    plt.grid(True); plt.xlabel("Horizon"); plt.ylabel("MSE")

# MSE dependence on interference variance
def ex2(time, horizon):
    triangle_wave = generateTriangleWave(time)

    interferances = np.arange(0, 10, 0.5)
    MSE_values = np.zeros([interferances.size])

    for interferance in interferances:
        noise_wave = generateNoiseWave(triangle_wave, interferance)
        MSE_values[int(interferance * 2)] = MSE(triangle_wave, denoising(noise_wave, horizon), horizon)

    # MSE to interference variance plot
    plt.figure(figsize=(10, 6))
    plt.plot(interferances ** 2 / 3, MSE_values, marker='.', markersize=8, linestyle=' ')
    plt.grid(True); plt.xlabel("Variance"); plt.ylabel("MSE")

# Optimal horizon dependence on interference variance
def ex3(time, horizon_range):
    triangle_wave = generateTriangleWave(time)

    interferances = np.arange(0, 10, 0.5)
    optimal_horizons = np.zeros([interferances.size])

    for interferance in interferances:
        noise_wave = generateNoiseWave(triangle_wave, interferance)
        optimal_horizons[int(interferance * 2)] = np.argmin(simulateHorizons(triangle_wave, noise_wave, horizon_range)) + 1

    # Optimal horizont to interference variance plot
    plt.figure(figsize=(10, 6))
    plt.plot(interferances ** 2 / 3, optimal_horizons, marker='.', markersize=8, linestyle=' ')
    plt.grid(True); plt.xlabel("Variance"); plt.ylabel("Optimal horizons")

def main():
    plot_range = 100
    horizon_range = 50
    numb_points = 1500

    time = np.linspace(0, plot_range, num=numb_points)

    ex1(time, horizon_range)
    ex2(time, horizon=8)
    ex3(time, horizon_range)

    allWavesPlot(time, horizon=8)

    plt.show()

if __name__ == "__main__":
    main()