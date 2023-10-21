import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def denoising(points, horizon):
    denoised = np.zeros(len(points))

    for segment in range(len(points)):
        sum = 0

        for point in range(horizon):
             if horizon + segment > len(points): break
             sum += points[segment + point]
        
        denoised[segment] = sum / horizon
        
    return denoised

def addNoise(points): 
    noise_points = np.copy(points)

    for i in range(len(points)):
        noise_points[i] += np.random.uniform(0, 1) - 0.5

    return noise_points

def generateWaves(time, horizon):
    amp = 2         # amplitude
    freq = 1        # frequency

    # Generate triangle wave, points with noise and denoised wave
    triangle_wave = amp * signal.sawtooth(freq * time, width=0.5)
    generated_noise_points = addNoise(triangle_wave)
    denoised_wave = denoising(generated_noise_points, horizon)

    return triangle_wave, generated_noise_points, denoised_wave

def MSE(traingle_wave, denoised_points, horizon):
    mse = 0

    for point in range(len(traingle_wave) - horizon):
        mse += pow(denoised_points[point] - traingle_wave[point], 2)

    return mse / (len(traingle_wave) - horizon)

def simulateHorizons(horizon_range, triangle_wave, denoised_wave):
    MSE_values = np.zeros([horizon_range])

    for horizon in range(1, horizon_range + 1): 
        MSE_values[horizon - 1] = MSE(triangle_wave, denoised_wave, horizon)

    # print(min(MSE_values))

    # MSE values on plot
    plt.scatter(np.arange(0, horizon_range, 1), MSE_values, marker=".")
    # Highlight min value
    plt.plot(np.argmin(MSE_values), min(MSE_values), marker='.', color='r')
    plt.grid()

def createWavePlots(triangle_wave, generated_noise_points, denoised_wave, horizon, numb_points, time):
    plt.figure(figsize=(12, 7))

    # Create plots with scope lowered by the horizon
    plt.plot(time, triangle_wave, label="Function without noise", color="r") 
    plt.plot(time, generated_noise_points, ".", markersize=5, label="Points with noise")
    plt.plot(time[:numb_points - horizon], denoised_wave[:numb_points - horizon], "*-", markersize=5,label="Denoised function")

    # Plot properties
    plt.grid(); plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16)); plt.show()

def main():
    horizon = 20
    plot_range = 100
    numb_points = 2000

    time = np.linspace(0, plot_range, num=numb_points)
    triangle_wave, generated_noise_points, denoised_wave = generateWaves(time, horizon)

    simulateHorizons(horizon, triangle_wave, denoised_wave)
    createWavePlots(triangle_wave, generated_noise_points, denoised_wave, horizon, numb_points, time)

if __name__ == "__main__":
    main()