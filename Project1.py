import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def denoising(noise_points, horizon):
    denoised = np.zeros(len(noise_points))

    for segment in range(len(noise_points)):
        sum = 0

        # Use values "from the future"
        for point in range(horizon):
            if horizon + segment > len(noise_points): break
            sum += noise_points[segment + point]
        
        # Use values "from the past"
        # for point in range(horizon):
        #     if point + segment < horizon: break
        #     sum += points[segment - point]
        
        denoised[segment] = sum / horizon
        
    return denoised

def addNoise(triangle_wave, rand_range): 
    noise_points = np.copy(triangle_wave)

    for i in range(len(triangle_wave)):
        noise_points[i] += np.random.uniform(0, rand_range) - (rand_range / 2)

    return noise_points

def generateWaves(time, horizon, rand_range):
    amp = 2         # amplitude
    freq = 1        # frequency

    # Generate triangle wave, points with noise and denoised wave
    triangle_wave = amp * signal.sawtooth(freq * time, width=0.5)
    generated_noise_points = addNoise(triangle_wave, rand_range)
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

    print(min(MSE_values), np.argmin(MSE_values))

    # MSE values on plot
    plt.scatter(np.arange(0, horizon_range, 1), MSE_values, marker=".")
    # Highlight min value
    plt.plot(np.argmin(MSE_values), min(MSE_values), marker='.', color='r')
    plt.grid(); plt.xlabel("Horizon"); plt.ylabel("MSE")

def createWavePlots(triangle_wave, generated_noise_points, denoised_wave, horizon, numb_points, time):
    plt.figure(figsize=(12, 7))

    # Create plots with scope lowered by the horizon
    plt.plot(time, triangle_wave, label="Function without noise", color="r") 
    plt.plot(time, generated_noise_points, ".", markersize=5, label="Points with noise")
    plt.plot(time[:numb_points - horizon], denoised_wave[:numb_points - horizon], "*-", markersize=5,label="Denoised function")

    # Plot properties
    plt.grid(); plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16))

def main():
    horizon = 20
    horizon_simulate = 75
    plot_range = 100
    numb_points = 2000
    rand_range = 1

    time = np.linspace(0, plot_range, num=numb_points)
    triangle_wave, generated_noise_points, denoised_wave = generateWaves(time, horizon, rand_range)

    simulateHorizons(horizon_simulate, triangle_wave, denoised_wave)
    createWavePlots(triangle_wave, generated_noise_points, denoised_wave, horizon, numb_points, time)

    plt.show()

if __name__ == "__main__":
    main()