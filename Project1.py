import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

numb_points = 10000                     # number of points that creat plot
horizon = 10

def denoising(points):
    denoised = np.empty([numb_points])

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

def MSE(traingle_wave, denoised_points):
    MSE = 0

    for point in range(len(traingle_wave) - horizon):
        MSE += pow(denoised_points[point] - traingle_wave[point], 2)

    return MSE / (len(traingle_wave) - horizon)

# frequency, amplitude
def plots(freq, amp):
    # Size of the window
    plt.figure(figsize=(12, 7))

    time = np.linspace(0, 100, num=numb_points)

    # Generate triangle wave, 
    triangle_wave = amp * signal.sawtooth(freq * time, width=0.5)
    generated_noise_points = addNoise(triangle_wave)
    denoised_wave = denoising(generated_noise_points)

    # Create plots with scope lowered by the horizon
    plt.plot(time[:numb_points - horizon], triangle_wave[:numb_points - horizon], label="Function without noise", color="r") 
    plt.plot(time[:numb_points - horizon], generated_noise_points[:numb_points - horizon], ".", markersize=5, label="Points with noise")
    plt.plot(time[:numb_points - horizon], denoised_wave[:numb_points - horizon], label="Denoised function")

    print(MSE(triangle_wave, denoised_wave))

    # Plot properties
    plt.grid(); plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16)); plt.show()

def main():
    plots(freq=1, amp=2)

if __name__ == "__main__":
    main()
