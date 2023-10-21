import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

numb_points = 100                     # number of points that creat plot
horizon = 5                           # 

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
    for i in range(len(points)):
        points[i] += np.random.uniform(0, 1) - 0.5

    return points

# frequency, amplitude
def plots(freq, amp):
    # Size of the window
    plt.figure(figsize=(12, 7))

    time = np.linspace(0, 10, num=numb_points)

    # Generate triangle wave 
    generator = amp * signal.sawtooth(freq * time, width=0.5)

    # Create plots with scope lowered by the horizon
    plt.plot(time[:numb_points - horizon], generator[:numb_points - horizon], label="Function without noise", color="r") 
    plt.plot(time[:numb_points - horizon], addNoise(generator)[:numb_points - horizon], ".", markersize=5, label="Function with noise")
    plt.plot(time[:numb_points - horizon], denoising(generator)[:numb_points - horizon], label="Denoised function")

    # Plot properties
    plt.grid(); plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16)); plt.show()

def main():
    plots(freq=1, amp=2)

if __name__ == "__main__":
    main()
