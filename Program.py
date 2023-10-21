import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

numb_points = 300                     # number of points that creat plot
freq = 1                              # frequency
amp = 1                              # amplitude

time = np.linspace(0, 10, num=numb_points)

def plots():
    # Size of the window
    plt.figure(figsize=(12, 6))

    # Generate triangle wave 
    generator = amp * signal.sawtooth(freq * time, width=0.5)

    # Add noise 
    for i in range(numb_points):
        generator[i] += np.random.uniform(0, 1) - 0.5

    # Add triangle functions to the plot
    plt.plot(time, generator, ".", markersize=5, label="Function with noise")
    # plt.plot(time, amp * signal.sawtooth(freq * time, width=0.5), label="Function without noise", color="r") 

    # Plot properties
    plt.grid(); plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12)); plt.show()

def main():
    plots()

if __name__ == "__main__":
    main()