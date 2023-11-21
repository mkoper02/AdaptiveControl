import matplotlib.pyplot as plt
import numpy as np

# Output based on set parameters and noise
def calculateOutput(input_k, input_k1, input_k2, interferance):
    bs = [1, 1, 1]

    return input_k * bs[0] + input_k1 * bs[1] + input_k2 * bs[2] + interferance


# Generate random samples from a normal distribution and return as an array 
def generateInput(size):
    row_size = 3
    x = np.random.normal(0, 1.5, size)
    x_n = np.zeros((size - 2, row_size))  

    for row in range(len(x_n)):
        for i in range(row_size): x_n[row][i] = x[row + i]

    return x_n


# Generate output based on set parameters and noise
def generateOutput(input):
    output = np.zeros((len(input), 1))
    interferance_range = 1

    for point in range(len(input)):
        output[point] = calculateOutput(input[point][0], input[point][1], input[point][2], np.random.uniform(-interferance_range / 2, interferance_range / 2))

    return output


def wrmnk(Xn, Yn, _lambda):
    Pn = np.identity(3) * 10 ** 5
    b = np.ones((3, 1))
    b_arr = []

    for x, y in zip(Xn, Yn):
        x = np.array([x]).T
        y = np.array([y])

        numerator = Pn @ x @ x.T @ Pn
        denominator = _lambda + (x.T @ Pn @ x)
        Pn = (1 / _lambda) * (Pn - (numerator / denominator))
        epsilon = y - np.dot(x.T, b)
        b = b + (Pn @ x @ epsilon)
        b_arr.append(b)

    return np.array(b_arr)


def main():
    plot_range = 1000    
    numb_points = 10000

    time = np.linspace(0, plot_range, num=numb_points)
    Xn = generateInput(len(time))
    Yn = generateOutput(Xn)    

    bs1 = wrmnk(Xn, Yn, _lambda=1)

    # b0
    plt.figure(figsize=(14, 8))
    plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.06)
    plt.plot(time[75 + 2:], [b[0] for b in bs1][75:])
    plt.plot(time[75:], np.ones(len(time))[75:])
    plt.grid(True); plt.ylabel(r'$b_{0}$'); plt.xlabel('Czas [s]')

    # # b1
    plt.figure(figsize=(14, 8))
    plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.06)
    plt.plot(time[75 + 2:], [b[1] for b in bs1][75:])
    plt.plot(time[75:], np.ones(len(time))[75:])
    plt.grid(True); plt.ylabel(r'$b_{1}$'); plt.xlabel('Czas [s]')
    
    # b2
    plt.figure(figsize=(14, 8))
    plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.06)
    plt.plot(time[75 + 2:], [b[2] for b in bs1][75:])
    plt.plot(time[75:], np.ones(len(time))[75:])
    plt.grid(True); plt.ylabel(r'$b_{2}$'); plt.xlabel('Czas [s]')

    plt.show()


if __name__ == "__main__":
    main()