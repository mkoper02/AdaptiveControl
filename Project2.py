import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


# Generate traingle wave
def generateTriangleWave(time):
    amp = 0.1           # amplitude
    freq = 0.08         # frequency

    return (amp * signal.sawtooth(freq * time, width=0.5)) + 1


# Generate random samples from a normal distribution and return as an array 
def generateInput(size):
    row_size = 3
    x = np.random.normal(0, 1, size)
    x_n = np.zeros((size - 2, row_size))  

    for row in range(len(x_n)):
        for i in range(row_size): x_n[row][i] = x[row + i]

    return x_n


# Output based on set parameters and noise
def calculateOutput(input_k, input_k1, input_k2, parameters, interferance):
    return input_k * parameters[0] + input_k1 * parameters[1] + input_k2 * parameters[2] + interferance


# Generate output based on set parameters and noise (static system)
def generateOutputStatic(input, parameters, noise):
    output = np.zeros((len(input), 1))

    for point in range(len(input)):
        output[point] = calculateOutput(input[point][0], input[point][1], input[point][2], parameters, noise[point])

    return output


# Generate output based on set parameters and noise (dynamic system)
def generateOutputDynamic(input, triangle_wave, noise):
    output = np.zeros((len(input), 1))

    for point in range(len(input)):
        parameters = [1, triangle_wave[point], 1]

        output[point] = calculateOutput(input[point][0], input[point][1], input[point][2], parameters, noise[point])

    return output


def mse(original, estimated):
    return np.mean(np.square(np.subtract(original, estimated)))


def wrmnk(Xn, Yn, _lambda):
    Pn = np.identity(3) * 10 ** 5
    b = np.zeros((3, 1))
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


def _wrmnk(b, P, x, y, _lambda):
    x = np.array([x]).T

    numerator = np.matmul(np.matmul(np.matmul(P, x), x.T), P)
    denominator = _lambda + np.matmul(np.matmul(x.T, P), x)
    P = (1 / _lambda) * (P - (numerator / denominator))
    epsilon = y - np.matmul(x.T, b)
    b = b + np.matmul(np.matmul(P, x), epsilon)

    return b, P


def adaptiveInput(input, _lambda, wanted_value, noise, time):
    Y = np.zeros((len(time), 1))
    P = np.identity(3) * 10 ** 5
    b = np.zeros((3, 1))
    triangle_wave = generateTriangleWave(time)

    for i in range(len(time)):
        y = np.array([calculateOutput(input[0], input[1], input[2], [1, triangle_wave[i], 1], noise[i])])
        Y[i] = y

        b, P = _wrmnk(b, P, input, y, _lambda)

        input_n = (wanted_value - b[1] * input[0] - b[2] * input[1]) / b[0]
        input = np.concatenate([input_n, input[:2]])

    # print(Y)
    return Y


def main():
    plot_range = 100
    numb_points = 1000

    time = np.linspace(0, plot_range, num=numb_points)

    interferance_range = 0.01
    noise = np.zeros((len(time), 1))
    rng = np.random.default_rng(50)
    noise = rng.uniform(-interferance_range / 2, interferance_range / 2, len(time))

    Xn = generateInput(len(time))
    Yn1 = generateOutputStatic(Xn, [1, 1, 1], noise)

    # print(Yn1)

    # STATIC SYSTEM
    # b = 0

    # Input vs output static
    # plt.figure(figsize=(14, 8))
    # plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.06)
    # plt.plot(time[:250], [i[0] for i in Xn][:250])
    # plt.plot(time[:250], Yn1[:250])
    # plt.grid(True); plt.xlabel('Czas')

    bs1 = wrmnk(Xn, Yn1, _lambda=1)

    # b0
    # plt.figure(figsize=(14, 8))
    # plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.06)
    # plt.plot(time[75 + 2:], [b[0] for b in bs1][75:])
    # plt.plot(time[75:], np.ones(len(time))[75:])
    # plt.grid(True); plt.title(r'$b_{0}$'); plt.xlabel('Czas')

    # b1
    # plt.figure(figsize=(14, 8))
    # plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.06)
    # plt.plot(time[75 + 2:], [b[1] for b in bs1][75:])
    # plt.plot(time[75:], np.ones(len(time))[75:])
    # plt.grid(True); plt.title(r'$b_{1}$'); plt.xlabel('Czas')
    
    # b2
    # plt.figure(figsize=(14, 8))
    # plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.06)
    # plt.plot(time[75 + 2:], [b[2] for b in bs1][75:])
    # plt.plot(time[75:], np.ones(len(time))[75:])
    # plt.grid(True); plt.title(r'$b_{2}$'); plt.xlabel('Czas')

    # DYNAMIC SYSTEM
    # b = generateTriangleWave(time)

    b1_triangle = generateTriangleWave(time)
    Yn2 = generateOutputDynamic(Xn, b1_triangle, noise)

    # Input vs output dynamic
    # plt.figure(figsize=(14, 8))
    # plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.06)
    # plt.plot(time[:250], [i[0] for i in Xn][:250])
    # plt.plot(time[:250], Yn2[:250])
    # plt.grid(True); plt.xlabel('Czas')

    # b1 - lambda 1
    bs2 = wrmnk(Xn, Yn2, _lambda=1)

    # plt.figure(figsize=(14, 8))
    # plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.06)
    # plt.plot(time[25:1000], [b[1] for b in bs2][25:1000])
    # plt.plot(time[25:1000], b1_triangle[25:1000])
    # plt.grid(True); plt.title(r'$b_{1}$'); plt.xlabel('Czas')

    # b1 - lambda 0.97
    bs2 = wrmnk(Xn, Yn2, _lambda=0.97)

    # plt.figure(figsize=(14, 8))
    # plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.06)
    # plt.plot(time[25:1000], [b[1] for b in bs2][25:1000])
    # plt.plot(time[25:1000], b1_triangle[25:1000])
    # plt.grid(True); plt.ylabel(r'$b_{1}$'); plt.xlabel('Czas')

    # b1 - lambda 0.95
    bs2 = wrmnk(Xn, Yn2, _lambda=0.95)

    # plt.figure(figsize=(14, 8))
    # plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.06)
    # plt.plot(time[25:1000], [b[1] for b in bs2][25:1000])
    # plt.plot(time[25:1000], b1_triangle[25:1000])
    # plt.grid(True); plt.ylabel(r'$b_{1}$'); plt.xlabel('Czas')

    value = 1.5

    # [y values, mse, lambda]
    y_arr = []
    mse_lambda_arr = np.zeros((10, 2))
    i = 0

    for lmbd in np.arange(0.9, 1, 0.01):
        y = adaptiveInput(Xn[0], lmbd, value, noise, time)
        y_arr.append(y)
        mse_lambda_arr[i] = [mse(value, y), lmbd]

        i += 1

    print(mse_lambda_arr)
    print(min(mse_lambda_arr[:, 0]))
    print(min(mse_lambda_arr[:, 1]))

    # best lambda
    plt.figure(figsize=(14, 8))
    plt.subplots_adjust(left=0.06, right=0.95, top=0.98, bottom=0.06)
    plt.plot(time[10:len(y)], y_arr[np.argmin(mse_lambda_arr[:, 0])][10:])
    plt.plot(time[10:len(y)], np.full(len(y) - 10, value))
    plt.grid(True); plt.ylim(value - 0.1, value + 0.1); plt.xlabel('Czas')

    # lambda 0.98
    plt.figure(figsize=(14, 8))
    plt.subplots_adjust(left=0.06, right=0.95, top=0.98, bottom=0.06)
    plt.plot(time[10:len(y)], y_arr[8][10:])
    plt.plot(time[10:len(y)], np.full(len(y) - 10, value))
    plt.grid(True); plt.ylim(value - 0.1, value + 0.1); plt.xlabel('Czas')

    # lambda 0.95
    plt.figure(figsize=(14, 8))
    plt.subplots_adjust(left=0.06, right=0.95, top=0.98, bottom=0.06)
    plt.plot(time[10:len(y)], y_arr[5][10:])
    plt.plot(time[10:len(y)], np.full(len(y) - 10, value))
    plt.grid(True); plt.ylim(value - 0.1, value + 0.1); plt.xlabel('Czas')

    plt.show()


if __name__ == "__main__":
    main()