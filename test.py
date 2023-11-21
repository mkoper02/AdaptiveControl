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
    x = np.random.normal(0, 1.5, size)
    x_n = np.zeros((size - 2, row_size))  

    for row in range(len(x_n)):
        for i in range(row_size): x_n[row][i] = x[row + i]

    return x_n


# Output based on set parameters and noise
def calculateOutput(input_k, input_k1, input_k2, parameters, interferance):
    return input_k * parameters[0] + input_k1 * parameters[1] + input_k2 * parameters[2] + interferance


# Generate output based on set parameters and noise (static system)
def generateOutputStatic(input, parameters):
    output = np.zeros((len(input), 1))
    interferance_range = 1

    for point in range(len(input)):
        output[point] = calculateOutput(input[point][0], input[point][1], input[point][2], parameters, np.random.uniform(-interferance_range / 2, interferance_range / 2))

    return output


# Generate output based on set parameters and noise (dynamic system)
def generateOutputDynamic(input, triangle_wave):
    output = np.zeros((len(input), 1))
    interferance_range = 1

    for point in range(len(input)):
        parameters = [1, triangle_wave[point], 1]

        output[point] = calculateOutput(input[point][0], input[point][1], input[point][2], parameters, np.random.uniform(-interferance_range / 2, interferance_range / 2))

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

def calculateQualityCriterion(Yn, expected_output):
    N = len(Yn)
    return (1/N) * np.sum((Yn - expected_output)**2)

def main():
    plot_range = 1000    
    numb_points = 10000

    time = np.linspace(0, plot_range, num=numb_points)
    Xn = generateInput(len(time))
    # Yn1 = generateOutputStatic(Xn, [1, 1, 1])    

    # #Input vs output
    # plt.figure(figsize=(14, 8))
    # plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.06)
    # plt.plot(time[:250], [i[0] for i in Xn][:250])
    # plt.plot(time[:250], Yn1[:250])
    # plt.grid(True); plt.xlabel('Czas [s]')

    # bs1 = wrmnk(Xn, Yn1, _lambda=1)

    # #b0
    # plt.figure(figsize=(14, 8))
    # plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.06)
    # plt.plot(time[75 + 2:], [b[0] for b in bs1][75:])
    # plt.plot(time[75:], np.ones(len(time))[75:])
    # plt.grid(True); plt.ylabel(r'$b_{0}$'); plt.xlabel('Czas [s]')

    # # b1
    # plt.figure(figsize=(14, 8))
    # plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.06)
    # plt.plot(time[75 + 2:], [b[1] for b in bs1][75:])
    # plt.plot(time[75:], np.ones(len(time))[75:])
    # plt.grid(True); plt.ylabel(r'$b_{1}$'); plt.xlabel('Czas [s]')
    
    # # b2
    # plt.figure(figsize=(14, 8))
    # plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.06)
    # plt.plot(time[75 + 2:], [b[2] for b in bs1][75:])
    # plt.plot(time[75:], np.ones(len(time))[75:])
    # plt.grid(True); plt.ylabel(r'$b_{2}$'); plt.xlabel('Czas [s]')

    b1_triangle = generateTriangleWave(time)

    Yn2 = generateOutputDynamic(Xn, b1_triangle)
    # bs2 = wrmnk(Xn, Yn2, _lambda=1)

    # plt.plot(time[2:], Yn2)

    # # b1
    # plt.figure(figsize=(14, 8))
    # plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.06)
    # plt.plot(time[25:1000], [b[1] for b in bs2][25:1000])
    # plt.plot(time[25:1000], b1_triangle[25:1000])
    # plt.grid(True); plt.ylabel(r'$b_{1}$'); plt.xlabel('Czas [s]')

    bs2 = wrmnk(Xn, Yn2, _lambda=0.98)

    # # b1 - lambda diffrent from 1
    # plt.figure(figsize=(14, 8))
    # plt.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.06)
    # plt.plot(time[25:1000], [b[1] for b in bs2][25:1000])
    # plt.plot(time[25:1000], b1_triangle[25:1000])
    # plt.grid(True); plt.ylabel(r'$b_{1}$'); plt.xlabel('Czas [s]')
    
    for b in bs2:
        b0, b1, b2 = b[0], b[1], b[2]
        for point in range(len(time)):
            u2 = np.zeros((len(time), 1))
            u3 = np.zeros((len(time), 1))
            u2[point][0] = [point][0], u3[point][0] = [point][1]
            u=(1-b1+u2-b2+u3)/b0
            print(u)


    # liczenie tego lambda optymalnego    
    # plt.show()
    # expected_output = np.ones(len(Yn1))
    # lambdas = np.arange(0.9,1,0.01)  
    # criteria_values = []
    # for _lambda in lambdas:
    #    bs = wrmnk(Xn, Yn1, _lambda)
    #    Yn_estimated = Xn @ bs[-1]  # Ostatnia estymowana wartość parametrów
    #    criterion_value = calculateQualityCriterion(Yn_estimated, expected_output)
    #    criteria_values.append(criterion_value)

    # min_index = np.argmin(criteria_values)
    # best_lambda = lambdas[min_index]
    # best_criterion_value = criteria_values[min_index]
 
    # # Wykres kryterium jakości w zależności od lambdy
    # plt.figure(figsize=(14, 8))
    # plt.plot(lambdas, criteria_values, marker='o')
    # plt.xlabel('Wartość lambda')
    # plt.ylabel('Kryterium jakości')
    # plt.title('Kryterium jakości w zależności od lambdy')
    # plt.plot(best_lambda,best_criterion_value, marker='X', color='r', markersize=8)
    # plt.grid(True)
    # plt.show()
    

if __name__ == "__main__":
    main()