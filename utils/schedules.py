from numpy import pi, tanh, cos


def scheduler(fn, max_iterations, cycles, cycle_proportion, beta_lag):
    ## Time period of each cycle
    T = max_iterations / cycles
    ## reach max value (1) at
    N = float(T) * cycle_proportion

    ## lag beta by these many iterations
    lag = float(T) * beta_lag

    def linear(x):
        return min(1, (x % T) / N)

    def cosine(x):
        if x % T > N:
            return 1
        return 0.5 * (1 - cos((pi * (x % T)) / N))

    def tan_h(x):
        if x % T > N:
            return 1
        return 0.5 * (1 + tanh((x % T - N / 2) / (N / 6.5)))

    _alpha_val = {
        'linear': linear,
        'cosine': cosine,
        'tanh': tan_h
    }.get(fn, 'linear')

    def schedule(x):
        alpha = _alpha_val(x)
        beta = alpha
        if lag != 0:
            if x < lag:
                return alpha, 0
            beta = _alpha_val(x - lag)
        return alpha, beta

    return schedule


def test():
    import matplotlib as mpl
    # mpl.use("qt5Agg")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    n = 40000
    # sch = scheduler('cosine', n, 3, 0.75, 0)
    # cos = [sch(i)[0] for i in range(n)]
    # sch = scheduler('tanh', n, 3, 0.75, 0)
    # tan = [sch(i)[0] for i in range(n)]
    # plt.plot(cos, color="blue", label="cos")
    # plt.plot(tan, color="red", label="tan")

    sch = scheduler('tanh', n, 3, 0.75, 0.1)
    alpha = []
    beta = []
    for i in range(n):
        a, b = sch(i)
        alpha.append(a)
        beta.append(b)

    plt.plot(alpha, color="blue", label="alpha")
    plt.plot(beta, color="red", label="beta")
    plt.legend()
    plt.savefig('./schedule_test.png')


if __name__ == '__main__':
    test()
