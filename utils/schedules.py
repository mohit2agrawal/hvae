from numpy import pi, tanh, cos


def scheduler(
    fn, max_iterations, cycles, cycle_proportion, beta_lag, zero_start
):
    ## Time period of each cycle
    T = max_iterations / cycles
    # T = 5000
    ## reach max value (1) at
    N = float(T) * cycle_proportion

    ## lag beta by these many iterations
    lag = float(T) * beta_lag

    def linear(x, t, n):
        return min(1, (x % t) / n)

    def cosine(x, t, n):
        if x % t > n:
            return 1
        return 0.5 * (1 - cos((pi * (x % t)) / n))

    def tan_h(x, t, n):
        if x % t > n:
            return 1
        return 0.5 * (1 + tanh((x % t - n / 2) / (n / 6.5)))

    _alpha_val = {
        'linear': linear,
        'cosine': cosine,
        'tanh': tan_h
    }.get(fn, 'linear')

    ## simple schedule without zero start
    def schedule(x):
        alpha = _alpha_val(x, T, N)
        beta = alpha
        if lag != 0:
            if x < lag:
                return alpha, 0
            beta = _alpha_val(x - lag, T, N)
        return alpha, beta

    # return schedule

    # def schedule(x):
    #     if x % T < 700:
    #         return 0, 0
    #     x -= 700
    #     alpha = tan_h(x, T, N)
    #     beta = alpha
    #     if lag != 0:
    #         if x % T < lag:
    #             return alpha, 0
    #         beta = tan_h(x - lag, T, N)
    #     return alpha, beta

    r = zero_start

    ## schedule with zero start
    def schedule_zs(x):
        x = float(x)
        # r = 0
        # r = 0.75
        alpha_x = x % T
        if alpha_x / T < r:
            alpha = 0
            return 0, 0
        else:
            alpha_x -= r * T
            alpha = _alpha_val(alpha_x, T * (1 - r), N * (1 - r))
        beta = alpha
        if lag != 0:
            ## for initial zero
            if x % T < lag:
                return alpha, 0
            x -= lag
            x %= T
            if x / T < r:
                return alpha, 0
            x -= r * T
            # beta = _alpha_val(x, T * (1 - r) - lag, N * (1 - r))
            ## to reach 1 along with alpha
            beta = _alpha_val(x, T * (1 - r) - lag, N * (1 - r) - lag)
        return alpha, beta

    if zero_start:
        return schedule_zs
    return schedule


def test():
    import matplotlib as mpl
    # mpl.use("qt5Agg")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    n = 15000
    # sch = scheduler('cosine', n, 3, 0.75, 0)
    # cos = [sch(i)[0] for i in range(n)]
    # sch = scheduler('tanh', n, 3, 0.75, 0)
    # tan = [sch(i)[0] for i in range(n)]
    # plt.plot(cos, color="blue", label="cos")
    # plt.plot(tan, color="red", label="tan")

    sch = scheduler('tanh', n, 2, 1, 0.15, 0.75)
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
