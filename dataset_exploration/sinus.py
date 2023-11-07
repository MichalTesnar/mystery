from math import sin, exp
import numpy as np
import matplotlib.pyplot as plt


def load_data(size):
    """
    Load the data from the storage.
    """
    sample_rate = int(10000*size)
    domain = np.linspace(-6, 6, num=sample_rate)
    domain_y = toy_function(domain)
    return domain, domain_y


def toy_function(input):
    """
    Generating sinusiod training data with added noise.
    """
    output = []
    for inp in input:
        std = max(0.15 / (1.0 + exp(-inp)), 0)
        out = sin(inp)  # + np.random.normal(0, std)
        output.append(5 * out)
    return np.array(output)


def extra_plots(domain, domain_y):
    fig, ax = plt.subplots()
    ax.set_title(f"Sinus Toy Function", fontsize=20)
    ax.plot(domain, domain_y, '.', markersize=5, label=r'$5\sin(x)$')
    ax.legend(loc='upper left', fontsize=20)
    plt.savefig(f"sinus.png")
    plt.show()
    plt.close()


d, d_y = load_data(0.01)
extra_plots(d, d_y)
