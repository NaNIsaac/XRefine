import numpy as np

from xrefine.simulator import simulate_reflectivity


def main():
    q = np.linspace(0.01, 1, 100)
    simulate_reflectivity(q)


if __name__ == "__main__":
    main()
