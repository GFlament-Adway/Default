from data_generator import get_data
from sampler import frailty


if __name__ == "__main__":
    X, Y, Times, Cens = get_data(10, 10, 1)
    frail = frailty(X, Times, Cens)
    frail._likelihood()
