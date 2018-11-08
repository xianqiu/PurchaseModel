import time
import statistics

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt


class Problem(object):

    W = 100  # Warehouse capacity
    T = 7  # Period of purchase
    size = 180  # Data size

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.sale_data = generate_sale_data(self.mu, self.sigma, self.size)
        self.stock = [0] * self.size
        self.purchase_history = [0] * self.size
        self.stock_cache = 0
        self.empty_days = 0
        self.full_days = 0

    @property
    def bad_days(self):
        return self.empty_days + self.full_days

    def is_purchase_day(self, day):
        return True if day % self.T == 0 else False

    def reset(self):
        self.stock = [0] * self.size
        self.purchase_history = [0] * self.size
        self.empty_days = 0
        self.full_days = 0

    def out_stock(self, day, num):
        self.stock[day] = max(self.stock[day] - num, 0)

    def in_stock(self, day, num):
        self.purchase_history[day] = num
        self.stock_cache = num

    def get_stock(self, day):
        return self.stock[day]

    @property
    def sale_days(self):
        for day in range(self.size):
            self.stock[day] = self.stock[day-1] if day != 0 else 0
            yield day
            self.__fill_stock(day)
            self.out_stock(day, self.sale_data[day])

    def __fill_stock(self, day):
        if self.stock_cache == 0:
            return
        if self.is_purchase_day(day+1):
            self.stock[day] += self.stock_cache
            self.stock_cache = 0
        else:
            delta = self.W - self.stock[day]
            if self.stock_cache > delta:
                self.stock[day] = self.W
                self.stock_cache -= delta
            else:
                self.stock[day] += self.stock_cache
                self.stock_cache = 0

    def evaluate(self):
        """ Evaluate result.
        """
        empty_days = 0
        full_days = 0
        for i in range(self.size):
            if self.stock[i] <= 0:
                empty_days += 1
            if self.stock[i] > self.W:
                full_days += 1
        self.empty_days = empty_days
        self.full_days = full_days

        mean = statistics.mean(self.stock)
        stdev = statistics.stdev(self.stock)
        bad_days = empty_days + full_days
        bad_rate = bad_days / self.size
        print("bad days = %d (empty = %d, full = %d) -- bad rate = %.2f -- stock mean = %d -- stock stdev = %d"
              % (bad_days, empty_days, full_days, bad_rate, mean, stdev))


def generate_sale_data(mu, sigma, size):
    """ Sales data generates from N(mu, sigma^2).

    :param mu: mean of the normal distribution
    :param sigma: standard error of normal distribution
    :param size: data size
    :return: list
    """
    np.random.seed(int(time.time()))
    data = np.random.normal(mu, sigma, size)
    return [max(round(x), 0) for x in data.tolist()]


# Naive algorithm.
def naive(prob):
    for day in prob.sale_days:
        if not prob.is_purchase_day(day):
            continue
        needs = max(prob.T * prob.mu - prob.get_stock(day), 0)
        prob.in_stock(day, needs)


# Algorithm that considers uncertainty
def uncertain(prob):
    for day in prob.sale_days:
        if not prob.is_purchase_day(day):
            continue
        # upper bound of purchase
        ub = prob.T * prob.W - prob.get_stock(day)
        # compute optimal solution of the model
        obj = 4
        optimal_purchase_number = 0
        for x in range(0, ub+1):
            s = prob.get_stock(day)
            z1 = (x + s - 7 * prob.mu) / (2.64575 * prob.sigma)
            gamma1 = 1 - ss.norm.cdf(z1)
            z2 = (x + s - prob.W - 7 * prob.mu) / (2.64575 * prob.sigma)
            gamma2 = ss.norm.cdf(z2)
            o = (gamma1 + gamma2) * (gamma1 + gamma2)
            if o < obj:
                obj = o
                optimal_purchase_number = x
        prob.in_stock(day, optimal_purchase_number)


def run_prob(mu, sigma):
    """ Run Problem instance with naive and uncertain algorithm respectively.
    """
    prob = Problem(mu, sigma)

    # Run naive algorithm
    naive(prob)
    prob.evaluate()
    bad_days_naive = prob.bad_days

    prob.reset()

    # Run algorithm that considers uncertainty
    uncertain(prob)
    prob.evaluate()
    bad_days_uncertain = prob.bad_days

    return bad_days_naive, bad_days_uncertain


def test(sigma):
    lb, ub = 20, 80
    bad_days_naive = []
    bad_days_uncertain = []
    for mu in range(lb, ub+1):
        a, b = run_prob(mu, sigma)
        bad_days_naive.append(a)
        bad_days_uncertain.append(b)

    x = range(20, 81)
    plt.plot(x, bad_days_naive, label='naive')
    plt.plot(x, bad_days_uncertain, label='uncertain')
    plt.legend()
    plt.show()


def have_a_look(mu, sigma):
    prob = Problem(mu, sigma)

    # Run naive algorithm
    naive(prob)
    prob.evaluate()
    plt.plot(prob.stock, label='naive')

    prob.reset()

    # Run algorithm that considers uncertainty
    uncertain(prob)
    prob.evaluate()
    plt.plot(prob.stock, label='uncertain')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    #test(10)
    #test(20)
    have_a_look(20, 10)


