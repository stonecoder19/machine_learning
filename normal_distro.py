from matplotlib import pyplot as plt
import math
import random
from collections import Counter


def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))


def normal_cdf(x, mu=0,sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def bernoulli_trial(p):
    return 1 if random.random() < p else 0

def binomial(n,p):
    return sum(bernoulli_trial(p) for _ in range(n))


def bucketize(point,bucket_size):
    return bucket_size * math.floor(point / bucket_size)

def make_histogram(points, bucket_size):
    return Counter(bucketize(point, bucket_size) for  point in points)

def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    if  mu!=0 or sigma != 1:
        return mu + sigma * inverse_norma_cdf(p, tolerance=tolerance)

    low_z, low_p = -10.0, 0
    hi_z, hi_p = 10.0, 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            hi_z, hi_p = mid_z, mid_p
        else:
            break
    return mid_z

def random_normal():
    return inverse_normal_cdf(random.random())


xs = [random.random() for _ in range(1000)]
ys1 = [x + random_normal() / 2 for x in xs]
ys2 = [-x + random_normal() / 2 for x in xs]

def plot_histogram(points, bucket_size, title=""):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)
    plt.show()

def make_hist(p, n, num_points):
    data = [binomial(n,p) for _ in range(num_points)]
    
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')
    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))
    xs = range(min(data), max(data)+1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma)
           for i in xs]
    plt.plot(xs, ys)
    plt.title("Binomial Distribution vs. Normal Approximation")
    plt.show()

if __name__ == '__main__':
    #make_hist(0.75, 100, 10000)
    #random.seed(0)
    #uniform = [200 * random.random() - 100 for _ in range(10000)]

    #normal = [57 * inverse_normal_cdf(random.random()) for _ in range(10000)]

    #plot_histogram(uniform, 10, "Uniform Histogram")

    #plot_histogram(normal, 10, "Normal Histogram")
    plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
    plt.scatter(xs, ys2, marker='.', color='gray', label='ys2')
    plt.xlabel('xs')
    plt.ylabel('ys')
    plt.legend(loc=9)
    plt.title("Very Different Joint Distributions")
    plt.show()

#xs = [x / 10.0 for x in range(-50,50)]

#plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs],'-',label='mu=0,sigma=1')
#plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs],'--',label='mu=0,sigma=2')
#plt.plot(xs, [normal_pdf(x, sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
#plt.plot(xs, [normal_pdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
#plt.plot(xs, [normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
#plt.plot(xs, [normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
#plt.plot(xs, [normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
#plt.plot(xs, [normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
#plt.legend(loc=4)
#plt.title("Various Normal pdfs")
#plt.show()
