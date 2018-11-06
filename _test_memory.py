from histogram.histogram import Histogram1D
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':

    n = 1

    hist = Histogram1D(bin_edges=np.arange(0, 4095), data_shape=(n, 1296, ))

    input('Press Enter to start')
    for i in tqdm(range(10000)):

        for j in range(n):

            data = np.random.uniform(1000, 3000, size=(1296, 10))
            hist.fill(data, indices=(j, ))
