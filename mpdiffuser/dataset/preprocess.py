import numpy as np
import jax.numpy as jnp

class STDNormalizer:
    def __init__(self, X):
        super().__init__()
        self.mean = X.mean(axis=tuple(range(X.ndim - 1)))
        self.std = X.std(axis=tuple(range(X.ndim - 1)))

    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-6)
    
    def unnormalize(self, x):
        return x * self.std + self.mean
    

class MinMaxNormalizer:
    def __init__(self, X):
        super().__init__()
        self.xmin = X.min(axis=tuple(range(X.ndim - 1)))
        self.xmax = X.max(axis=tuple(range(X.ndim - 1)))

    def normalize(self, x):
        xmin = np.expand_dims(self.xmin, axis=tuple(range(x.ndim - 1)))
        xmax = np.expand_dims(self.xmax, axis=tuple(range(x.ndim - 1)))

        x = (x - xmin) / (xmax - xmin + 1e-6) # [ 0, 1 ]
        return 2 * x - 1  # [ -1, 1 ]
    
    def unnormalize(self, x):
        xmin = np.expand_dims(self.xmin, axis=tuple(range(x.ndim - 1)))
        xmax = np.expand_dims(self.xmax, axis=tuple(range(x.ndim - 1)))

        x = (x + 1) / 2.  # [ 0, 1 ]
        return x * (xmax - xmin) + xmin

class CDFNormalizer:
    '''
        makes training data uniform (over each dimension) by transforming it with marginal CDFs
    '''

    def __init__(self, X):
        super().__init__()
        self.dim = X.shape[-1]
        self.cdfs = [CDFNormalizer1d(X[..., i]) for i in range(self.dim)]

    @property
    def xmin(self):
        return np.array([cdf.xmin for cdf in self.cdfs])
    
    @property
    def xmax(self):
        return np.array([cdf.xmax for cdf in self.cdfs])
    
    @property
    def ymin(self):
        return np.array([cdf.ymin for cdf in self.cdfs])
    
    @property
    def ymax(self):
        return np.array([cdf.ymax for cdf in self.cdfs])
    
    @property
    def quantiles(self):
        return [cdf.quantiles for cdf in self.cdfs]
    
    @property
    def cumprobs(self):
        return [cdf.cumprobs for cdf in self.cdfs]

    def __repr__(self):
        return f'[ CDFNormalizer ] dim: {self.dim}\n' + '    |    '.join(
            f'{i:3d}: {cdf}' for i, cdf in enumerate(self.cdfs)
        )

    def wrap(self, fn_name, x):
        shape = x.shape
        lib = np if isinstance(x, np.ndarray) else jnp
        # reshape to 2d
        x = x.reshape(-1, self.dim) 
        out = []
        for i, cdf in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            out.append(fn(x[:, i]))
        out = lib.stack(out, axis=-1)
        return out.reshape(shape)

    def normalize(self, x):
        return self.wrap('normalize', x)

    def unnormalize(self, x):
        return self.wrap('unnormalize', x)


class CDFNormalizer1d:
    '''
        CDF normalizer for a single dimension
    '''

    def __init__(self, X):
        self.quantiles, self.cumprobs = empirical_cdf(X.flatten())

        self.xmin, self.xmax = self.quantiles.min(), self.quantiles.max()
        self.ymin, self.ymax = self.cumprobs.min(), self.cumprobs.max()

    def __repr__(self):
        return (
            f'[{np.round(self.xmin, 2):.4f}, {np.round(self.xmax, 2):.4f}'
        )

    def normalize(self, x):
        x_shp = x.shape
        lib = np if isinstance(x, np.ndarray) else jnp
    
        x = lib.clip(x.flatten(), self.xmin, self.xmax)
        # [ 0, 1 ]
        y = lib.interp(x, self.quantiles, self.cumprobs)
        # [ -1, 1 ]

        y = 2 * y - 1
        return y.reshape(x_shp)

    def unnormalize(self, x):
        '''
            X : [ -1, 1 ]
        '''
        x_shp = x.shape
        lib = np if isinstance(x, np.ndarray) else jnp
        # [ -1, 1 ] --> [ 0, 1 ]
        x = (x.flatten() + 1) / 2.

        x = lib.clip(x, self.ymin, self.ymax)
        y = lib.interp(x, self.cumprobs, self.quantiles)
        return y.reshape(x_shp)


def empirical_cdf(sample):
    # https://stackoverflow.com/a/33346366
    lib = np if isinstance(sample, np.ndarray) else jnp

    # find the unique values and their corresponding counts
    quantiles, counts = lib.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = lib.cumsum(counts).astype(lib.double) / sample.size

    return quantiles, cumprob

def get_normalizer(name):
    if name == 'cdf':
        return CDFNormalizer
    elif name == 'std':
        return STDNormalizer
    elif name == 'minmax':
        return MinMaxNormalizer
    else:
        raise ValueError(f'Unknown normalizer: {name}')


def atleast_2d(x):
    if x.ndim < 2:
        x = x[:, None]
    return x
