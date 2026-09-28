import numpy as np
import tsaug


class Jittering:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, x):
        return x + np.random.normal(loc=0., scale=self.std, size=x.shape)
        # return x + torch.randn(x.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Scaling:
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        # self.mean = mean

    def __call__(self, x):
        n_scale = np.random.normal(loc=1, scale=self.sigma, size=(x.shape[0], x.shape[1]))
        # n_scale = torch.randn(x.size()) * self.std + self.mean
        return x * n_scale


class Flipping:
    def __init__(self, axis=1):
        self.axis = axis  # on the direction of time

    def __call__(self, x):
        return np.flip(x, axis=self.axis).copy()


class CustomAugmentation:
    def __init__(self, n_speed_change, max_speed_ratio, scale, p):
        self.n_speed_change = n_speed_change
        self.max_speed_ratio = max_speed_ratio
        self.scale = scale
        self.p = p

        self.custom_augmenter = (tsaug.TimeWarp(n_speed_change=6, max_speed_ratio=3)
                                 + tsaug.AddNoise(scale=0.01)
                                 + tsaug.Dropout(p=0.1, size=(5, 20), fill=float(0))
                                 )

    def __call__(self, x):
        x_aug = self.custom_augmenter.augment(x)
        return x_aug
