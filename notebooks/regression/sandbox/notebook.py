# %%
from matplotlib import pyplot as plt
import numpy as np

from scipy.stats import logistic
from scipy.special import logit

scale = 0.10

fig, ax = plt.subplots(figsize=(10, 10))

x = np.linspace(-1, 1, num=200)
y = logistic.cdf(x, loc=0, scale=scale)
y_inv = scale * logit(x)
plt.plot(x, y, '.')
plt.plot(x, y_inv, 'r.')
plt.ylim(-1, 1)
plt.show()

# %%
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from notebooks.regression import pipeline_wrapper, k_neighbors

# plt.plot(
#     np.linspace(-1, 1, num=1000),
#     norm.pdf(np.linspace(-1, 1, num=1000), 0, 0.05),
# )
# plt.show()


def _w(d):
    return -(d ** 2) + 0.3 ** 2


reg = k_neighbors.RadiusNeighborsWeightedRegressor(
    radius=0.3, metric='manhattan', weights=_w  # 'uniform'  # _w,
)

# reg = k_neighbors.KNeighborsWeightedRegressor(
#     n_neighbors=10,
#     metric='manhattan',
#     weights=_w  # 'uniform'  # _w,
# )

x = np.linspace(-1, 1, num=40)
y = np.sin(x * 3)
w = 1 + 100 * x ** 2

X = x.reshape(-1, 1)
reg.fit(X, y, w)
y_pred = reg.predict(X)

plt.plot(x, y, '.')
plt.plot(x, y_pred, '-')
plt.show()

# %%
from notebooks.regression.k_neighbors import KNeighborsWeightedRegressor
from scipy.stats import norm

x = np.linspace(-1, 1, num=21)
X = x.reshape(-1, 1)
y = np.sign(X)

radius = 0.1

regressor = KNeighborsWeightedRegressor(
    n_neighbors=16,
    weights=lambda d: norm.pdf(d / radius),
    n_jobs=-1,
)

regressor.fit(X, y)

x_pred = np.linspace(-1, 1, num=81)
X_pred = x_pred.reshape(-1, 1)
y_pred = regressor.predict(X_pred)

plt.plot(x_pred, y_pred)
plt.show()
