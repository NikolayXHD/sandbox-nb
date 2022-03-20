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
import pandas as pd

df = pd.DataFrame(
    {
        'i_0': [0, 0, 1, 1],
        'i_1': [0, 1, 0, 1],
        'X_0': [0.1, 0.1, 1.1, 1.1],
        'X_1': [0.1, 1.1, 0.1, 1.1],
    }
)
num_features = 2
df

# %%
df_agg = df.groupby(['i_0', 'i_1']).agg(
    {'X_0': 'mean', 'X_1': 'mean', 'i_0': 'first', 'i_1': 'first'}
)
df_agg

# %%
(*df_agg.iloc[:, 2:].max(axis=0), 2)

# %%
index = np.zeros((*(df_agg.iloc[:, 2:].max(axis=0) + 1), 2))
index

# %%
index[df_agg['i_0'], df_agg['i_1']]

# %%
index[tuple(df_agg.iloc[:, num_features:].T.values)] = df_agg.iloc[
    :, :num_features
].values
index[tuple(df_agg.iloc[:, num_features:].T.values)]

# %%
df_agg.iloc[:, :num_features].values
