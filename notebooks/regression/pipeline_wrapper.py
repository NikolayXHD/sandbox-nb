from __future__ import annotations

from sklearn.pipeline import Pipeline


# noinspection PyPep8Naming
class PipelineWrapper:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def get_params(self, deep=True):
        return {'pipeline': self.pipeline}

    def fit(self, X, y, sample_weight=None, **fit_params):
        name, estimator = self.pipeline.steps[-1]
        if sample_weight is not None:
            fit_params[f'{name}__sample_weight'] = sample_weight
        self.pipeline = self.pipeline.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.pipeline.score(X, y, sample_weight)


__all__ = ['PipelineWrapper']
