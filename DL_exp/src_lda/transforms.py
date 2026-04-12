
import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class IdentityTransform:
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x

    def fit_transform(self, x, y=None):
        return x


@dataclass
class LDAProjector:
    kind: str = 'lda'

    def __post_init__(self):
        if self.kind == 'reglda':
            self.model = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
        elif self.kind == 'lda':
            self.model = LinearDiscriminantAnalysis(solver='svd')
        else:
            raise ValueError(f'Unknown projector kind: {self.kind}')

    def fit(self, features, labels):
        labels = np.asarray(labels)
        if len(np.unique(labels)) < 2:
            self.model = IdentityTransform().fit(features, labels)
            return self
        self.model.fit(features, labels)
        return self

    def transform(self, features):
        return self.model.transform(features)

    def fit_transform(self, features, labels):
        return self.fit(features, labels).transform(features)

    def dumps(self) -> bytes:
        return pickle.dumps(self.model)

    @classmethod
    def loads(cls, payload: bytes) -> 'LDAProjector':
        obj = cls()
        obj.model = pickle.loads(payload)
        return obj
