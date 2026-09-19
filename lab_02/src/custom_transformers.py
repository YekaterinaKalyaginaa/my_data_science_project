from sklearn.base import BaseEstimator, TransformerMixin

class MissingValueAdder(BaseEstimator, TransformerMixin):
    """
    Кастомный трансформер: добавляет колонку 'missing_count'
    с количеством пропусков в каждой строке.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['missing_count'] = X_copy.isnull().sum(axis=1)
        return X_copy