from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

__all__ = ['evaluate', 'get_model']


def get_model():
    return LinearRegression()


def evaluate(model, *, features, target):
    predicted = model.predict(features)
    return {'mean_absolute_error': mean_absolute_error(target, predicted)}
