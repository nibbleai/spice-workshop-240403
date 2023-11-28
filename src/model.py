from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error

__all__ = ['get_model', 'evaluate']


def get_model():
    return LinearRegression()


def evaluate(model, *, features, target):
    predicted = model.predict(features)
    return {'mean_squared_log_error': mean_squared_log_error(target, predicted)}
