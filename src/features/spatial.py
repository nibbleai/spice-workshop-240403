import numpy as np
from sklearn.preprocessing import StandardScaler

from src.features.registry import registry


@registry.register(
    name="euclidean_distance",
    depends=["pickup_lon", "pickup_lat", "dropoff_lon", "dropoff_lat"]
)
def euclidean_distance(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat):
    """Euclidean distance between pickup & dropoff positions."""
    return np.sqrt(
        abs(pickup_lon - dropoff_lon) ** 2 + abs(pickup_lat - dropoff_lat) ** 2
    )


@registry.register(
    name="manhattan_distance",
    depends=["pickup_lon", "pickup_lat", "dropoff_lon", "dropoff_lat"]
)
def manhattan_distance(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat):
    """Manhattan distance between pickup & dropoff positions."""
    return abs(pickup_lon - dropoff_lon) + abs(pickup_lat - dropoff_lat)


class ScaledDistance:
    def fit(self, distance):
        self.scaler_ = StandardScaler().fit(distance.values.reshape(-1, 1))
        return self

    def transform(self, distance):
        return self.scaler_.transform(distance.values.reshape(-1, 1))


for distance in ('euclidean_distance', 'manhattan_distance'):
    registry.register(
        ScaledDistance, name=f"[scaled]_{distance}", depends=[distance]
    )
