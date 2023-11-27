import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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


class PCACoordinate:
    """2 components PCA built from coordinates."""
    def fit(self, longitude, latitude):
        n_components = 2

        coordinates = pd.concat([longitude, latitude], axis=1)
        self.pca_ = PCA(n_components=n_components).fit(coordinates)

        return self

    def transform(self, longitude, latitude):
        coordinates = pd.concat([longitude, latitude], axis=1)
        return self.pca_.transform(coordinates)


PCA_COORDINATES_DEPENDENCIES = {
    "pca_pickup": ["pickup_lon", "pickup_lat"],
    "pca_dropoff": ["dropoff_lon", "dropoff_lat"],
}
for name, depends in PCA_COORDINATES_DEPENDENCIES.items():
    registry.register(PCACoordinate, name=name, depends=depends)
