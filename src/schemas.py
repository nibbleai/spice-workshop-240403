"""This modules defines schemas of datasets used in the demo."""

from spice.utils import StrEnum


class TaxiColumn(StrEnum):
    """Column names in the taxi dataset"""

    PICKUP_TIME = 'pickup_datetime'
    DROPOFF_TIME = 'dropoff_datetime'
    PICKUP_LON = 'pickup_longitude'
    PICKUP_LAT = 'pickup_latitude'
    DROPOFF_LON = 'dropoff_longitude'
    DROPOFF_LAT = 'dropoff_latitude'


class WeatherColumn(StrEnum):
    """Column names in the weather dataset"""

    DATE = 'DATE'
    PRECIPITATIONS = 'PRCP'
    MAX_TEMPERATURE = 'TMAX'
    MIN_TEMPERATURE = 'TMIN'
    AVERAGE_WIND_SPEED = 'AWND'
