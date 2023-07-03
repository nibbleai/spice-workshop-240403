import httpx
import pandas as pd
from sklearn.model_selection import train_test_split as _train_test_split

from src.config import config
from src.directories import directories
from src.schemas import TaxiColumn, WeatherColumn

TAXI_DATA_FILENAME = 'nyc-taxi-2015.csv'
WEATHER_DATA_FILENAME = 'nyc-weather-2015.csv'

DATASET_URLS = {
    TAXI_DATA_FILENAME: 'https://nibble-datasets.s3.eu-west-1.amazonaws.com/nyc-taxi/nyc-taxi-2015.csv',  # noqa: E501
    WEATHER_DATA_FILENAME: 'https://nibble-datasets.s3.eu-west-1.amazonaws.com/nyc-taxi/nyc-weather-2015.csv'  # noqa: E501
}


def get_dataset():
    filepath = directories.taxi_data_dir / TAXI_DATA_FILENAME
    if not filepath.exists():
        download_data(filepath)

    return pd.read_csv(
        filepath,
        parse_dates=[TaxiColumn.PICKUP_TIME, TaxiColumn.DROPOFF_TIME]
    )


def get_weather_data():
    filepath = directories.weather_data_dir / WEATHER_DATA_FILENAME
    if not filepath.exists():
        download_data(filepath)

    weather = pd.read_csv(filepath, parse_dates=[WeatherColumn.DATE])

    return weather.assign(**{
        WeatherColumn.DATE: lambda x: x[WeatherColumn.DATE].dt.date
    })


def train_test_split(data):
    return _train_test_split(
        data.sort_values(TaxiColumn.PICKUP_TIME),
        test_size=config.test_size,
        shuffle=False
    )


def get_target(data):
    """Target is the total duration of trip, in seconds"""
    target = (
        data[TaxiColumn.DROPOFF_TIME] - data[TaxiColumn.PICKUP_TIME]
    ).dt.seconds
    return target.rename(config.target)


def download_data(filepath):
    filepath.parent.mkdir(exist_ok=True, parents=True)
    r = httpx.get(DATASET_URLS[filepath.name])
    with open(filepath, 'w') as fp:
        fp.write(r.text)
