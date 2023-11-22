import logging

from src.data import get_live_dataset
from src.features import base, spatial, temporal, weather
from src.io_ import load_generator, load_model
from src.resources import get_resources


def main():
    live_data = get_live_dataset()

    logging.info("Loading features generator...")
    generator = load_generator()
    generator.resources = get_resources()

    live_features = generator.transform(live_data).to_pandas()

    logging.info("Loading model...")
    model = load_model()

    logging.info("Making predictions")
    model.predict(live_features)


if __name__ == '__main__':
    main()
