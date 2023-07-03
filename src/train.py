import logging
from spice import Generator

from src.data import get_train_dataset, get_target, train_test_split
from src.features.registry import registry
from src.preprocessing import preprocess
from src.model import get_model, evaluate
from src.resources import get_resources

logger = logging.getLogger(__name__)


def main():
    data = get_train_dataset()
    target = get_target(data)
    data, target = preprocess(data, target)

    train_data, test_data = train_test_split(data)
    train_target = target.loc[train_data.index]
    test_target = target.loc[test_data.index]

    feature_generator = Generator(
        registry,
        resources=get_resources(),
        features=[
            "cyclical_pickup_hour",
            "quantile_bin_hour",
            "is_raining",
            "euclidean_distance",
            "manhattan_distance",
            "pickup_cluster",
            "dropoff_cluster"
        ]
    )
    train_features = feature_generator.fit_transform(train_data)
    test_features = feature_generator.transform(test_data)

    logger.info("Training model...")
    model = get_model().fit(train_features, train_target)

    metrics = evaluate(model, features=test_features, target=test_target)
    logger.info(f"Model metrics are:\n{metrics}")


if __name__ == '__main__':
    main()
