from src.data import get_dataset, get_target, train_test_split
from src.preprocessing import preprocess


def main():
    data = get_dataset()
    target = get_target(data)
    data, target = preprocess(data, target)

    train_data, test_data = train_test_split(data)

    train_features = ...  # TO IMPLEMENT
    test_features = ...  # TO IMPLEMENT


if __name__ == '__main__':
    main()
