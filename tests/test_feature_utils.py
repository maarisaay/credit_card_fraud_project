import pandas as pd

from fraud_detection_mlops.feature_utils import add_engineered_features


def test_add_engineered_features_creates_expected_columns():
    data = pd.DataFrame(
        {
            "Time": [0, 3600, 7200],
            "Amount": [0.0, 10.0, 100.0],
            "V17": [1.0, 2.0, 3.0],
            "V14": [1.0, 2.0, 3.0],
            "V12": [1.0, 2.0, 3.0],
            "V10": [1.0, 2.0, 3.0],
        }
    )

    result = add_engineered_features(data)

    expected_columns = {
        "hour_of_day",
        "amount_log",
        "amount_x_V17",
        "amount_x_V14",
        "amount_x_V12",
        "amount_x_V10",
    }

    assert expected_columns.issubset(result.columns)


def test_add_engineered_features_calculates_hour_of_day():
    data = pd.DataFrame(
        {
            "Time": [0, 3600, 7200],
            "Amount": [10.0, 10.0, 10.0],
            "V17": [1.0, 1.0, 1.0],
            "V14": [1.0, 1.0, 1.0],
            "V12": [1.0, 1.0, 1.0],
            "V10": [1.0, 1.0, 1.0],
        }
    )

    result = add_engineered_features(data)

    assert result["hour_of_day"].tolist() == [0, 1, 2]


def test_add_engineered_features_calculates_interactions():
    data = pd.DataFrame(
        {
            "Time": [0],
            "Amount": [10.0],
            "V17": [2.0],
            "V14": [3.0],
            "V12": [4.0],
            "V10": [5.0],
        }
    )

    result = add_engineered_features(data)

    assert result.loc[0, "amount_x_V17"] == 20.0
    assert result.loc[0, "amount_x_V14"] == 30.0
    assert result.loc[0, "amount_x_V12"] == 40.0
    assert result.loc[0, "amount_x_V10"] == 50.0