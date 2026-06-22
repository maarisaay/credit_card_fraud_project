"""
This is a boilerplate pipeline 'fraud_pipeline'
generated using Kedro 1.4.0
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    preprocess_data,
    engineer_features,
    scale_features,
    select_features,
    tune_hyperparameters,
    train_model,
    evaluate_model,
    prepare_serving_artifacts,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["raw_data", "params:test_size", "params:random_state"],
                outputs=["X_train_raw", "X_test_raw", "y_train", "y_test"],
                name="preprocess_data_node",
            ),
            node(
                func=engineer_features,
                inputs=["X_train_raw", "X_test_raw"],
                outputs=["X_train_fe", "X_test_fe"],
                name="engineer_features_node",
            ),
            node(
                func=scale_features,
                inputs=["X_train_fe", "X_test_fe"],
                outputs=["X_train_scaled", "X_test_scaled", "fitted_scaler"],
                name="scale_features_node",
            ),
            node(
                func=select_features,
                inputs=["X_train_scaled", "X_test_scaled", "params:selected_features"],
                outputs=["X_train", "X_test"],
                name="select_features_node",
            ),
            node(
                func=tune_hyperparameters,
                inputs=["X_train", "y_train", "params:n_trials", "params:random_state"],
                outputs="best_params",
                name="tune_hyperparameters_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "best_params", "params:random_state"],
                outputs="trained_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["trained_model", "X_test", "y_test"],
                outputs="evaluation_metrics",
                name="evaluate_model_node",
            ),
            node(
                func=prepare_serving_artifacts,
                inputs=["raw_data", "params:selected_features"],
                outputs="serving_metadata",
                name="prepare_serving_artifacts_node",
            ),
        ]
    )