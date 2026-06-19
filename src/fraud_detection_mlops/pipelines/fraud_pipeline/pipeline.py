"""
This is a boilerplate pipeline 'fraud_pipeline'
generated using Kedro 1.4.0
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_data, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["raw_data", "params:test_size", "params:random_state"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="preprocess_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:random_state"],
                outputs="trained_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["trained_model", "X_test", "y_test"],
                outputs="evaluation_metrics",
                name="evaluate_model_node",
            ),
        ]
    )