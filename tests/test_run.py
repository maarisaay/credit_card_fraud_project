from pathlib import Path

from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project


def test_project_registers_real_fraud_pipeline():
    """Smoke-test the Kedro project wiring without running model training."""
    bootstrap_project(Path.cwd())

    registered_pipelines = pipelines
    fraud_pipeline = registered_pipelines["fraud_pipeline"]

    node_names = {node.name for node in fraud_pipeline.nodes}

    assert "fraud_pipeline" in registered_pipelines
    assert registered_pipelines["__default__"].nodes == fraud_pipeline.nodes
    assert node_names == {
        "preprocess_data_node",
        "engineer_features_node",
        "scale_features_node",
        "select_features_node",
        "tune_hyperparameters_node",
        "train_model_node",
        "evaluate_model_node",
        "prepare_serving_artifacts_node",
    }

    preprocessing_node = fraud_pipeline.only_nodes("preprocess_data_node").nodes[0]
    training_node = fraud_pipeline.only_nodes("train_model_node").nodes[0]

    assert preprocessing_node.inputs == ["raw_data", "params:test_size", "params:random_state"]
    assert training_node.inputs == ["X_train", "y_train", "best_params", "params:random_state"]
    assert training_node.outputs == ["trained_model"]
    assert "raw_data" in fraud_pipeline.inputs()
    assert "evaluation_metrics" in fraud_pipeline.outputs()
