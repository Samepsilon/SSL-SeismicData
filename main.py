import mlflow
mlflow.set_tracking_uri("sqlite:///MLflow.db")
mlflow.set_experiment("SSL_FineTuning_Model_Comparison")

if __name__ == '__main__':

    # Print connection information
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Active Experiment: {mlflow.get_experiment_by_name('my-first-experiment')}")

    # Test logging
    with mlflow.start_run():
        mlflow.log_param("test_param", "test_value")
        print("✓ Successfully connected to MLflow!")