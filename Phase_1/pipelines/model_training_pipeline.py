from src.inference import (
    fetch_days_data,
    load_metrics_from_registry,
    load_model_from_registry,
    get_hopsworks_project
)
from hsml.model_schema import ModelSchema
from hsml.schema import Schema


from src.pipeline_utils import get_pipeline
from src.data_utils import transform_ts_data_info_features_and_target
import src.config as config
import joblib
from sklearn.metrics import mean_absolute_error

print(f"Fetching data from group store..")
ts_data = fetch_days_data(180)

print(f"Transforming to ts_data ...")

features,targets = transform_ts_data_info_features_and_target(
    ts_data, window_size = 24*28, step_size = 23
)
pipeline = get_pipeline()
print(f"Training model ...")
pipeline.fit(features, targets)

predictions = pipeline.predict(features)

test_mae = mean_absolute_error(targets, predictions)
print(f"The new MAE is {test_mae:.4f}")


metric = load_metrics_from_registry()

print(f"The previous MAE is {metric['test_mae']:.4f}")

if test_mae < metric.get("test_mae"):
    print(f"Registering new model")
    model_path = config.MODELS_DIR / "lgbmodel.pkl"
    joblib.dump(pipeline, model_path)
    input_schema = Schema(features)
    output_schema = Schema(targets)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()
    modelv2 = model_registry.sklearn.create_model(
        name = "taxi_demand_predictor_next_hour",
        metrics = {"test_mae": test_mae},
        description="LightGBM regressor V2",
        input_example=features.sample(),
        model_schema = model_schema,
    )
    modelv2.save(model_path)
else:
    print(f"Skipping model registration as new model is not better!")

