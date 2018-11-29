from __future__ import print_function
from sklearn.linear_model import ElasticNet
from sys import argv
import mlflow
import mlflow.sklearn

def main():
    with mlflow.start_run():
        alpha    = float(argv[1]) if len(argv) > 1 else 0
        l1_ratio = float(argv[2]) if len(argv) > 2 else 0
        print("Running with alpha=%.2f, l1_ratio=%.2f" % (alpha, l1_ratio))

        (x_train, y_train) = load_data("train.parquet")
        (x_test, y_test) = load_data("test.parquet")

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        (mae, rmse, r2) = eval_metrics(y_test, y_pred)

        print("MAE", mae)
        print("RMSE", rmse)
        print("R2", r2)


def load_data(parquet_file):
    import pandas as pd
    df = pd.read_parquet(parquet_file)
    y = df[["price"]]
    x = df.drop(["price"], axis=1)
    return (x, y)


def eval_metrics(actual, pred):
    from numpy import sqrt
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(actual, pred)
    rmse = sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    return (mae, rmse, r2)


if __name__ == "__main__":
    main()
