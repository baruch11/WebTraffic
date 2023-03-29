"""Main script to train and make submission."""
import argparse
import pandas as pd
import os
import logging


from webtraffic.inout import (load_data, get_training_datasets, Submission,
                              get_root_dir)
from webtraffic.models.median_model import median_model
from webtraffic.models.dense_model import dense_model
from webtraffic.models.rnn_model import rnn_model
from webtraffic.webtraffic_utils import smape_np


MODELS = [rnn_model, dense_model, median_model]

parser = argparse.ArgumentParser(
    description='Main script for training and submitting')
parser.add_argument('--nsamp', '-n', type=int, default=None,
                    help="Nb samples to work on"
                    "(all samples if None (default))")
parser.add_argument('--epochs', '-e', type=int, default=100,
                    help="Max epochs"
                    "(all samples if None (default))")
parser.add_argument('--model', '-m', type=str, default=MODELS[0].__name__,
                    help=f"Model to train in {[ii.__name__ for ii in MODELS]}")
parser.add_argument('--val', '-v', action='store_true',
                    help='Compute validation score')

args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)


print("loading traffic dataset.")
traffic = load_data(args.nsamp)
print(f"Loaded traffic of {len(traffic)} pages")

X_train, X_test, Y_train, Y_test = get_training_datasets(traffic)

for modname in MODELS:
    if modname.__name__ == args.model:
        model = modname(output_len=Y_train.shape[1],
                        epochs=args.epochs)

print("Fitting the model")

val_data = None
if args.val:
    val_data = (X_test, Y_test)
model.fit(X_train, Y_train, val_data=val_data)

print("Model evaluation\n")
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print(pd.Series(index=["train", "test"],
                data=[smape_np(train_pred, Y_train.values),
                      smape_np(test_pred, Y_test.values)]))

# blind retraining on the last datas ?


# exporting the predictions
print("Predicting for submission")
final_preds = model.predict(traffic.iloc[:, -X_train.shape[1]:])

keys_path = os.path.join(get_root_dir(),
                         "data/web-traffic-time-series-forecasting",
                         "key_2.csv.zip")

forecast_sub = Submission(keys_path,
                          final_preds,
                          traffic.index)
forecast_sub.to_csv("subm_med.csv")
