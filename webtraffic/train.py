"""Main script to train and make submission."""
import argparse
import pandas as pd

from webtraffic.inout import load_data, get_training_datasets
from webtraffic.models.median_model import median_model
from webtraffic.webtraffic_utils import smape_np


parser = argparse.ArgumentParser(
    description='Main script for training and submitting')
parser.add_argument('--nsamp', '-n', type=int, default=None,
                    help="Nb samples to work on"
                    "(all samples if None (default))")

args = parser.parse_args()

print("loading traffic dataset.")
traffic = load_data(args.nsamp)
print(f"Loaded traffic of {len(traffic)} pages")

X_train, X_test, Y_train, Y_test = get_training_datasets(traffic)

model = median_model()

print("Fitting the model")
model.fit(X_train, Y_train)

print("Model evaluation\n")
train_pred = model.predict(X_train.values)
test_pred = model.predict(X_test.values)

print(pd.Series(index=["train", "test"],
                data=[smape_np(train_pred, Y_train.values),
                      smape_np(test_pred, Y_test.values)]))
