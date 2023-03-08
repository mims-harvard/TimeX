from functools import lru_cache, partial
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from gluonts.time_feature import (
    get_lags_for_frequency,
    get_seasonality,
    time_features_from_frequency_str,
)
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
import datasets


from data import create_train_dataloader, create_test_dataloader
from metrics import mase_metric, smape_metric


@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)


def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch



def main():
    DATA_PATH = Path("datasets/tourism_monthly")
    freq = "1M"
    prediction_length = 24

    if not DATA_PATH.exists():
        dataset = datasets.load_dataset("monash_tsf", "tourism_monthly")
        dataset.save_to_disk(DATA_PATH)
    else:
        dataset = datasets.load_from_disk(DATA_PATH)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_dataset.set_transform(partial(transform_start_field, freq=freq))
    test_dataset.set_transform(partial(transform_start_field, freq=freq))

    time_features = time_features_from_frequency_str(freq)
    lags_sequence = get_lags_for_frequency(freq)


    config = TimeSeriesTransformerConfig(
        prediction_length=prediction_length,
        context_length=prediction_length * 3,  # context length
        lags_sequence=lags_sequence,
        num_time_features=len(time_features)
        + 1,  # we'll add 2 time features ("month of year" and "age", see further)
        num_static_categorical_features=1,  # we have a single static categorical feature, namely time series ID
        cardinality=[len(train_dataset)],  # it has 366 possible values
        embedding_dimension=[
            2
        ],  # the model will learn an embedding of size 2 for each of the 366 possible values
        encoder_layers=4,
        decoder_layers=4,
    )


    train_dataloader = create_train_dataloader(
        config=config, 
        freq=freq, 
        data=train_dataset, 
        batch_size=256, 
        num_batches_per_epoch=100,
    )

    test_dataloader = create_test_dataloader(
        config=config, 
        freq=freq, 
        data=test_dataset,
        batch_size=64,
    )

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TimeSeriesTransformerForPrediction(config)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(40):
        print("epoch:", epoch)
        model.train()
        for batch in train_dataloader:
            outputs = model(
                static_categorical_features=batch["static_categorical_features"].to(device),
                static_real_features=batch["static_real_features"].to(device),
                past_time_features=batch["past_time_features"].to(device),
                past_values=batch["past_values"].to(device),
                future_time_features=batch["future_time_features"].to(device),
                future_values=batch["future_values"].to(device),
                past_observed_mask=batch["past_observed_mask"].to(device),
                future_observed_mask=batch["future_observed_mask"].to(device),
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    forecasts = []

    for batch in test_dataloader:
        outputs = model.generate(
            static_categorical_features=batch["static_categorical_features"].to(device),
            static_real_features=batch["static_real_features"].to(device),
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
        )
        forecasts.append(outputs.sequences.cpu().numpy())

    forecasts = np.vstack(forecasts)

    forecast_median = np.median(forecasts, 1)

    mase_metrics = []
    smape_metrics = []
    for item_id, ts in enumerate(test_dataset):
        training_data = ts["target"][:-prediction_length]
        ground_truth = ts["target"][-prediction_length:]
        mase = mase_metric(
            predictions=forecast_median[item_id], 
            references=np.array(ground_truth), 
            training=np.array(training_data), 
            periodicity=get_seasonality(freq))
        mase_metrics.append(mase["mase"])
        
        smape = smape_metric(
            predictions=forecast_median[item_id], 
            references=np.array(ground_truth), 
        )
        smape_metrics.append(smape["smape"])

    print(f"MASE: {np.mean(mase_metrics)}")
    print(f"sMAPE: {np.mean(smape_metrics)}")


if __name__ == "__main__":
    main()