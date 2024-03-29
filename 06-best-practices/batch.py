#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd

def main(year, month):
    input_file = f'C:\\Users\\svksh\\Documents\\GitHub\\mlops-zoomcamp\\04-deployment\\homework\\data\\yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    print("Data file : ", str(input_file))

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')

    print("-----Loading Model-----")
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)

    print("-----Done-----")


def read_data(filename, categorical):
    df = pd.read_parquet(filename)

    print("-----Data Read-----")
    return df

def prepare_data(read_df, categorical):
    
    read_df['duration'] = read_df.tpep_dropoff_datetime - read_df.tpep_pickup_datetime
    read_df['duration'] = read_df.duration.dt.total_seconds() / 60

    df = read_df[(read_df.duration >= 1) & (read_df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    print("-----Data Prepared-----")
    return df


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    print("-----Reading Arguments-----")
    print("Year : ", year)
    print("Month : ", month)
    main(year, month)
