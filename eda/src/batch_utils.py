def iter_dataframe_batches(df, batch_size):
    if batch_size is None or batch_size <= 0:
        yield df
        return

    for start in range(0, len(df), batch_size):
        yield df.iloc[start:start + batch_size]