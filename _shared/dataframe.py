import dask.dataframe as dd
import pandas as pd


def load_data(file_path: str) -> dd.DataFrame:
    return dd.read_csv(file_path)


def clean_and_merge(
    left_df: dd.DataFrame,
    right_df: dd.DataFrame,
    left_on: str,
    right_on: str,
    suffix: str = "",
) -> dd.DataFrame:

    merged = left_df.merge(
        right_df,
        left_on=left_on,
        right_on=right_on,
        how="left",
        suffixes=("", suffix),
    )
    return merged[merged[right_on].notnull()]


def rename_columns_with_prefix(df: dd.DataFrame, prefix: str) -> dd.DataFrame:
    return df.rename(columns={col: f"{prefix}{col}" for col in df.columns})


def drop_columns(df: dd.DataFrame, columns: list) -> dd.DataFrame:
    return df.drop(columns=columns)


def _remove_milliseconds(date_str):
    if pd.isna(date_str):
        return date_str
    if "." in date_str:
        date_str = date_str.split(".")[0]
    return date_str


def clean_date_columns(df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    for col in date_columns:
        raw_col = f"{col}_raw"
        cleaned_col = f"{col}_cleaned"
        df[raw_col] = df[col]

        df[col] = pd.to_datetime(df[col], errors="coerce")

        mask_nat = df[col].isna()
        if mask_nat.any():
            df.loc[mask_nat, cleaned_col] = df.loc[mask_nat, raw_col].apply(
                _remove_milliseconds
            )
            df.loc[mask_nat, col] = pd.to_datetime(
                df.loc[mask_nat, cleaned_col], errors="coerce"
            )
    return df
