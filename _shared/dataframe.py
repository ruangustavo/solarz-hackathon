import dask.dataframe as dd

def load_data(file_path: str) -> dd.DataFrame:
    """
    Load a CSV file into a Dask DataFrame.
    :param file_path: Path to the CSV file.
    :return: Dask DataFrame.
    """
    return dd.read_csv(file_path)

def clean_and_merge(left_df: dd.DataFrame, right_df: dd.DataFrame, left_on: str, right_on: str, suffix: str = "") -> dd.DataFrame:
    """
    Merge two DataFrames and clean rows where the right join key is missing.
    :param left_df: Left DataFrame.
    :param right_df: Right DataFrame.
    :param left_on: Key in the left DataFrame for merging.
    :param right_on: Key in the right DataFrame for merging.
    :param suffix: Suffix to add to overlapping columns.
    :return: Cleaned merged DataFrame.
    """
    merged = left_df.merge(
        right_df,
        left_on=left_on,
        right_on=right_on,
        how="left",
        suffixes=("", suffix),
    )
    # Remove rows where the merged key in the right DataFrame is missing
    return merged[merged[right_on].notnull()]

def rename_columns_with_prefix(df: dd.DataFrame, prefix: str) -> dd.DataFrame:
    """
    Rename columns of a DataFrame by adding a prefix.
    :param df: Input DataFrame.
    :param prefix: Prefix to add.
    :return: DataFrame with renamed columns.
    """
    return df.rename(columns={col: f"{prefix}{col}" for col in df.columns})

def drop_columns(df: dd.DataFrame, columns: list) -> dd.DataFrame:
    """
    Drop columns from a DataFrame.
    :param df: Input DataFrame.
    :param columns: List of columns to drop.
    :return: DataFrame with dropped columns.
    """
    return df.drop(columns=columns)