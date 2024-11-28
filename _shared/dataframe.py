import dask.dataframe as dd
import pandas as pd

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

def _remove_milliseconds(date_str):
    if pd.isna(date_str):
        return date_str
    if '.' in date_str:
        date_str = date_str.split('.')[0]
    return date_str


def clean_date_columns(df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    """
    Limpa e formata colunas de tipo date ou datetime em um DataFrame.

    Args:
        df (pd.DataFrame): O DataFrame a ser processado.
        date_columns (list): Lista de colunas a serem tratadas como datas.
        remove_milliseconds_func (function): Função para remover milissegundos.

    Returns:
        pd.DataFrame: O DataFrame com as colunas de datas tratadas.
    """
    for col in date_columns:
        # Manter os dados originais em uma nova coluna
        raw_col = f"{col}_raw"
        cleaned_col = f"{col}_cleaned"
        df[raw_col] = df[col]
        
        # Converter para datetime, tratando erros
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Identificar valores NaT e aplicar a função remove_milliseconds
        mask_nat = df[col].isna()
        if mask_nat.any():
            df.loc[mask_nat, cleaned_col] = df.loc[mask_nat, raw_col].apply(_remove_milliseconds)
            df.loc[mask_nat, col] = pd.to_datetime(df.loc[mask_nat, cleaned_col], errors='coerce')
    
    return df