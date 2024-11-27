def dataframe_to_parquet(df, file_output):
    """
    Salva um dataframe pandas no formato Parquet.

    Parâmetros:
    -----------
    df : dask.DataFrame
        O dataframe a ser salvo.
    file_name : str
        O caminho do arquivo de saída (deve terminar em .parquet).
    compression : str, opcional
        Método de compressão a ser usado no Parquet (padrão: 'snappy').
        Outras opções incluem 'gzip', 'brotli', etc.

    Retorno:
    --------
    None
    """
    try:
        name_function = lambda x: f"data-{x}.parquet"  # noqa: E731
        df.to_parquet(file_output, name_function=name_function)
        print(f"Dataframe salvo com sucesso como Parquet: {file_output}")
    except ImportError:
        print("Erro: Certifique-se de que a biblioteca 'pyarrow' ou 'fastparquet' está instalada.")
    except Exception as e:
        print(f"Erro ao salvar o dataframe como Parquet: {e}")

