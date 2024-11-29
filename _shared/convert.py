def dataframe_to_parquet(df, file_output):
    try:
        name_function = lambda x: f"data-{x}.parquet" 
        df.to_parquet(file_output, name_function=name_function)
        print(f"Dataframe salvo com sucesso como Parquet: {file_output}")
    except ImportError:
        print("Erro: Certifique-se de que a biblioteca 'pyarrow' ou 'fastparquet' está instalada.")
    except Exception as e:
        print(f"Erro ao salvar o dataframe como Parquet: {e}")

