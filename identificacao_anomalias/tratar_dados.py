import pandas as pd
import os

def load_and_clean_data(file_path: str, clean_function):
    df = pd.read_csv(file_path)
    return clean_function(df)

def clean_unidade_consumidora(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["id_endereco"])
    df["id_endereco"] = df["id_endereco"].astype(int)
    return df

def clean_usina(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(subset=["last_plant_history_id"])
    df["last_plant_history_id"] = df["last_plant_history_id"].astype(int)
    return df

def clean_usina_historico(df: pd.DataFrame) -> pd.DataFrame:
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df = df.drop(columns=['performance_type_enum'])
    return df

def clean_endereco(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(subset=["id_cidade"])
    df["id_cidade"] = df["id_cidade"].astype(int)
    df["id"] = df["id"].astype(int)
    return df

def clean_cidade(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(subset=["id_estado"])
    df["id_estado"] = df["id_estado"].astype(int)
    df["id"] = df["id"].astype(int)
    df = df.drop(columns=["created_at"])
    return df

def merge_data(usina, unidade_consumidora, endereco, cidade):
    usina = usina.merge(
        unidade_consumidora,
        left_on="unidade_consumidora_id",
        right_on="id",
        suffixes=("", "_unidade_consumidora")
    ).dropna(subset=["id_unidade_consumidora"])

    usina = usina.merge(
        endereco,
        left_on="id_endereco",
        right_on="id",
        suffixes=("", "_endereco")
    ).dropna(subset=["id_endereco"])

    usina = usina.merge(
        cidade,
        left_on="id_cidade",
        right_on="id",
        suffixes=("", "_cidade")
    ).dropna(subset=["id_cidade"])

    columns_to_drop = [
        "id_unidade_consumidora", "id_endereco", "id_cidade", 
        "id_endereco_endereco", "id_cidade_cidade"
    ]
    usina = usina.drop(columns=[col for col in columns_to_drop if col in usina.columns])
    return usina

def filter_and_merge_usina_historico(usina, usina_historico, cidade_nome: str) -> pd.DataFrame:
    usinas_filtradas = usina[usina['cidade_nome'] == cidade_nome]

    ids_usinas_filtradas = usinas_filtradas['id']
    

    historico_usinas_filtradas = usina_historico[usina_historico['plant_id'].isin(ids_usinas_filtradas)]
    historico_usinas_filtradas = historico_usinas_filtradas.dropna(subset=['plant_id'])
    historico_usinas_sorted = historico_usinas_filtradas.sort_values(by=['plant_id', 'start_date'])

    historico_usina_atual = historico_usinas_sorted.groupby('plant_id').last().reset_index()


    historico_usina_atual = historico_usina_atual.rename(columns={
        'power': 'current_power',
        'start_date': 'last_update'
    })

    usina_com_potencia_atual = usinas_filtradas.merge(
        historico_usina_atual[['plant_id', 'current_power', 'last_update']],
        left_on='id',
        right_on='plant_id',
        how='left'
    )


    return usina_com_potencia_atual

def main(input_dir: str, output_dir: str, cidade_nome):
    os.makedirs(output_dir, exist_ok=True)

    usina = load_and_clean_data(f"{input_dir}/usina.csv", clean_usina)
    unidade_consumidora = load_and_clean_data(f"{input_dir}/unidade_consumidora.csv", clean_unidade_consumidora)
    endereco = load_and_clean_data(f"{input_dir}/endereco.csv", clean_endereco)
    cidade = load_and_clean_data(f"{input_dir}/cidade.csv", clean_cidade)
    usina_historico = load_and_clean_data(f"{input_dir}/usina_historico.csv", clean_usina_historico)
    usina_merged = merge_data(usina, unidade_consumidora, endereco, cidade)
    usina_filtered = filter_and_merge_usina_historico(usina_merged, usina_historico, cidade_nome)
    file_path = f"{output_dir}/usina_{cidade_nome.lower()}_merged.csv"
    usina_filtered.to_csv(file_path, index=False)
    print("Pipeline concluída! Arquivo salvo em:", file_path)


if __name__ == "__main__":
    OUTPUT_DIR = "cleaned"
    CIDADE = "Mossoró"
    INPUT_DIR = '_data'
    main(INPUT_DIR, OUTPUT_DIR, CIDADE)