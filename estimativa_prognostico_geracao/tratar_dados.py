import pandas as pd
import pyarrow.parquet as pq

df_usinas = pd.read_csv("../_data/../_data/usina.csv")
df_endereco = pd.read_csv("../_data/../_data/endereco.csv")
df_cidade = pd.read_csv("../_data/../_data/cidade.csv")
df_unidade_consumidora = pd.read_csv("../_data/../_data/unidade_consumidora.csv")

df_usinas_com_unidade = df_usinas.merge(
    df_unidade_consumidora,
    left_on="unidade_consumidora_id",
    right_on="id",
    how="left",
    suffixes=("_usina", "_unidade"),
)

df_usinas_com_endereco = df_usinas_com_unidade.merge(
    df_endereco,
    left_on="id_endereco",
    right_on="id",
    how="left",
    suffixes=("", "_endereco"),
)

df_usinas_com_cidade = df_usinas_com_endereco.merge(
    df_cidade, left_on="id_cidade", right_on="id", how="left", suffixes=("", "_cidade")
)

df_usinas_com_cidade = df_usinas_com_cidade.drop(columns=["id_endereco", "id_cidade"])

usinas_mossoro = df_usinas_com_cidade[df_usinas_com_cidade["nome"] == "Mossoró"]
usinas_mossoro = usinas_mossoro[["id_usina", "potencia"]]
usinas_mossoro.to_csv("../_data/usinas_mossoro.csv", index=False)

df_usinas_mossoro = pd.read_csv("../_data/usinas_mossoro.csv")

usinas_mossoro_ids = set(df_usinas_mossoro["id_usina"])

table = pq.read_table(
    "../_data/geracao.parquet",
    columns=["id_usina", "data", "quantidade"],
    filters=[("id_usina", "in", usinas_mossoro_ids)],
)

df_geracao = table.to_pandas()

df_geracao = df_geracao.merge(
    df_usinas_mossoro, left_on="id_usina", right_on="id_usina"
)

df_geracao.to_csv("../_data/geracao_mossoro.csv", index=False)
