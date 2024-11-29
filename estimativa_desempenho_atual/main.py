import dask.dataframe as dd
import numpy as np
import logging
import pandas as pd

from _shared.path import path

from tqdm.dask import TqdmCallback

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

if __name__ == "__main__":
    usinas_cols = ["id", "power", "cidade_nome"]
    usinas = pd.read_csv(path("usina_mossoró_potencia.csv"))

    geracao_cols = ["id_usina", "quantidade", "data"]
    geracao = pd.read_csv(path("geracao_mossoro.csv"))

    logging.info("Convertendo a coluna 'data' para datetime...")
    geracao["data"] = dd.to_datetime(
        geracao["data"], format="%Y-%m-%d", errors="coerce"
    )

    data_inicio = "2023-01-01"
    logging.info(
        f"Aplicando filtro para incluir somente dados de {data_inicio} até a data atual..."
    )

    geracao = geracao[geracao["data"] >= data_inicio]

    logging.info("Iniciando o merge dos dataframes...")
    with TqdmCallback(desc="compute") as progress:

        geracao = geracao.merge(
            usinas,
            left_on="id_usina",
            right_on="id",
            how="inner",
            suffixes=("", "_usina"),
        )

        logging.info("Criando médias historica de cada usina potência...")

        media_historica = geracao.groupby("id_usina")["quantidade"].mean()

        usinas = usinas.merge(
            media_historica,
            left_on="id",
            right_on="id_usina",
            how="left",
            suffixes=("", "_media"),
        )

        logging.info("Calculando a faixa de potência de cada usina...")

        max_power = usinas["power"].max()
        bins = np.arange(0, max_power + 5, 5)
        usinas["power_range"] = pd.cut(usinas["power"], bins=bins)

        logging.info(
            "Calculando as média e variancia esperada por faixa de potência e cidade..."
        )

        media_e_desvio_geracao_por_grupo = (
            usinas.groupby(["power_range"])["quantidade"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "media_esperada", "std": "std_esperada"})
            .reset_index()
        )

        media_e_desvio_geracao_por_grupo = media_e_desvio_geracao_por_grupo.dropna(
            subset=["media_esperada", "power_range"]
        )

        usinas = usinas.merge(
            media_e_desvio_geracao_por_grupo, on="power_range", how="left"
        )

        logging.info("Detecting outliers using the z-score method...")

        usinas["z_score"] = (usinas["quantidade"] - usinas["media_esperada"]) / usinas[
            "std_esperada"
        ]

        outlier_threshold = 3
        usinas["is_outlier"] = usinas["z_score"] > outlier_threshold
        usinas_outliers = usinas[usinas["is_outlier"]]

        logging.info(
            "Identificando usinas com valor total de geração abaixo do percentual de 80%..."
        )

        usinas["anomalous"] = usinas["quantidade"] < usinas["media_esperada"] * 0.8
        usinas_anomalas_com_outliers = usinas[usinas["anomalous"]]
        logging.info(
            f"Número de usinas anômalas detectadas: {len(usinas_anomalas_com_outliers)}"
        )

        usinas_cleaned = usinas[~usinas["is_outlier"]]

        media_geracao_por_grupo_cleaned = (
            usinas_cleaned.groupby(["power_range"])["quantidade"]
            .mean()
            .rename("media_geracao")
            .reset_index()
        )

        usinas_cleaned = usinas_cleaned.merge(
            media_geracao_por_grupo_cleaned,
            on="power_range",
            how="left",
        )

        usinas_cleaned["anomalous"] = (
            usinas_cleaned["quantidade"] < usinas_cleaned["media_geracao"] * 0.8
        )
        usinas_anomalas_cleaned = usinas_cleaned[usinas_cleaned["anomalous"]]

        logging.info("Salvando resultados para CSV...")

        media_e_desvio_geracao_por_grupo.to_csv(
            path("geracao_power_range_mossoro.csv"), index=False
        )
        usinas_outliers.to_csv(path("outliers_power_range_mossoro.csv"), index=False)

        usinas.to_csv(path("usinas_power_range_mossoro.csv"), index=False)
        usinas_cleaned.to_csv(
            path("usinas_cleaned_power_range_mossoro.csv"), index=False
        )

        usinas_anomalas_com_outliers.to_csv(
            path("anomalous_power_range_mossoro.csv"), index=False
        )
        usinas_anomalas_cleaned.to_csv(
            path("anomalous_cleaned_power_range_mossoro.csv"), index=False
        )

        logging.info("Processamento concluído com sucesso!")
