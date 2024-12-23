import dask.dataframe as dd
import numpy as np
import logging
import pandas as pd
from tqdm.dask import TqdmCallback
import matplotlib.pyplot as plt

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

if __name__ == "__main__":
    usinas_cols = ["id", "current_power", "cidade_nome"]
    usinas = dd.read_csv("cleaned/usina_mossoró_merged.csv", usecols=usinas_cols)
    usinas["id"] = dd.to_numeric(usinas["id"])
    logging.info("Iniciando a leitura dos arquivos Parquet...")

    geracao_cols = ["id_usina", "quantidade", "data"]
    geracao = dd.read_parquet("_data/mossoro/*.parquet", columns=geracao_cols)

    logging.info(
        f"Leitura concluída. Dataframe contém {len(geracao.columns)} colunas e {geracao.npartitions} partições."
    )

    logging.info("Convertendo a coluna 'data' para datetime...")
    geracao["data"] = dd.to_datetime(
        geracao["data"], format="%Y-%m-%d", errors="coerce"
    )

    data_inicio = "2024-01-01"
    data_final = "2024-11-10"

    logging.info(
        f"Aplicando filtro para incluir somente dados de {data_inicio} até {data_final}..."
    )

    geracao = geracao[
        (geracao["data"] >= data_inicio) & (geracao["data"] <= data_final)
    ]

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

        media_historica = (
            geracao.groupby("id_usina")["quantidade"].mean().reset_index().compute()
        )
        usinas = usinas.merge(
            media_historica,
            left_on="id",
            right_on="id_usina",
            how="left",
            suffixes=("", "_media"),
        )

        logging.info("Calculando a faixa de potência de cada usina...")
        max_power = usinas["current_power"].max().compute()
        bins = np.arange(0, max_power + 5, 5)
        usinas["power_range"] = usinas["current_power"].map_partitions(
            pd.cut, bins=bins
        )

        logging.info(
            "Calculando as média e variancia esperada por faixa de potência e cidade..."
        )

        media_e_desvio_por_grupo = (
            usinas.groupby(["power_range"])["quantidade"]
            .agg(["mean", "std"])
            .reset_index()
            .compute()
        )

        media_e_desvio_por_grupo = media_e_desvio_por_grupo.rename(
            columns={"mean": "media_esperada", "std": "std"}
        )

        media_e_desvio_por_grupo = media_e_desvio_por_grupo.dropna(
            subset=["media_esperada", "std", "power_range"]
        )

        logging.info("Calculando o z-score para identificar usinas anômalas...")
        usinas = usinas.merge(media_e_desvio_por_grupo, on="power_range", how="left")

        usinas["zscore"] = (usinas["quantidade"] - usinas["media_esperada"]) / usinas[
            "std"
        ]

        zscore_threshold = 0.8
        usinas["anomalous"] = usinas["zscore"].abs() > zscore_threshold

        usinas_anomalas = usinas[usinas["anomalous"]].compute()
        logging.info(f"Número de usinas anômalas detectadas: {len(usinas_anomalas)}")

        media_e_desvio_por_grupo.to_csv("output/geracao_grupo_v1.csv", index=False)
        usinas.to_csv("output/usinas_grupo.csv", index=False, single_file=True)

        logging.info("Processamento concluído com sucesso!")

        usinas_pd = usinas.compute()
        target_interval = pd.Interval(30, 35, closed="right")
        usinas_pd = usinas_pd[usinas_pd["power_range"] == target_interval]
        usinas_pd = usinas_pd.sort_values(by="media_esperada")
        usinas_pd = usinas_pd.dropna(subset=["zscore"])

        plt.figure(figsize=(15, 6))

        normal_usinas = usinas_pd[usinas_pd["zscore"].abs() <= zscore_threshold]
        anomalous_usinas = usinas_pd[usinas_pd["zscore"].abs() > zscore_threshold]

        plt.scatter(
            normal_usinas["current_power"],
            normal_usinas["quantidade"],
            color="blue",
            label="Usinas Normais",
            alpha=0.6,
        )

        plt.scatter(
            anomalous_usinas["current_power"],
            anomalous_usinas["quantidade"],
            color="red",
            label="Usinas Anômalas",
            alpha=0.6,
        )
        plt.plot(
            usinas_pd["current_power"],
            usinas_pd["media_esperada"],
            color="yellow",
            linestyle="-",
            linewidth=1,
            label="Média de produção",
        )
        plt.text(
            0.02,
            0.98,
            f"Total de usinas: {len(usinas_pd)}\nUsinas normais: {len(normal_usinas)} - {(len(normal_usinas)/len(usinas_pd)*100):.2f}% \nUsinas anômalas: {len(anomalous_usinas)} - {(len(anomalous_usinas)/len(usinas_pd)*100):.2f}%",
            transform=plt.gca().transAxes,
            fontsize=12,
            color="black",
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        plt.xlabel("Potência")
        plt.ylabel("Geração no período")
        plt.title("Dispersão de Z-Scores por Faixa de Potência")
        plt.legend()
        plt.tight_layout()

        plt.savefig("output/zscore_scatter_plot.png")
        logging.info(
            "Gráfico de dispersão salvo como 'output/zscore_scatter_plot_anomalous.png'."
        )
