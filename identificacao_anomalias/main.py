import dask.dataframe as dd
import numpy as np
import logging
import pandas as pd

from tqdm.dask import TqdmCallback

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

if __name__ == "__main__":
    usinas_cols = ['id', 'current_power', 'cidade_nome']
    usinas = dd.read_csv("cleaned/usina_mossoró_merged.csv", usecols=usinas_cols)
    usinas['id'] = dd.to_numeric(usinas['id'])
    logging.info("Iniciando a leitura dos arquivos Parquet...")

    geracao_cols = ['id_usina', 'quantidade', 'data']
    geracao = dd.read_parquet("_data/mossoro/*.parquet", columns=geracao_cols)


    logging.info(f"Leitura concluída. Dataframe contém {len(geracao.columns)} colunas e {geracao.npartitions} partições.")


    logging.info("Convertendo a coluna 'data' para datetime...")
    geracao['data'] = dd.to_datetime(geracao['data'], format='%Y-%m-%d', errors='coerce')


    data_inicio = '2023-01-01'
    logging.info(f"Aplicando filtro para incluir somente dados de {data_inicio} até a data atual...")

    geracao = geracao[geracao['data'] >= data_inicio]

    logging.info("Iniciando o merge dos dataframes...")
    with TqdmCallback(desc="compute") as progress:

        geracao = geracao.merge(
            usinas, 
            left_on='id_usina', 
            right_on='id', 
            how='inner', 
            suffixes=('', '_usina')
        )


        logging.info("Criando médias historica de cada usina potência...")

        media_historica = (
            geracao.groupby('id_usina')['quantidade']
            .mean()
            .reset_index()
            .compute()
        )
        usinas = usinas.merge(
            media_historica,
            left_on='id',
            right_on='id_usina',
            how='left',
            suffixes=('', '_media')
        )

        logging.info("Calculando a faixa de potência de cada usina...")

        max_power = usinas['current_power'].max().compute()
        bins = np.arange(0, max_power + 5, 5)
        usinas['power_range'] = usinas['current_power'].map_partitions(pd.cut, bins=bins)

        logging.info("Calculando as média e variancia esperada por faixa de potência e cidade...")

        # # (Power_range, mean, std)
        media_e_desvio_por_grupo = (
            usinas.groupby(['power_range'])['quantidade']
            .agg(['mean', 'std'])
            .reset_index()
            .compute()
        )

        media_e_desvio_por_grupo = media_e_desvio_por_grupo.rename(
            columns={'mean': 'media_esperada', 'std': 'std'}
        )

        media_e_desvio_por_grupo = media_e_desvio_por_grupo.dropna(subset=['media_esperada', 'std', 'power_range'])

        logging.info("Calculando o z-score para identificar usinas anômalas...")
        usinas = usinas.merge(
            media_e_desvio_por_grupo,
            on='power_range',
            how='left'
        )

        usinas['zscore'] = (usinas['quantidade'] - usinas['media_esperada']) / usinas['std']
        
        zscore_threshold = 2
        usinas['anomalous'] = usinas['zscore'].abs() > zscore_threshold

        usinas_anomalas = usinas[usinas['anomalous']].compute()
        logging.info(f"Número de usinas anômalas detectadas: {len(usinas_anomalas)}")

        media_e_desvio_por_grupo.to_csv("output/geracao_grupo_v1.csv", index=False)
        usinas.to_csv("output/usinas_grupo.csv", index=False, single_file=True)

        logging.info("Processamento concluído com sucesso!")