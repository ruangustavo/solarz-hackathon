import warnings

import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
import pandas as pd

matplotlib.use("QtAgg")


def processar_dados_usina(dados_geracao, id_usina):
    # Filtrar dados inválidos de forma vetorizada
    mascara_valida = (
        dados_geracao["potencia"].notna()
        & (dados_geracao["potencia"] > 0)
        & dados_geracao["quantidade"].notna()
        & (dados_geracao["quantidade"] > 0)
        & dados_geracao["prognostico"].notna()
        & (dados_geracao["prognostico"] > 0)
    )

    dados_validos = dados_geracao[mascara_valida].copy()

    # Calcular capacidade máxima e filtrar
    dados_validos["capacidade_maxima"] = dados_validos["potencia"] * 24
    dados_validos = dados_validos[
        (dados_validos["prognostico"] <= dados_validos["capacidade_maxima"])
        & (dados_validos["quantidade"] <= dados_validos["capacidade_maxima"])
    ]

    # Extrair ano de forma eficiente
    dados_validos["ano"] = pd.to_datetime(dados_validos["data"]).dt.year

    # Agregar por ano usando numpy para maior velocidade
    dados_anuais = dados_validos.groupby("ano").agg(
        {"prognostico": "sum", "quantidade": "sum"}
    )

    if len(dados_anuais) < 4:
        return None

    # Criar array de anos completo
    anos_range = np.arange(1999, 2025)
    dados_completos = pd.DataFrame(index=anos_range)
    dados_completos = dados_completos.join(dados_anuais)
    dados_completos.fillna(0, inplace=True)

    # Calcular ajuste de forma vetorizada
    ano_inicial = dados_completos[dados_completos["prognostico"] > 0].index.min()
    anos_desde_inicio = dados_completos.index - ano_inicial

    # Vetorizar cálculo de ajuste
    ajuste = np.where(
        anos_desde_inicio <= 0,
        1,
        np.where(
            anos_desde_inicio == 1, 0.975, 0.975 - (0.005 * (anos_desde_inicio - 1))
        ),
    )

    dados_completos["previsao_ajustada"] = dados_completos["prognostico"] * ajuste

    # Calcular desempenho de forma vetorizada
    dados_completos["desempenho"] = np.where(
        dados_completos["previsao_ajustada"] > 0,
        dados_completos["quantidade"] / dados_completos["previsao_ajustada"],
        0,
    )
    dados_completos["desempenho"] = dados_completos["desempenho"].round(2)

    return dados_completos


def verificar_problemas(desempenho_serie):
    queda_minima = 0.05
    queda_maxima = 0.1

    desempenho_array = desempenho_serie.values
    for i in range(len(desempenho_array) - 3):
        valores = desempenho_array[i : i + 4]
        quedas = (valores[:-1] - queda_minima >= valores[1:]) & (
            valores[:-1] - queda_maxima < valores[1:]
        )
        if quedas.sum() == 3:
            return True
    return False


def processar_usinas(caminho_arquivo, limite_usinas=10):
    # Ler dados uma única vez
    dados_geracao = pd.read_csv(
        caminho_arquivo,
        usecols=["id_usina", "data", "quantidade", "prognostico", "potencia"],
    )

    # Obter usinas únicas
    usinas_unicas = dados_geracao["id_usina"].unique()

    for id_usina in usinas_unicas[:limite_usinas]:
        # Filtrar dados da usina
        dados_usina = dados_geracao[dados_geracao["id_usina"] == id_usina]

        # Processar dados
        resultados = processar_dados_usina(dados_usina, id_usina)
        if resultados is None:
            continue

        # Verificar problemas
        problematica = verificar_problemas(resultados["desempenho"])

        # Plotar
        # plt.figure(figsize=(10, 6))
        # plt.plot(resultados.index, resultados["desempenho"], "ro")
        # plt.grid(True)
        # plt.savefig(f"./usinas/{'defeito' if problematica else 'comum'}/{id_usina}.png")
        # plt.close()
        print(f"Usina {id_usina} {'com' if problematica else 'sem'} problemas")


if __name__ == "__main__":
    processar_usinas("../_data/geracao_mossoro.csv", 10000)
