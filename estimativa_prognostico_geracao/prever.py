import argparse

import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error


def preparar_dados(df):
    """
    Prepara os dados normalizando a geração pela potência instalada,
    removendo valores problemáticos
    """
    # Remove registros com quantidade ou potência zero/nula
    df = df[
        (df["quantidade"] > 0)
        & (df["potencia"] > 0)
        & (~df["quantidade"].isna())
        & (~df["potencia"].isna())
    ].copy()

    # Calcula geração relativa
    df["geracao_relativa"] = df["quantidade"] / df["potencia"]

    # Remove outliers extremos
    q1 = df["geracao_relativa"].quantile(0.01)  # 1%
    q3 = df["geracao_relativa"].quantile(0.99)  # 1%
    df = df[(df["geracao_relativa"] >= q1) & (df["geracao_relativa"] <= q3)]

    # Agrupa por _data e calcula média da geração relativa
    df_prep = df.groupby("data")["geracao_relativa"].mean().reset_index()

    # Renomeia colunas para o Prophet
    df_prep.columns = ["ds", "y"]
    df_prep["ds"] = pd.to_datetime(df_prep["ds"])

    return df_prep


def criar_modelo_prophet():
    """
    Cria modelo Prophet otimizado para previsão de geração relativa
    """
    modelo = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    return modelo


def desnormalizar_previsao(previsoes, potencia_alvo):
    """
    Converte previsões normalizadas para valores absolutos
    """
    previsoes_copy = previsoes.copy()
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        previsoes_copy[col] = previsoes_copy[col] * potencia_alvo
    return previsoes_copy


def avaliar_modelo(y_true, y_pred):
    """
    Calcula MAPE ignorando valores nulos (MAPE: Erro Percentual Absoluto Médio)
    """
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mape * 100


def plotar_grafico(previsoes):
    """
    Apresenta os resultados das previsões de forma amigável
    """
    # Calcula estatísticas importantes
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(previsoes["ds"], previsoes["yhat"], label="Previsão")
        plt.fill_between(
            previsoes["ds"],
            previsoes["yhat_lower"],
            previsoes["yhat_upper"],
            alpha=0.2,
            label="Intervalo de Confiança",
        )
        plt.title("Previsão de Geração Solar")
        plt.xlabel("data")
        plt.ylabel("Geração (kWh)")
        plt.legend()
        plt.savefig("previsao_geracao.png")
        print("\nGráfico salvo em 'previsao_geracao.png'")
    except Exception as e:
        print("\nNão foi possível gerar o gráfico", e)


def main():
    parser = argparse.ArgumentParser(description="Solar generation forecasting")
    parser.add_argument(
        "--mode",
        choices=["test", "predict"],
        default="predict",
        help="test: evaluate model with MAPE, predict: forecast future values",
    )
    args = parser.parse_args()

    df = pd.read_csv("../_data/geracao_mossoro.csv")
    df_prophet = preparar_dados(df)

    if len(df_prophet) < 2:
        print("Dados insuficientes após limpeza")
        return

    if args.mode == "test":
        cutoff_date = df_prophet["ds"].max() - pd.Timedelta(days=90)
        df_treino = df_prophet[df_prophet["ds"] <= cutoff_date]
        df_teste = df_prophet[df_prophet["ds"] > cutoff_date]

        if len(df_treino) < 2 or len(df_teste) < 1:
            print("Dados insuficientes para treino/teste após divisão")
            return

        modelo = criar_modelo_prophet()
        modelo.fit(df_treino)

        futuro = modelo.make_future_dataframe(periods=180)
        previsoes = modelo.predict(futuro)

        previsoes_teste = previsoes[previsoes["ds"].isin(df_teste["ds"])]
        mape = avaliar_modelo(df_teste["y"].values, previsoes_teste["yhat"].values)
        print(f"MAPE: {mape:.2f}%")
    else:
        modelo = criar_modelo_prophet()
        modelo.fit(df_prophet)

        last_date = df_prophet["ds"].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=30, freq="D"
        )
        futuro = pd.DataFrame({"ds": future_dates})
        previsoes = modelo.predict(futuro)

    POTENCIA_ALVO = 10.495
    previsoes_final = desnormalizar_previsao(previsoes, potencia_alvo=POTENCIA_ALVO)
    previsoes_final.to_csv(f"previsoes_{POTENCIA_ALVO:.0f}kWh.csv", index=False)
    plotar_grafico(previsoes)


if __name__ == "__main__":
    main()
