import argparse

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error


def preparar_dados(df):
    df = df[
        (df["quantidade"] > 0)
        & (df["potencia"] > 0)
        & (~df["quantidade"].isna())
        & (~df["potencia"].isna())
    ].copy()
    df["geracao_relativa"] = df["quantidade"] / df["potencia"]

    q1 = df["geracao_relativa"].quantile(0.01)
    q3 = df["geracao_relativa"].quantile(0.99)
    df = df[(df["geracao_relativa"] >= q1) & (df["geracao_relativa"] <= q3)]

    df_prep = df.groupby("data")["geracao_relativa"].mean().reset_index()

    df_prep.columns = ["ds", "y"]
    df_prep["ds"] = pd.to_datetime(df_prep["ds"])

    return df_prep


def criar_modelo_prophet():
    modelo = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    return modelo


def desnormalizar_previsao(previsoes, potencia_alvo):
    previsoes_copy = previsoes.copy()
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        previsoes_copy[col] = previsoes_copy[col] * potencia_alvo
    return previsoes_copy


def avaliar_modelo(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mape * 100


def plotar_grafico(previsoes_final):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=previsoes_final["data"],
            y=previsoes_final["previsao"],
            name="Previsão",
            line=dict(color="rgb(31, 119, 180)"),
            hovertemplate="Data: %{x}<br>"
            + "Geração: %{y:.1f} kWh<br>"
            + "<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=previsoes_final["data"].tolist() + previsoes_final["data"].tolist()[::-1],
            y=previsoes_final["limite_superior"].tolist()
            + previsoes_final["limite_inferior"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Intervalo de Confiança 95%",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    fig.update_layout(
        title="Previsão de Geração Solar",
        xaxis_title="Data",
        yaxis_title="Geração (kWh)",
        hovermode="x unified",
        plot_bgcolor="white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128, 128, 128, 0.2)",
        showline=True,
        linewidth=1,
        linecolor="black",
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128, 128, 128, 0.2)",
        showline=True,
        linewidth=1,
        linecolor="black",
        rangemode="nonnegative",
    )

    fig.write_image("previsao_geracao_solar.png")


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

        today = pd.Timestamp.now().normalize()
        start_date = today + pd.Timedelta(days=1)
        future_dates = pd.date_range(
            start=start_date,
            periods=30,
            freq="D",
        )
        futuro = pd.DataFrame({"ds": future_dates})
        previsoes = modelo.predict(futuro)

    POTENCIA_ALVO = 15
    previsoes_final = desnormalizar_previsao(previsoes, potencia_alvo=POTENCIA_ALVO)
    previsoes_final = previsoes_final[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    previsoes_final.columns = ["data", "previsao", "limite_inferior", "limite_superior"]
    previsoes_final = previsoes_final.round(2)
    previsoes_final.to_csv(f"previsoes_{POTENCIA_ALVO:.0f}kWh.csv", index=False)
    plotar_grafico(previsoes_final)


if __name__ == "__main__":
    main()
