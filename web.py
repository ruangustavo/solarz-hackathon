from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared.path import path

# Reuse your existing functions
from estimativa_prognostico_geracao.prever import (
    avaliar_modelo,
    criar_modelo_prophet,
    desnormalizar_previsao,
    preparar_dados,
)


def create_streamlit_app():
    st.set_page_config(page_title="Previsão de Geração Solar", layout="wide")

    # Header with logo and title
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("_shared/images/solarz_logo.png")
    with col2:
        st.title("Previsão de Geração Solar")

    st.write("Selecione os dados e gere previsões de energia solar")

    # File selection via radio buttons
    arquivo_selecionado = st.radio(
        "Selecione a cidade para análise:",
        options=["Mossoró - RN", "Salvador - BA", "Caxias do Sul - RS"],
        index=0,
    )

    # Map selection to file names
    arquivo_map = {
        "Mossoró - RN": path("geracao_mossoro.csv"),
        "Salvador - BA": path("geracao_salvador.csv"),
        "Caxias do Sul - RS": path("geracao_caxias_do_sul.csv"),
    }

    # Load selected file
    try:
        df = pd.read_csv(arquivo_map[arquivo_selecionado])
        st.subheader("Visualização dos Dados")
        st.dataframe(df.head())

        # Input for installed power
        potencia_alvo = st.number_input("Potência Instalada (kW)", value=15, step=1)

        # Mode selection
        mode = st.radio("Selecione o Modo", ["Fazer Previsão", "Testar Modelo"])

        if st.button("Executar Análise"):
            with st.spinner("Processando dados..."):
                # Prepare data
                df_prophet = preparar_dados(df)

                if mode == "Testar Modelo":
                    # Test mode logic
                    cutoff_date = df_prophet["ds"].max() - pd.Timedelta(days=90)
                    df_treino = df_prophet[df_prophet["ds"] <= cutoff_date]
                    df_teste = df_prophet[df_prophet["ds"] > cutoff_date]

                    modelo = criar_modelo_prophet()
                    modelo.fit(df_treino)

                    futuro = modelo.make_future_dataframe(periods=180)
                    previsoes = modelo.predict(futuro)

                    previsoes_teste = previsoes[previsoes["ds"].isin(df_teste["ds"])]
                    mape = avaliar_modelo(
                        df_teste["y"].values, previsoes_teste["yhat"].values
                    )

                    st.metric("Erro Percentual Médio Absoluto (MAPE)", f"{mape:.2f}%")

                else:
                    # Prediction mode logic
                    modelo = criar_modelo_prophet()
                    modelo.fit(df_prophet)

                    # Forecast next 30 days
                    last_date = df_prophet["ds"].max()
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1), periods=30, freq="D"
                    )
                    futuro = pd.DataFrame({"ds": future_dates})
                    previsoes = modelo.predict(futuro)

                # Denormalize and display results
                previsoes_final = desnormalizar_previsao(previsoes, potencia_alvo)

                # Create interactive plot with Plotly
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=previsoes_final["ds"],
                        y=previsoes_final["yhat"],
                        name="Previsão",
                        line=dict(color="blue"),
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=previsoes_final["ds"],
                        y=previsoes_final["yhat_upper"],
                        fill=None,
                        mode="lines",
                        line=dict(color="rgba(0,0,255,0)"),
                        showlegend=False,
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=previsoes_final["ds"],
                        y=previsoes_final["yhat_lower"],
                        fill="tonexty",
                        mode="lines",
                        line=dict(color="rgba(0,0,255,0)"),
                        name="Intervalo de Confiança",
                    )
                )

                fig.update_layout(
                    title="Previsão de Geração Solar",
                    xaxis_title="Data",
                    yaxis_title="Geração (kWh)",
                    hovermode="x unified",
                )

                st.plotly_chart(fig, use_container_width=True)

                # Download predictions
                st.download_button(
                    label="Baixar Previsões (CSV)",
                    data=previsoes_final.to_csv(index=False),
                    file_name=f"previsoes_solar_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {str(e)}")


if __name__ == "__main__":
    create_streamlit_app()
