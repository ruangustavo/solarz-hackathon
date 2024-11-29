from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from estimativa_prognostico_geracao.prever import (
    avaliar_modelo,
    criar_modelo_prophet,
    desnormalizar_previsao,
    preparar_dados,
)


def create_streamlit_app():
    st.set_page_config(
        page_title="Previsão de Geração Solar",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="☀️",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("_shared/images/solarz_logo.png")
    with col2:
        st.title("Previsão de Geração Solar")

    tab1, tab2 = st.tabs(["📊 Previsão", "ℹ️ Sobre"])

    with tab1:
        with st.container():
            st.subheader("Seleção de Dados")
            arquivo_selecionado = st.radio(
                "Selecione a cidade para análise:",
                options=["Mossoró - RN", "Salvador - BA", "Caxias do Sul - RS"],
                index=0,
                help="Escolha a cidade para a qual deseja realizar a previsão",
            )

            arquivo_map = {
                "Mossoró - RN": "./_data/geracao_mossoro.csv",
                "Salvador - BA": "./_data/geracao_salvador.csv",
                "Caxias do Sul - RS": "./_data/geracao_caxias_do_sul.csv",
            }

            st.subheader("Configurações")
            col1, col2 = st.columns(2)
            with col1:
                potencia_alvo = st.number_input(
                    "Potência Instalada (kW)",
                    value=15,
                    step=1,
                    help="Insira a potência instalada do sistema em kilowatts",
                )
            with col2:
                mode = st.radio(
                    "Modo de Operação",
                    ["Fazer Previsão", "Testar Modelo"],
                    help="Escolha entre fazer uma nova previsão ou testar a precisão do modelo",
                )

            if st.button("Executar Análise", key="execute"):
                try:
                    with st.spinner("Processando dados..."):
                        df = pd.read_csv(arquivo_map[arquivo_selecionado])

                        with st.expander("Ver dados brutos (utilizados)"):
                            st.dataframe(df.head())

                        df_prophet = preparar_dados(df)

                        if mode == "Testar Modelo":
                            cutoff_date = df_prophet["ds"].max() - pd.Timedelta(days=90)
                            df_treino = df_prophet[df_prophet["ds"] <= cutoff_date]
                            df_teste = df_prophet[df_prophet["ds"] > cutoff_date]

                            modelo = criar_modelo_prophet()
                            modelo.fit(df_treino)

                            futuro = modelo.make_future_dataframe(periods=180)
                            previsoes = modelo.predict(futuro)

                            previsoes_teste = previsoes[
                                previsoes["ds"].isin(df_teste["ds"])
                            ]
                            mape = avaliar_modelo(
                                df_teste["y"].values, previsoes_teste["yhat"].values
                            )

                            st.metric(
                                "Erro Percentual Médio Absoluto (MAPE)",
                                f"{mape:.2f}%",
                                help="Menor valor indica melhor precisão",
                            )

                        else:
                            modelo = criar_modelo_prophet()
                            modelo.fit(df_prophet)
                            today = pd.Timestamp.now().normalize()

                            future_dates = pd.date_range(
                                start=today + pd.Timedelta(days=1),
                                periods=30,
                                freq="D",
                            )
                            futuro = pd.DataFrame({"ds": future_dates})
                            previsoes = modelo.predict(futuro)

                        previsoes_final = desnormalizar_previsao(
                            previsoes, potencia_alvo
                        )

                        fig = go.Figure()

                        fig.add_trace(
                            go.Scatter(
                                x=previsoes_final["ds"],
                                y=previsoes_final["yhat"],
                                name="Previsão",
                                line=dict(color="#00cc96", width=2),
                                hovertemplate="Data: %{x}<br>"
                                + "Geração: %{y:.1f} kWh<br>"
                                + "<extra></extra>",
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=previsoes_final["ds"].tolist()
                                + previsoes_final["ds"].tolist()[::-1],
                                y=previsoes_final["yhat_upper"].tolist()
                                + previsoes_final["yhat_lower"].tolist()[::-1],
                                fill="toself",
                                fillcolor="rgba(0, 204, 150, 0.2)",
                                line=dict(color="rgba(0,0,0,0)"),
                                name="Intervalo de Confiança 95%",
                                hoverinfo="skip",
                                showlegend=True,
                            )
                        )

                        fig.update_layout(
                            title={
                                "text": "Previsão de Geração Solar",
                                "y": 0.95,
                                "x": 0.5,
                                "xanchor": "center",
                                "yanchor": "top",
                                "font": {"size": 24},
                            },
                            xaxis_title="Data",
                            yaxis_title="Geração (kWh)",
                            hovermode="x unified",
                            plot_bgcolor="#0e1117",
                            paper_bgcolor="#0e1117",
                            font=dict(color="#fafafa"),
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01,
                                bgcolor="rgba(14,17,23,0.5)",
                                font=dict(color="#fafafa"),
                            ),
                            margin=dict(l=40, r=40, t=60, b=40),
                            height=600,
                        )

                        fig.update_xaxes(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="rgba(255,255,255,0.1)",
                            showline=True,
                            linewidth=1,
                            linecolor="rgba(255,255,255,0.2)",
                        )

                        fig.update_yaxes(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="rgba(255,255,255,0.1)",
                            showline=True,
                            linewidth=1,
                            linecolor="rgba(255,255,255,0.2)",
                            rangemode="nonnegative",
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        df_previsao = pd.read_csv(
                            f"./estimativa_prognostico_geracao/previsoes_{potencia_alvo:.0f}kWh.csv"
                        )
                        with st.expander("Ver previsões (últimos 5 dias)"):
                            st.dataframe(df_previsao.tail())

                        st.download_button(
                            label="📥 Baixar Previsões (CSV)",
                            data=previsoes_final.to_csv(index=False),
                            file_name=f"previsoes_solar_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                        )
                except FileNotFoundError:
                    st.error(
                        "Arquivo não encontrado. Por favor, verifique se os dados estão disponíveis."
                    )
                except ValueError as e:
                    st.error(f"Erro nos valores fornecidos: {str(e)}")
                except Exception as e:
                    st.error(f"Ocorreu um erro inesperado: {str(e)}")

    with tab2:
        st.header("Sobre o Sistema")
        st.write("""
        ### Como usar esta ferramenta
        1. Selecione a cidade desejada para análise
        2. Insira a potência instalada do sistema em kilowatts
        3. Escolha o modo de operação:
            - **Fazer Previsão**: Gera previsões para os próximos 30 dias
            - **Testar Modelo**: Avalia a precisão do modelo usando dados históricos
        4. Clique em 'Executar Análise' para ver os resultados

        ### Sobre as Previsões
        - As previsões são baseadas em dados históricos de geração
        - O modelo considera padrões sazonais e tendências
        - O intervalo de confiança de 95% é mostrado na área sombreada
        - Os resultados podem ser exportados em formato CSV
        """)


if __name__ == "__main__":
    create_streamlit_app()
