import streamlit as st
import dask.dataframe as dd
import pandas as pd
import logging
from tqdm.dask import TqdmCallback
import matplotlib.pyplot as plt

from _shared.path import path

# Configuração de logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Função principal
def estimativa_desempenho_atual():
    st.title("Dashboard de Análise de Usinas")
    st.sidebar.header("Filtros")
    
    # Inputs do usuário
    data_inicio = st.sidebar.date_input("Data de Início", value=pd.to_datetime('2023-01-01'))
    cidade_filtro = st.sidebar.text_input("Cidade (opcional)")

    # Leitura dos dados
    st.info("Carregando dados...")
    usinas_cols = ['id', 'current_power', 'cidade_nome']
    usinas = dd.read_csv(path("usina_natal_merged.csv"), usecols=usinas_cols)

    geracao_cols = ['id_usina', 'quantidade', 'data']
    geracao = dd.read_parquet(path("geracao/*.parquet"), columns=geracao_cols)

    # Filtrando dados com base na data de início
    st.info(f"Filtrando dados a partir de {data_inicio}...")
    geracao['data'] = dd.to_datetime(geracao['data'], format='%Y-%m-%d', errors='coerce')
    geracao = geracao[geracao['data'] >= str(data_inicio)]

    # Aplicando filtro de cidade, se fornecido
    if cidade_filtro:
        st.info(f"Aplicando filtro para a cidade: {cidade_filtro}")
        usinas = usinas[usinas['cidade_nome'] == cidade_filtro]

    # Merge dos dataframes
    st.info("Realizando merge dos dataframes...")
    with TqdmCallback(desc="Merge progress"):
        geracao = geracao.merge(
            usinas,
            left_on='id_usina',
            right_on='id',
            how='inner',
            suffixes=('', '_usina')
        )

    # Calculando métricas
    st.info("Calculando métricas...")
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

    usinas['anomalous'] = usinas['quantidade'] < usinas['quantidade'].mean() * 0.8

    # Dashboard de análise
    st.header("Análise de Usinas")

    total_usinas = usinas['id'].nunique().compute()
    total_anomalas = usinas['anomalous'].sum().compute()
    st.metric("Total de Usinas", total_usinas)
    st.metric("Usinas Anômalas Detectadas", total_anomalas)

    # Gráficos
    st.subheader("Distribuição de Geração por Cidade")
    geracao_cidade = geracao.groupby('cidade_nome')['quantidade'].mean().compute()

    fig, ax = plt.subplots()
    geracao_cidade.plot(kind='bar', ax=ax)
    ax.set_ylabel("Média de Geração")
    ax.set_xlabel("Cidade")
    ax.set_title("Geração Média por Cidade")
    st.pyplot(fig)

    st.subheader("Distribuição de Anomalias")
    fig, ax = plt.subplots()
    usinas['anomalous'].value_counts().compute().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_ylabel("")
    ax.set_title("Percentual de Usinas Anômalas")
    st.pyplot(fig)

    st.subheader("Tabela de Usinas Anômalas")
    usinas_anomalas = usinas[usinas['anomalous']].compute()
    st.dataframe(usinas_anomalas)

    st.success("Análise concluída!")