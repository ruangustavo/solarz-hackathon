from datetime import datetime

import pandas as pd
import plotly.express as px

import streamlit as st

import dask.dataframe as dd
import numpy as np

import logging

from _shared.path import path

def load_and_analyze_data(cidade, data_inicio, potencia_alvo):
    """
    Load and process solar generation data with advanced anomaly detection
    """
    logging.basicConfig(level=logging.INFO)

    try:
        usinas_cols = ['id', 'current_power', 'cidade_nome']
        usinas = dd.read_csv(path("usina_natal_merged.csv"), usecols=usinas_cols)
        

        # intervalo de potência de +/- 2.5 kW
        usinas = usinas[
            (usinas['cidade_nome'] == cidade) & 
            (usinas['current_power'] >= potencia_alvo - 2.5) & 
            (usinas['current_power'] <= potencia_alvo + 2.5)
        ].compute()
        
        geracao_cols = ['id_usina', 'quantidade', 'data']
        geracao = dd.read_parquet(path("geracao/*.parquet"), columns=geracao_cols)
        
        geracao['data'] = dd.to_datetime(geracao['data'], format='%Y-%m-%d', errors='coerce')
        geracao = geracao[geracao['data'] >= pd.to_datetime(data_inicio)].compute()
        
        geracao = geracao.merge(
            usinas,
            left_on='id_usina',
            right_on='id',
            how='inner',
            suffixes=('', '_usina')
        ).compute()

        media_historica = (
            geracao.groupby('id_usina')['quantidade']
            .mean()
            .reset_index()
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

        media_e_desvio_por_grupo = (
            usinas.groupby(['power_range'])['quantidade']
            .agg(['mean', 'std'])
            .reset_index()
        )

        media_e_desvio_por_grupo = media_e_desvio_por_grupo.rename(
            columns={'mean': 'media_esperada', 'std': 'std'}
        )

        media_e_desvio_por_grupo = media_e_desvio_por_grupo.dropna(subset=['media_esperada', 'std', 'power_range'])
        
        usinas = usinas.merge(
            media_e_desvio_por_grupo,
            on='power_range',
            how='left'
        )

        anomalies = usinas[usinas['quantidade'] < usinas['media_esperada'] * 0.8]
        
        return usinas, anomalies
    
    except Exception as e:
        st.error(f"Erro no processamento: {str(e)}")
        return None, None
    

def create_visualization(data, anomalies):
    """
    Create interactive visualizations for anomaly detection
    """
    # Scatter plot of Generation vs Power
    scatter_fig = px.scatter(
        data, 
        x='current_power', 
        y='quantidade', 
        color=np.abs(data['z_score']) > 2,
        title='Geração Solar: Potência vs Quantidade',
        labels={'current_power': 'Potência Instalada (kW)', 'quantidade': 'Geração (kWh)'},
        color_discrete_map={True: 'red', False: 'blue'}
    )
    
    # Boxplot of Generation by Power Range
    data['power_category'] = pd.cut(data['current_power'], bins=5)
    boxplot_fig = px.box(
        data, 
        x='power_category', 
        y='quantidade', 
        title='Distribuição de Geração por Faixa de Potência'
    )
    
    return scatter_fig, boxplot_fig




def create_anomaly_detection_tab():
    st.header("Identificação de Anomalias em Usinas Solares")
    
    # Inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # cidades = ["Mossoró - RN", "Salvador - BA", "Caxias do Sul - RS"]
        cidades = ["Natal"]
        cidade = st.selectbox("Cidade", cidades)
    
    with col2:
        data_inicio = st.date_input("Data de Início", value=datetime(2023, 1, 1))
    
    with col3:
        potencia_alvo = st.number_input("Potência Alvo (kW)", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
    
    if st.button("Analisar Anomalias"):
        with st.spinner('Processando dados...'):
            merged_data, anomalias = load_and_analyze_data(cidade, data_inicio, potencia_alvo)
            
            if merged_data is not None and not anomalias.empty:
                st.subheader(f"Análise de Usinas Anômalas em {cidade}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total de Usinas", len(merged_data))
                with col2:
                    st.metric("Usinas Anômalas", len(anomalias))
                with col3:
                    st.metric("% Geração Abaixo do Esperado", f"{(len(anomalias)/len(merged_data)*100):.2f}%")
                
                st.dataframe(anomalias[[ 'id_usina', 'current_power', 'quantidade', 'media_esperada' ]])
                
                fig = px.scatter(
                    merged_data, 
                    x='current_power', 
                    y='quantidade', 
                    color='generation_percentage',
                    title='Geração vs Potência Instalada',
                    labels={
                        'current_power': 'Potência Instalada (kW)', 
                        'quantidade': 'Geração Real (kWh)',
                        'media_esperada': 'Média Esperada',
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.download_button(
                    label="Baixar Dados de Anomalias",
                    data=anomalias.to_csv(index=False),
                    file_name=f"anomalias_{cidade}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            else:
                st.warning("Nenhuma anomalia significativa encontrada.")