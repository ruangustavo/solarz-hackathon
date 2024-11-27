import streamlit as st
import logging

from app.tabs.previsao import previsao

def main():
    st.title("ETAL - SolaZ Hackathon 2024")

    previsao_tab, estimativa_desempenho_atual_tab = st.tabs(["Estimativa de Prognóstico de Geração", "Estimativa de Desempenho Atual"])

    with previsao_tab:
        previsao()

    with estimativa_desempenho_atual_tab:
        ...



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()