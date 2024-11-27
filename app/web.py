import streamlit as st
import logging

from app.tabs.previsao import previsao
from app.tabs.estimativa_desempenho_atual_tab import estimativa_desempenho_atual

def main():
    st.title("ETAL - SolaZ Hackathon 2024")

    previsao_tab, estimativa_desempenho_atual_tab = st.tabs(["Estimativa de Prognóstico de Geração", "Estimativa de Desempenho Atual"])

    with previsao_tab:
        previsao()

    with estimativa_desempenho_atual_tab:
        estimativa_desempenho_atual()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()