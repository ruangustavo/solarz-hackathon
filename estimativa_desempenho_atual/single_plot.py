import pandas as pd
import matplotlib.pyplot as plt

from _shared.path import path

# Carregar o dataset
data = pd.read_csv(path("usinas_power_range_mossoro.csv"))

# Filtrar uma usina anômala aleatória
anomalous_data = data[data['anomalous'] == True]
selected_plant = anomalous_data.sample(1)  # Selecionar uma usina anômala aleatoriamente

# Obter dados da usina selecionada
plant_id = selected_plant['id'].values[0]
potencia = selected_plant['potencia'].values[0]
media_esperada = selected_plant['media_esperada'].values[0]
media_real = selected_plant['quantidade'].values[0]

# Calcular o limite de 80% da geração esperada
limite_80 = 0.8 * media_esperada

# Preparar os dados para o gráfico
categories = ['Geração Real (kWh)', 'Geração minima esperada 80%', 'Geração Média Região (kWh)']
values = [media_real, limite_80, media_esperada]

# Criar o gráfico de barras
plt.figure(figsize=(12, 8))
bars = plt.bar(categories, values, color=['red', 'orange', 'green'], alpha=0.8)

# Adicionar valores acima das barras
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.2f}', ha='center', fontsize=12)

# Personalizar o gráfico
plt.title(f"Análise Detalhada - Usina Anômala ID: {plant_id}", fontsize=18, pad=20)
plt.ylabel("Energia (kWh)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionar explicação sobre o motivo da anomalia dentro do gráfico
explanation_text = (
    f"A usina não alcançou 80% da média esperada ({limite_80:.2f} kWh) da região.\n"
    f"Potência Instalada: {potencia} kWp\n"
    f"Geração Real: {media_real:.2f} kWh ({(media_real / media_esperada) * 100:.2f}% da média esperada)."
)
plt.gca().text(
    0.5, 0.92, explanation_text, 
    transform=plt.gca().transAxes, 
    fontsize=12, 
    color='black',
    ha='center', 
    va='top', 
    bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray')
)

plt.savefig(f'analise_detalhada_usina_{plant_id}_ajustado.png', dpi=300, bbox_inches='tight')

# Exibir o gráfico
plt.show()
