import pandas as pd
import matplotlib.pyplot as plt
from _shared.path import path

data = pd.read_csv(path("usinas_power_range_mossoro.csv"))

anomalous_data = data[data['anomalous'] == True]
selected_plant = anomalous_data.sample(1)  

plant_id = selected_plant['id'].values[0]
potencia = selected_plant['potencia'].values[0]
media_esperada = selected_plant['media_esperada'].values[0]
media_real = selected_plant['quantidade'].values[0]

limite_80 = 0.8 * media_esperada

categories = ['Geração Real (kWh)', 'Geração minima esperada 80%', 'Geração Média Região (kWh)']
values = [media_real, limite_80, media_esperada]

plt.figure(figsize=(12, 8))
bars = plt.bar(categories, values, color=['red', 'orange', 'green'], alpha=0.8)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.2f}', ha='center', fontsize=12)

plt.title(f"Análise Detalhada - Usina Anômala ID: {plant_id}", fontsize=18, pad=20)
plt.ylabel("Energia (kWh)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

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

plt.show()
