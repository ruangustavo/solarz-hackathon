from _shared.path import path
import pandas as pd

import matplotlib.pyplot as plt


df = pd.read_csv(path('analise_usina_depreciacao.csv'))

plt.figure(figsize=(10, 6))
x_labels = df['year'].astype(str)
width = 0.35

x = range(len(x_labels))

plt.bar(x, df['actual_generation'], width, label='Geração Real')
plt.bar([p + width for p in x], df['expected_generation'], width, label='Geração Esperada')

plt.xlabel('Ano')
plt.ylabel('Geração (MWh)')
plt.title('Geração Real vs Esperada por Ano')
plt.xticks([p + width / 2 for p in x], x_labels)
plt.legend()

plt.tight_layout()
plt.savefig(path('analise_usina_depreciacao_teste.png'))