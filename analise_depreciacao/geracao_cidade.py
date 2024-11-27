import math
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt


# Gerar planilhas com a geração de todas as usinas da cidade
df = pd.concat(
  map(
    pd.read_csv,
    [
      '/home/lucasnbs/Downloads/usina',
      '/home/lucasnbs/Downloads/unidade_consumidora',
      '/home/lucasnbs/Downloads/endereco',
      '/home/lucasnbs/Downloads/cidade'
    ]
  ),
  ignore_index=True
)

a = df[df['id_cidade'] == 1]
id_enderecos = []
for x in a['id']:
  id_enderecos.append(x)



b = df[df['id_endereco'].isin(id_enderecos)]
id_unidades = []
for x in b['id']:
  id_unidades.append(x)



c = df[df['unidade_consumidora_id'].isin(id_unidades)]
id_usinas = []
for x in c['id']:
  id_usinas.append(x)



for index in range(528):
  geracao = pd.read_parquet(
    f"/home/lucasnbs/hackathon/cleaned/geracao/data-{index}.parquet", engine='pyarrow'
  )
  usinas_mossoro = geracao[geracao['id_usina'].isin(id_usinas)]

  id_geracoes = []
  for _, row in usinas_mossoro.iterrows():

    if math.isnan(float(row[2])) or math.isnan(float(row[3])):
      continue

    if float(row[2]) == 0:
      continue

    id_geracoes.append(row)

  dataframe = pd.DataFrame(id_geracoes)

  dataframe.to_parquet(f'mossoro/geracao-{index}.parquet')
