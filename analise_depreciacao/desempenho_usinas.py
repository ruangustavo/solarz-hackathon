import PyQt5
import math
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt


def is_invalid(row):
  return math.isnan(row['potencia']) or row['potencia'] <= 0 or math.isnan(row['quantidade']) or row['quantidade'] < 0 or math.isnan(row['prognostico']) or row['prognostico'] < 0

# Gerar o gráfico de desempenho de usinas
id_usinas_analisadas = []
TOTAL_USINAS_ANALISADAS = 0
NUMERO_USINAS = 10


def calcular_desempenho(id_usina, TOTAL_USINAS_ANALISADAS):
  quantidade = [0] * 26
  prognostico = [0] * 26
  
  table = pq.read_table(
    './_data/geracao_mossoro.parquet',
    columns=['id_usina', 'data', 'quantidade', 'prognostico', 'potencia'],
    filters=[('id_usina', '==', id_usina)]
  )
  geracao = table.to_pandas()

  for _, row in geracao.iterrows():

    if is_invalid(row):
      continue

    potencia_maxima_dia = row['potencia'] * 24

    # Detectar anomalias
    if row['prognostico'] > potencia_maxima_dia or row['quantidade'] > potencia_maxima_dia:
      continue

    ano, _, _ = row['data'].split('-')
    prognostico[int(ano) % 1999] += row['prognostico']
    quantidade[int(ano) % 1999] += row['quantidade']

  # Verificar se há dados de anos o suficiente
  anos_ativos = 0
  for valor in quantidade:
    if valor > 0:
      anos_ativos += 1
  if anos_ativos < 4:
    return TOTAL_USINAS_ANALISADAS

  first_year_index = -1
  prognostico_real = [0] * 26

  for index in range(len(prognostico)):

    if float(prognostico[index]) > 0:
      if first_year_index < 0:
        first_year_index = index

      if first_year_index == index:
        prognostico_real[index] = float(prognostico[index])
      elif first_year_index + 1 == index:
        prognostico_real[index] = float(prognostico[index]) * 0.975
      elif first_year_index + 1 < index:
        prognostico_real[index] = float(prognostico[index]) * (
          0.975 - (0.005 * (index - (first_year_index + 1)))
        )


  anos = []
  dados = []

  for index in range(len(prognostico_real)):
    anos.append(index + 1999)

    if prognostico_real[index] == 0:
      dados.append(0)
    else:
      valor = quantidade[index] / prognostico_real[index]
      dados.append(float("%.2f" % valor))


  defeituoso = False
  BAIXA_MINIMA = 0.05
  BAIXA_MAXIMA = 0.1

  for index in range(len(dados) - 3):
    contagem = 0

    desempenho_ano_atual = float(dados[index])
    desempenho_ano_atual_mais_1 = float(dados[index + 1])
    desempenho_ano_atual_mais_2 = float(dados[index + 2])
    desempenho_ano_atual_mais_3 = float(dados[index + 3])

    if desempenho_ano_atual - BAIXA_MINIMA >= desempenho_ano_atual_mais_1 and desempenho_ano_atual - BAIXA_MAXIMA < desempenho_ano_atual_mais_1:
      contagem += 1

    if desempenho_ano_atual_mais_1 - BAIXA_MINIMA >= desempenho_ano_atual_mais_2 and desempenho_ano_atual_mais_1 - BAIXA_MAXIMA < desempenho_ano_atual_mais_2:
      contagem += 1

    if desempenho_ano_atual_mais_2 - BAIXA_MINIMA >= desempenho_ano_atual_mais_3 and desempenho_ano_atual_mais_2 - BAIXA_MAXIMA < desempenho_ano_atual_mais_3:
      contagem += 1

    if contagem == 3:
      defeituoso = True

  plt.plot(anos, dados, 'ro')
  plt.grid()
  plt.savefig(f"./analise_depreciacao/usinas/{'defeito' if defeituoso else 'comum'}/{id_usina}.png")
  plt.clf()

  return TOTAL_USINAS_ANALISADAS + 1


table = pq.read_table(
  './_data/geracao_mossoro.parquet',
  columns=['id_usina']
)
geracao = table.to_pandas()

for _, row in geracao.iterrows():
  if TOTAL_USINAS_ANALISADAS == NUMERO_USINAS:
    break

  if int(row['id_usina']) not in id_usinas_analisadas:
    id_usinas_analisadas.append(int(row['id_usina']))
    TOTAL_USINAS_ANALISADAS = calcular_desempenho(int(row['id_usina']), TOTAL_USINAS_ANALISADAS)
  else:
    continue
