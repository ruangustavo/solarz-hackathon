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
  return math.isnan(row['potencia']) or row['potencia'] <= 0 or math.isnan(row['quantidade']) or row['quantidade'] <= 0 or math.isnan(row['prognostico']) or row['prognostico'] <= 0

# Gerar o gráfico de desempenho de uma usina
ID_USINA = 18175

quantidade = [0] * 26
prognostico = [0] * 26

table = pq.read_table(
    './_data/geracao_mossoro.parquet',
    columns=['id_usina', 'data', 'quantidade', 'prognostico', 'potencia'],
    filters=[('id_usina', '==', ID_USINA)]
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


# Computar o prognóstico considerando a depreciação
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


# Computar a eficiência
anos = []
dados = []
for index in range(len(prognostico_real)):
  anos.append(index + 1999)

  if prognostico_real[index] == 0:
    dados.append(0)
  else:
    valor = quantidade[index] / prognostico_real[index]
    dados.append(float("%.2f" % valor))


# Comparar eficiência ao longo dos anos
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
plt.savefig(f"./analise_depreciacao/usinas/{'defeito' if defeituoso else 'comum'}/{ID_USINA}.png")
plt.clf()
