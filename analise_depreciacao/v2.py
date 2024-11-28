import pandas as pd

from _shared.path import path

import numpy as np

import matplotlib.pyplot as plt

"""
1. Resolver problema com o start date com formato invalido
A coluna start_date não está com formato invalido o que está causando a remoção de 605220 linhas

2. Analisar se a start_date está sendo a mais atual.

3. Limitar a um periodo de tempo para analisar a depreciação?

4. Analisar se o geracao_mossoro está com TODOS os dados de analise para a regiao de mossoro.
"""


generation = pd.read_csv(path('geracao_mossoro.csv'))  # contains id_usina, data, quantidade, prognostico
usina_history = pd.read_csv(path('usina_historico.csv'))  # contains power, plant_id, start_date

def calculate_expected_generation(start_year, start_prognostico, years_active):
    expected_generation = {}
    current_generation = start_prognostico
    for year in range(years_active):
        if year == 0:
            depreciation_rate = 0.025  # 2.5% for Year 1
        else:
            depreciation_rate = 0.005  # 0.5% for subsequent years
        current_generation *= (1 - depreciation_rate)
        expected_generation[start_year + year] = current_generation
    return expected_generation

def process_generation_data(generation, usina_history):
    merged = pd.merge(
        generation,
        usina_history,
        left_on='id_usina',
        right_on='plant_id'
    )

    print('Total antes da filtragem', merged.shape[0])

    # Convert dates
    merged['data'] = pd.to_datetime(merged['data'], errors='coerce')
    merged['start_date'] = pd.to_datetime(merged['start_date'], errors='coerce')

    merged = merged.dropna(subset=['start_date']) # 605220 linhas foram apagadas isto é considerável

    print('Total da filtragem', merged.shape[0])


    # Group by plant and year
    merged['year'] = merged['data'].dt.year.astype(int)
    grouped = merged.groupby(['plant_id', 'year'])
    
    # Analyze each plant
    results = []
    for (plant_id, year), group in grouped:
        actual_generation = group['quantidade'].sum()
        # expected_generation = group['prognostico'].sum()
        
        # Identify start year and initial prognostico
        plant_data = group.iloc[0]
        start_year = plant_data['start_date'].year
        start_prognostico = group[group['year'] == start_year]['prognostico'].sum()

        # Calculate expected generation based on depreciation
        years_active = year - start_year + 1
        expected_by_year = calculate_expected_generation(start_year, start_prognostico, years_active)
        
        # Identify underperformance or outliers
        expected_for_year = expected_by_year.get(year, 0)
        deficit = expected_for_year - actual_generation
        if actual_generation > expected_for_year:
            status = 'Outlier'
        elif year == start_year and deficit > (0.025 * expected_for_year):
            status = 'Underperformance (Year 1)'
        elif year > start_year and deficit > (0.005 * expected_for_year):
            status = 'Underperformance (Subsequent Years)'
        else:
            status = 'Normal'
        
        results.append({
            'plant_id': plant_id,
            'year': year,
            'actual_generation': actual_generation,
            'expected_generation': expected_for_year,
            'deficit': deficit,
            'status': status
        })
    
    return pd.DataFrame(results)

results = process_generation_data(generation, usina_history)
results.to_csv(path('results_analise_depreciacao.csv'), index=False)

print(results.shape[0])
print(results[results['status'] == 'Outlier'].count())
