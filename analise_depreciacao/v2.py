import pandas as pd
from _shared.path import path

"""
(X) - 1. Resolver problema com o start date com formato invalido
A coluna start_date não está com formato invalido o que está causando a remoção de 605220 linhas

RESULTADO: Aumentou o numero de outliers.

2. Analisar se a start_date está sendo a mais atual para caso de usinas com mais de um registro ou mudança de potencia ao longo do tempo.

3. Analisar se o geracao_mossoro está com TODOS os dados de analise para a regiao de mossoro.

4. Analisar o metodo calculate_expected_generation para verificar se está correto. SIM

6. Analisar QUANDO realmente é um outlier ou não (melhorar a solução para identificar isso) talvez seja necessario utilizar o codigo de
outra frente para identificar uma media ou z-score para identificar se ta muito fora do normal.

7. Nao faz sentido marcar com outlier em alguns casos pois o deficit é positivo, ou seja, se o calculo é com base no prognostico nao da para admitir 
que ele seja totalmente preciso. é preciso com um valor percentual de erro para esse caso.
"""

def remove_milliseconds(date_str):
    if pd.isna(date_str):
        return date_str
    if '.' in date_str:
        date_str = date_str.split('.')[0]
    return date_str


generation = pd.read_csv(path('geracao_mossoro.csv'))  # contains id_usina, data, quantidade, prognostico
usina_history = pd.read_csv(path('usina_historico.csv'))  # contains power, plant_id, start_date

def calculate_expected_generation_year_by_year(actual_first_year, depreciation_rate, years_active):
    """Calculates the expected generation year by year."""
    expected_generation = {}
    current_generation = actual_first_year

    for year in range(years_active):
        if year == 0:
            depreciation_rate = 0.025  # 2.5% for 1st Year
        else:
            depreciation_rate = 0.005  # 0.5% for subsequent years
        current_generation *= (1 - depreciation_rate)
        expected_generation[year] = current_generation

    return expected_generation


def process_generation_data_optimized(generation, usina_history):
    # Merge and preprocess
    merged = pd.merge(
        generation,
        usina_history,
        left_on='id_usina',
        right_on='plant_id'
    )

    # Convert dates
    merged['data'] = pd.to_datetime(merged['data'], errors='coerce')
    merged['start_date'] = pd.to_datetime(merged['start_date'], errors='coerce')

    # Filter and prepare data
    merged['year'] = merged['data'].dt.year
    grouped_plants = merged.groupby('plant_id')

    results = []

    # Analyze each plant
    for plant_id, plant_group in grouped_plants:
        plant_group = plant_group.sort_values('data')
        start_date = plant_group['start_date'].iloc[0]
        start_year = start_date.year

        # Get actual generation for the first year
        actual_first_year = plant_group[plant_group['year'] == start_year]['quantidade'].sum()

        if actual_first_year <= 0:
            print(f"Skipping plant {plant_id} due to invalid actual generation in the first year.")
            continue

        # Calculate expected generation year by year
        years_active = plant_group['year'].max() - start_year + 1
        expected_by_year = calculate_expected_generation_year_by_year(actual_first_year, 0.025, years_active)

        # Process each year
        for year, year_group in plant_group.groupby('year'):
            actual_generation = year_group['quantidade'].sum()
            years_since_start = year - start_year

            # Expected generation for current year
            expected_for_year = expected_by_year.get(years_since_start, 0)
            deficit = expected_for_year - actual_generation

            # Determine status
            if years_since_start == 0 and deficit > (0.025 * expected_for_year):
                status = 'Underperformance (Year 1)'
            elif years_since_start > 0 and deficit > (0.005 * expected_for_year):
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

results = process_generation_data_optimized(generation, usina_history)
results.to_csv(path('results_analise_depreciacao.csv'), index=False)

print(results.shape[0])
print(results[results['status'] == 'Outlier'].shape[0])
