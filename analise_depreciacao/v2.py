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

    merged['start_date_raw'] = merged['start_date']

    # Convert dates
    merged['data'] = pd.to_datetime(merged['data'], errors='coerce')
    merged['start_date'] = pd.to_datetime(merged['start_date'], errors='coerce')

    mask_nat = merged['start_date'].isna()
    merged.loc[mask_nat, 'start_date_cleaned'] = merged.loc[mask_nat, 'start_date_raw'].apply(remove_milliseconds)
    merged.loc[mask_nat, 'start_date'] = pd.to_datetime(merged.loc[mask_nat, 'start_date_cleaned'], errors='coerce')

    # Group by plant and year
    merged['year'] = merged['data'].dt.year.astype(int)
    grouped_plants = merged.groupby('plant_id')

    
    # Analyze each plant
    results = []
    for plant_id, plant_group in grouped_plants:
        # Sort the plant data by date to ensure chronological order
        plant_group = plant_group.sort_values('data')

        # Get the start date and start year of the plant
        plant_start_data = plant_group.iloc[0]
        start_date = plant_start_data['start_date']
        start_year = start_date.year

        # Sum 'prognostico' for the start year
        start_prognostico = plant_group[plant_group['year'] == start_year]['prognostico'].sum()

        if pd.isna(start_prognostico) or start_prognostico == 0:
            print(f"Invalid 'start_prognostico' for plant_id {plant_id} in start year {start_year}. Skipping this plant.")
            continue

        # Process each year for the plant
        for year, year_group in plant_group.groupby('year'):
            actual_generation = year_group['quantidade'].sum()

            # Calculate expected generation based on depreciation
            years_active = year - start_year + 1
            expected_by_year = calculate_expected_generation(start_year, start_prognostico, years_active)

            # Expected generation for current year
            expected_for_year = expected_by_year.get(year, 0)
            deficit = expected_for_year - actual_generation

            # Determine status
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
print(results[results['status'] == 'Outlier'].shape[0])
