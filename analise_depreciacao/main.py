import pandas as pd
from _shared.path import path

def remove_milliseconds(date_str):
    if pd.isna(date_str):
        return date_str
    if "." in date_str:
        date_str = date_str.split(".")[0]
    return date_str


generation = pd.read_csv(path("geracao_mossoro.csv"))
usina_history = pd.read_csv(path("usina_historico.csv"))


def calculate_expected_generation_year_by_year(
    actual_first_year, depreciation_rate, years_active
):
    expected_generation = {}
    current_generation = actual_first_year

    for year in range(years_active):
        if year == 0:
            depreciation_rate = 0
        elif year == 1:
            depreciation_rate = 0.025
        else:
            depreciation_rate = 0.005
        current_generation *= 1 - depreciation_rate
        expected_generation[year] = current_generation

    return expected_generation


def process_generation_data_optimized(generation, usina_history):

    merged = pd.merge(
        generation, usina_history, left_on="id_usina", right_on="plant_id"
    )

    merged["data"] = pd.to_datetime(merged["data"], errors="coerce")
    merged["start_date"] = pd.to_datetime(merged["start_date"], errors="coerce")

    merged["year"] = merged["data"].dt.year
    grouped_plants = merged.groupby("plant_id")

    results = []

    for plant_id, plant_group in grouped_plants:
        plant_group = plant_group.sort_values("data")
        start_date = plant_group["start_date"].iloc[0]
        start_year = start_date.year

        actual_first_year = plant_group[plant_group["year"] == start_year][
            "quantidade"
        ].sum()

        if actual_first_year <= 0:
            print(
                f"Skipping plant {plant_id} due to invalid actual generation in the first year."
            )
            continue

        actual_first_year_prognostico = plant_group[plant_group["year"] == start_year][
            "prognostico"
        ].sum()

        if actual_first_year_prognostico <= 0:
            print(
                f"Skipping plant {plant_id} due to invalid actual generation prognostico in the first year."
            )
            continue

        years_active = plant_group["year"].max() - start_year + 1

        expected_by_year = calculate_expected_generation_year_by_year(
            actual_first_year, 0.025, years_active
        )

        years = list(plant_group["year"].unique())
        contador = 0

        for year, year_group in plant_group.groupby("year"):

            actual_generation = year_group["quantidade"].sum()
            actual_prognostico_generation = year_group["prognostico"].sum()

            years_since_start = year - start_year
            expected_for_year = expected_by_year.get(years_since_start, 0)
            deficit = expected_for_year - actual_generation

            current_year = years[contador]
            last_year_generation = actual_first_year

            if contador != 0:
                last_year_generation = plant_group[
                    plant_group["year"] == current_year - 1
                ]["quantidade"].sum()

            if actual_prognostico_generation < (actual_first_year_prognostico * 0.95):
                if actual_generation < (actual_prognostico_generation * 0.95):
                    status = "Problema"
                else:
                    status = "Normal 1"
                continue

            if actual_prognostico_generation > (actual_first_year_prognostico * 1.05):
                if actual_generation < (last_year_generation * 0.95):
                    status = "Problema"
                else:
                    status = "Normal 2"
                continue

            if years_since_start == 0 and deficit > (0.025 * expected_for_year):
                status = "Underperformance (Year 1)"
            elif years_since_start > 0 and deficit > (0.005 * expected_for_year):
                status = "Underperformance (Subsequent Years)"
            else:
                status = "Normal"

            results.append(
                {
                    "plant_id": plant_id,
                    "year": year,
                    "actual_generation": actual_generation,
                    "actual_prognostico_generation": actual_prognostico_generation,
                    "last_year_generation": last_year_generation,
                    "expected_generation": expected_for_year,
                    "deficit": deficit,
                    "status": status,
                }
            )
            contador += 1

    return pd.DataFrame(results)


results = process_generation_data_optimized(generation, usina_history)
results.to_csv(path("results_analise_depreciacao.csv"), index=False)

print(results.shape[0])
print(results[results["status"] == "Normal"].shape[0])
print(results[results["status"] == "Underperformance (Subsequent Years)"].shape[0])
print(results[results["status"] == "Underperformance (Year 1)"].shape[0])
