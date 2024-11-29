import pandas as pd

generation = pd.read_csv(
    "../_data/geracao_mossoro.csv"
)  # contains id_usina, data, quantidade, prognostico
usina_history = pd.read_csv(
    "../_data/usina_historico.csv"
)  # contains power, plant_id, start_date


def calculate_expected_generation(start_year, start_prognostico, years_active):
    expected_generation = {}
    current_generation = start_prognostico

    for year in range(years_active):
        if year == 0:
            expected_generation[start_year] = current_generation
        elif year == 1:
            current_generation *= 1 - 0.025  # 2.5% first year
            expected_generation[start_year + year] = current_generation
        else:
            current_generation *= 1 - 0.005  # 0.5% subsequent years
            expected_generation[start_year + year] = current_generation
    return expected_generation


def process_generation_data(generation, usina_history):
    merged = pd.merge(
        generation, usina_history, left_on="id_usina", right_on="plant_id"
    )

    # Convert dates and add month column
    merged["data"] = pd.to_datetime(merged["data"], errors="coerce")
    merged["start_date"] = pd.to_datetime(merged["start_date"], errors="coerce")
    merged["year"] = merged["data"].dt.year.astype(int)
    merged["month"] = merged["data"].dt.month

    # Calculate monthly averages for each plant/year
    monthly_avg = (
        merged.groupby(["plant_id", "year", "month"])["quantidade"].mean().reset_index()
    )
    yearly_data = (
        monthly_avg.groupby(["plant_id", "year"])["quantidade"].mean().reset_index()
    )

    results = []
    for plant_id in yearly_data["plant_id"].unique():
        plant_data = merged[merged["plant_id"] == plant_id].iloc[0]
        start_year = plant_data["start_date"].year

        # Use prognostico as baseline
        start_prognostico = (
            merged[(merged["plant_id"] == plant_id) & (merged["year"] == start_year)][
                "prognostico"
            ].mean()
            * 12
        )  # Convert to annual equivalent

        plant_years = yearly_data[yearly_data["plant_id"] == plant_id]

        for _, row in plant_years.iterrows():
            year = row["year"]
            actual_generation = row["quantidade"] * 12  # Convert to annual equivalent

            years_active = year - start_year + 1
            expected_by_year = calculate_expected_generation(
                start_year, start_prognostico, years_active
            )

            expected_for_year = expected_by_year.get(year, 0)
            deficit = expected_for_year - actual_generation

            # Calculate variance-based tolerance
            plant_variance = merged[merged["plant_id"] == plant_id]["quantidade"].std()
            base_tolerance = 0.15  # 15% base tolerance
            variance_factor = min(
                plant_variance / expected_for_year, 0.10
            )  # Max 10% additional
            tolerance = (base_tolerance + variance_factor) * expected_for_year

            if abs(deficit) <= tolerance:
                status = "Normal"
            elif actual_generation > (expected_for_year + tolerance):
                status = "Outlier (High)"
            elif actual_generation < (expected_for_year - tolerance):
                status = "Outlier (Low)"

            results.append(
                {
                    "plant_id": plant_id,
                    "year": year,
                    "actual_generation": actual_generation,
                    "expected_generation": expected_for_year,
                    "deficit": deficit,
                    "tolerance_percentage": (base_tolerance + variance_factor) * 100,
                    "status": status,
                }
            )

    return pd.DataFrame(results)


results = process_generation_data(generation, usina_history)
print(results.shape[0])
print(
    results[
        (results["status"] == "Outlier (High)") | (results["status"] == "Outlier (Low)")
    ].shape[0]
)
