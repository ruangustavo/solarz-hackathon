from _shared.path import path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
CIDADE = "Mossoró"

data = pd.read_csv(path("usinas_power_range_mossoro.csv"))
most_common_range = data["power_range"].value_counts().idxmax()
filtered_data = data[data["power_range"] == most_common_range]

total_plants = len(filtered_data)
anomalous_plants = filtered_data["anomalous"].sum()
anomalous_percentage = (anomalous_plants / total_plants) * 100

sns.set_theme(style="whitegrid")

plt.figure(figsize=(14, 7))
scatter_plot = sns.scatterplot(
    data=filtered_data,
    x="potencia",
    y="quantidade",
    hue="anomalous",
    palette={True: "red", False: "blue"},
    s=60,
    alpha=0.8,
)

plt.plot(
    filtered_data["potencia"],
    filtered_data["media_esperada"],
    color="green",
    linestyle="--",
    linewidth=2,
    label="Média Esperada",
)

plt.title(
    f"Análise de Usinas Solares no Intervalo Mais Frequente: {most_common_range}",
    fontsize=18,
)
plt.xlabel("Potência Instalada (kWp)", fontsize=14)
plt.ylabel("Geração Diária (kWh)", fontsize=14)
plt.legend(
    loc="upper right", fontsize=12, labels=["Normal", "Anomalies", "Média Esperada"]
)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

summary_text = (
    f"Total de Usinas: {total_plants}\n"
    f"Usinas Anômalas: {anomalous_plants} ({anomalous_percentage:.2f}%)"
)
plt.text(
    0.02,
    0.98,
    summary_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    color="black",
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
)
plt.savefig(path("scatter_plot.png"))
plt.show()
