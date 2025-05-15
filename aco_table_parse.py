import re
import pandas as pd

with open("ant_colony_results.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

pattern = re.compile(
    r"Запуск (\d+)\nПараметры: ALPHA=(\d+\.\d+), BETA=(\d+\.\d+), RHO=(\d+\.\d+), Q=(\d+), ANTS=(\d+), ITERS=(\d+)\n"
    r"Лучший маршрут: \[.*?\]\n"
    r"Длина маршрута: (\d+)\n"
    r"Общая длительность: ([\d\.]+) секунд\n"
    r"Длительность итерации: max=([\d\.]+), min=([\d\.]+) секунд",
    re.MULTILINE
)

results = []
for match in pattern.finditer(full_text):
    results.append({
        "Run": int(match.group(1)),
        "ALPHA": float(match.group(2)),
        "BETA": float(match.group(3)),
        "RHO": float(match.group(4)),
        "Q": int(match.group(5)),
        "ANTS": int(match.group(6)),
        "ITERS": int(match.group(7)),
        "Best Length": int(match.group(8)),
        "Total Time (s)": float(match.group(9)),
        "Max Iter Time (s)": float(match.group(10)),
        "Min Iter Time (s)": float(match.group(11)),
    })

df = pd.DataFrame(results)

# Сохраняем в CSV
df.to_csv("ant_colony_summary.csv", index=False)
print("✅ Результаты сохранены в ant_colony_summary.csv")
