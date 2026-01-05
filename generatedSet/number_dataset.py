import json

# Datei laden
with open("C:\\ws2025\\aktuelle\\RecencyQA\\generatedSet\\recencyqa_final.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# q_id neu durchnummerieren (1, 2, 3, ...)
for i, item in enumerate(data, start=1):
    item["q_id"] = i

# Datei wieder speichern (oder neuen Namen vergeben)
with open("data_fixed.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("q_id erfolgreich von 1 bis", len(data), "neu gesetzt.")
