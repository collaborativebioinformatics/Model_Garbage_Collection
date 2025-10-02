import json
import csv

# Simple script to convert alzheimers_subgraph.json triples to CSV
input_path = "data/alzheimers_subgraph.json"
output_path = "data/alzheimers_triples.csv"

with open(input_path, "r") as f:
    data = json.load(f)

with open(output_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["subject", "predicate", "object"])

    # Extract triples from results[0].data[].row
    for item in data["results"][0]["data"]:
        subject, predicate, object_id = item["row"]
        writer.writerow([subject, predicate, object_id])

print(f"Triples written to {output_path}")
