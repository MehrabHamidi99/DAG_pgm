import os
import json
import csv

# Base directory to start from
base_dir = '/home/mila/m/mehrab.hamidi/pgm_project/DAG_pgm/scripts/no_tears_res_er'

# List to hold all the data dictionaries
data_list = []

# Walk through the directory tree
for root, dirs, files in os.walk(base_dir):
    if 'res.txt' in files:
        # Path to the res.txt file
        res_path = os.path.join(root, 'res.txt')
        with open(res_path, 'r') as f:
            lines = f.readlines()
            # Strip whitespace and remove empty lines
            lines = [line.strip() for line in lines if line.strip()]
            if len(lines) < 2:
                print(f"File {res_path} does not have enough lines.")
                continue
            # First line contains parameters
            param_line = lines[0]
            # Last line contains results
            result_line = lines[-1]
            try:
                # Parse JSON data
                params = json.loads(param_line)
                results = json.loads(result_line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in file {res_path}: {e}")
                continue
            # Combine parameters and results into one dictionary
            combined = {**params, **results}
            # Append the combined data to the list
            data_list.append(combined)

# Get all unique field names for the CSV header
fieldnames = set()
for d in data_list:
    fieldnames.update(d.keys())
fieldnames = list(fieldnames)

# Write the collected data to a CSV file
with open('results.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in data_list:
        writer.writerow(row)

print("Data has been successfully written to results.csv")
