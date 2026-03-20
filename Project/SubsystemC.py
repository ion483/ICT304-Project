import pandas as pd
import numpy as np
import csv

safety_buffers = {
    0: 10,  # Frontdoor keys
    1: 25,  # Phone
    2: 18,  # Toothpaste
    3: 20   # Wrist watch
}

forecast_df = None

try:
    forecast_df = pd.read_csv("ForecastReport.csv")
    print("--- Subsystem C: Processing Reorders ---")
except FileNotFoundError:
    print("Error: ForecastReport.csv not found. Run Subsystem B first.")
    exit()


reorder_results = []


for _, row in forecast_df.iterrows():
    item_id = int(row['item_id'])
    item_name = row['item_name']
    current_qty = int(row['current_stock'])
    three_day_demand = int(row['sum'])

    buffer = safety_buffers.get(item_id, 10)

    remaining_after_3_days = current_qty - three_day_demand

    reorder_count = buffer - remaining_after_3_days

    status = None

    if reorder_count <= 0:
        status = "Good"
    elif reorder_count <= buffer:
        status = "Warning"
    elif reorder_count > buffer:
        status = "Critical"

    reorder_results.append({
        'item_id': item_id,
        'item_name': item_name,
        'remaining_after_3_days': remaining_after_3_days,
        'reorder_needed': max(0, reorder_count), # Don't show negative reorders
        'status': status
    })


subsystemC_report = pd.DataFrame(reorder_results)
subsystemC_report.to_csv("FinalReport.csv", index=False)

print("\n--- Reorder Analysis Complete ---")
print(subsystemC_report[['item_name', 'remaining_after_3_days', 'reorder_needed', 'status']])
print("\nReorderReport.csv has been updated.")


