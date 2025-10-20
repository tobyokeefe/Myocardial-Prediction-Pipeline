import pandas as pd

# Load the CSV file
df = pd.read_csv("ensemble_predictions_new_data.csv")

# Extract only the Predicted_MI column
predicted_mi = df[["Predicted_MI"]]

# Save to a new CSV file
predicted_mi.to_csv("predicted_MI_only.csv", index=False)

print("New CSV file 'predicted_MI_only.csv' created successfully.")