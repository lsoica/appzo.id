import pandas as pd

# Load the logs from the CSV file into a DataFrame
input_file = 'loki_logs_last_hour.csv'
logs_df = pd.read_csv(input_file)

# Initialize lists to store anomalies and normal logs
anomalies = []
normal_logs = []

# Iterate over each row in the DataFrame
for index, row in logs_df.iterrows():
    log_message = row['log']  # Assuming the log messages are in a column named 'log'

    # Check if the log message contains the string "error" (case-insensitive)
    # or contains the string "opentelemetry-collector-contrib/pkg/stanza"
    if ('error' in log_message.lower()) or ('opentelemetry-collector-contrib/pkg/stanza' in log_message):
        anomalies.append(row)
    else:
        normal_logs.append(row)

# Convert lists to DataFrames
anomalies_df = pd.DataFrame(anomalies)
normal_logs_df = pd.DataFrame(normal_logs)

# Write anomalies and normal logs to separate CSV files
anomalies_output_file = 'loki_logs_last_hour_anomalies.csv'
normal_output_file = 'loki_logs_last_hour_normal.csv'

anomalies_df.to_csv(anomalies_output_file, index=False)
normal_logs_df.to_csv(normal_output_file, index=False)

print(f"Anomalies written to {anomalies_output_file}")
print(f"Normal logs written to {normal_output_file}")
