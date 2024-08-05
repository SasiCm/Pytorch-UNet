import os
from tensorboardX import SummaryWriter
import pandas as pd

# Set the path to your results.csv and the directory for TensorBoard logs
results_csv_path = 'loss_lists.csv' 
log_dir = 'tensorboard_logfiles'

# Initialize the TensorBoard writer
writer = SummaryWriter(log_dir=log_dir)

# Load the CSV data
df = pd.read_csv(results_csv_path)

# Define the number of epochs
n_epochs = 5

# Calculate the number of rows per epoch
rows_per_epoch = len(df) // n_epochs

# Split the DataFrame into groups and calculate the mean for each epoch
means = []
for i in range(n_epochs):
    start_row = i * rows_per_epoch
    end_row = (i + 1) * rows_per_epoch
    epoch_group = df.iloc[start_row:end_row]
    mean_values = epoch_group.mean()
    means.append(mean_values)

# Create a DataFrame with the mean values for each epoch
log_df = pd.DataFrame(means, columns=df.columns)
log_df['LR'] = [0.00001] * n_epochs

print(log_df)

# Log the desired metrics to TensorBoard
for epoch in range(len(log_df)):
    loss = log_df.iloc[epoch]['Loss']
    lr = log_df.iloc[epoch]['LR']

    writer.add_scalar('Result/Loss', loss, epoch)
    writer.add_scalar('Result/LR', lr, epoch)

# Close the TensorBoard writer
writer.close()
