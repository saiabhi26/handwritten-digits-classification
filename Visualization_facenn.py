import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('results2.csv')

# Pivot the dataframe to create a 2D grid for hidden nodes (x-axis) and regularization (y-axis)
pivot_df = df.pivot(index = "regularization hyperparameter", columns = "hidden nodes", values = "test accuracy")

# Plotting the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
plt.title('Test Accuracy Heatmap')
plt.xlabel('Number of Hidden Nodes')
plt.ylabel('Regularization Hyperparameter')
plt.show()

df = pd.read_csv('results2.csv')

# Pivot the dataframe to create a 2D grid for hidden nodes (x-axis) and regularization (y-axis)
pivot_df = df.pivot(index = "regularization hyperparameter", columns = "hidden nodes", values = "time taken")

# Plotting the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
plt.title('Time taken Heatmap')
plt.xlabel('Number of Hidden Nodes')
plt.ylabel('Regularization Hyperparameter')
plt.show()


# Load the dataset
df = pd.read_csv('results2.csv')

# Get unique values of hidden nodes and regularization hyperparameters
hidden_nodes_values = df['hidden nodes'].unique()
regularization_values = df['regularization hyperparameter'].unique()

# Iterate over hidden node values and create a separate plot for each
for hidden_node in hidden_nodes_values:
    # Subset the data for the current hidden node value
    subset = df[df['hidden nodes'] == hidden_node]

    # Create a new figure for each hidden node
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Create bar plot for accuracy
    ax1.bar(subset['regularization hyperparameter'], subset['test accuracy'], alpha=0.6, label='Test Accuracy', color='blue')

    # Create a secondary y-axis for time taken
    ax2 = ax1.twinx()
    ax2.plot(subset['regularization hyperparameter'], subset['time taken'], marker='o', label='Time Taken', color='red')

    # Set titles and labels
    ax1.set_title(f'Accuracy and Time Taken vs Regularization Hyperparameter for Hidden Nodes = {hidden_node}')
    ax1.set_xlabel('Regularization Hyperparameter')
    ax1.set_ylabel('Test Accuracy', color='blue')
    ax2.set_ylabel('Time Taken (seconds)', color='red')

    # Add grid, legends, and labels
    ax1.grid(True)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')

    # Legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Adjust layout to ensure everything fits nicely
    plt.tight_layout()

    # Show the plot
    plt.show()
