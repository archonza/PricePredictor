import pandas as pd

# Read your CSV file into a DataFrame
df = pd.read_csv('./resources/bitcoin_csv.csv')

# Create a dictionary to store correlations
correlation_dict = {}

# Filter out non-numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Loop through each numeric column
for column1 in numeric_columns:
    for column2 in numeric_columns:
        if column1 != column2:
            # Calculate correlation between the two numeric columns
            correlation = df[column1].corr(df[column2])
            correlation_dict[f'{column1} - {column2}'] = correlation

# Print correlations to a file
with open('./output/correlations.txt', 'w') as file:
    for column_pair, correlation in correlation_dict.items():
        file.write(f'{column_pair}: {correlation}\n')

print('Correlations written to correlations.txt')
