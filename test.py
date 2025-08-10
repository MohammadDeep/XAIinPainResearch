import pandas as pd
df = pd.read_csv('/home/mohammad/code_PMC/XAIinPainResearch/saved_models/output_data.csv')
for subgict in range(52):
    print(df[df['subject'] == subgict].shape)
print(df.columns)
