# 165
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
base_path = Path('../logfiles/165')

files = ['client_0.csv', 'client_1.csv', 'client_2.csv', 'client_3.csv', 'client_4.csv']

dfs = []
for f in files:
    df = pd.read_csv(base_path / f).reset_index()
    dfs.append(df)
server_df = pd.read_csv(base_path / 'server.csv').reset_index()
server_df = server_df[server_df['r2'] > 0]

print(dfs[0].columns)

for i, df in enumerate(dfs):
    print(df.columns)
    # make plots with connected lines
    sns.lineplot(x='index', y='r2_test', data=df, label=f'client {i}', linewidth=.75)

# make plots with connected lines and bold
sns.lineplot(x='index', y='r2', data=server_df, label='server', linewidth=4)
plt.xlabel('Training Rounds')
plt.ylabel('R2')
plt.title('R2 over Training Rounds (Client vs Server)')


plt.show()