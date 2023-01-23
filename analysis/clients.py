import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv('../results.csv')


fixed = df[(df['fraction_fit'] == 1.0) & (df['localbatchsize'] == 64) & (df['localepochs'] == 5) & (df['rounds'] == 5) & (df['distribution'] == 0)]

# Batch size
print(fixed.head())

fig, ax = plt.subplots(1, 1)
ax.set_ylim((0.75, 1.0))
fig.tight_layout()

sns.barplot(x='clients', y='r2', data=fixed, ax=ax, color='lightblue')
ax.set_xlabel("Number of clients")
ax.set_ylabel("R2")
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                   xytext=(0, 6), textcoords='offset points')
ax.set_title("Number of Clients compared to R2")

# Show concrete values for each barplot
plt.show()