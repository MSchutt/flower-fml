import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Training time and fraction fit

df = pd.read_csv('../results.csv')

# Plot the average traintime for fraction_fit
aggregated = df.groupby(['distribution']).mean().reset_index()

fig, ax = plt.subplots(1, 2)
fig.tight_layout()

sns.set_palette("pastel")
sns.barplot(x='distribution', y='traintime', data=aggregated, ax=ax[0], color='lightblue')
# Show concrete values for each barplot
for p in ax[0].patches:
    ax[0].annotate(f'{p.get_height():.0f}s', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                   xytext=(0, 6), textcoords='offset points')
ax[0].set_xlabel('Client Distribution enabled')
ax[0].set_ylabel('Training time [s]')

sns.barplot(x='distribution', y='r2', data=aggregated, ax=ax[1], color='lightblue')
# Show concrete values for each barplot
for p in ax[1].patches:
    ax[1].annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                   xytext=(0, 6), textcoords='offset points')
ax[1].set_xlabel('Client Distribution enabled')
ax[1].set_ylabel('R2 score')

plt.suptitle('Client distribution enabled compared to train time and R2 Score', fontsize=16)

# suptitle with bottom margin and bigger font
plt.gcf().subplots_adjust(top=0.93)

plt.show()
