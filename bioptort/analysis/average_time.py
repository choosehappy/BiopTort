#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PathCollection
import numpy as np

from bioptort.analysis.constants import RATER_NAMES

# csv_path = "/media/jackson/backup/dp_data/tortuosity_study/read_csvs/score_tables/r1_times_no_duplicates_sufficient_tissue.csv"
# csv_path2 = "/media/jackson/backup/dp_data/tortuosity_study/read_csvs/score_tables/r2_times_no_duplicates_sufficient_tissue.csv"
# csv_path3 = "/media/jackson/backup/dp_data/tortuosity_study/read_csvs/score_tables/r4_times.csv"

csv_path = "/media/jackson/SOMAI_backup_jja/tortuosity_study/read_csvs/score_tables/r1_times_no_duplicates_sufficient_tissue.csv"
csv_path2 = "/media/jackson/SOMAI_backup_jja/tortuosity_study/read_csvs/score_tables/r2_times_no_duplicates_sufficient_tissue.csv"
csv_path3 = "/media/jackson/SOMAI_backup_jja/tortuosity_study/read_csvs/score_tables/r4_times.csv"

#%%
df1 = pd.read_csv(csv_path)[RATER_NAMES]
df2 = pd.read_csv(csv_path2)[RATER_NAMES]
df3 = pd.read_csv(csv_path3)[RATER_NAMES]

df1['Read'] = 'Read 1'
df2['Read'] = 'Read 2'
df3['Read'] = 'Read 3'
sns.set_theme('notebook', 'whitegrid')
combined_df = pd.concat([df1, df2, df3])
melted_df = pd.melt(combined_df, id_vars='Read', var_name='Rater', value_name='Scoring Time (s)')
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_yscale('log', base=2)
# ax.set_xticklabels(['Read 1', 'Read 2', 'Read 3'])

sns.violinplot(x='Read', y='Scoring Time (s)', hue='Rater', data=melted_df, split=False, cut=0)
for artist in ax.lines:
    artist.set_zorder(10)
for artist in ax.findobj(PathCollection):
    artist.set_zorder(11)
sns.stripplot(x='Read', y='Scoring Time (s)', hue='Rater', data=melted_df, dodge=True, jitter=True, edgecolor='black', linewidth=1, size=2, legend=False)
# sns.swarmplot(x='Read', y='Scoring Time (s)', hue='Rater', data=melted_df, dodge=True, edgecolor='black', linewidth=1, size=2, legend=False)

plt.title('Scoring Time Distrbutions for Each Read')

custom_labels = ['Rater 1', 'Rater 2', 'Rater 3']
handles, _ = plt.gca().get_legend_handles_labels()
plt.legend(handles[:3], custom_labels)

plt.savefig('figures/time_dists.png', dpi=300)
plt.show()


#%%
paths = [csv_path, csv_path2, csv_path3]
read_names = ["r1", "r2", "r3"]

for i, path in enumerate(paths):
    print(f"Read: {read_names[i]} --------------")
    df = pd.read_csv(path)
    all_times = df[RATER_NAMES].to_numpy().flatten()
    
    for reader in RATER_NAMES:
        arr = df[reader].to_numpy()
        print(f"Reader: {reader}, mean: {arr.mean():.2f}, std: {arr.std():.2f}")
    
    print(f'All readers mean: {all_times.mean():.2f} std: {all_times.std():.2f}')
    print("-----------------------------")
        


# %%

# plot on multiple axes the violin plots