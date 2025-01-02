#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.metrics import cohen_kappa_score


df1 = pd.read_csv("/media/jackson/SOMAI_backup_jja/tortuosity_study/read_csvs/score_tables/r4_scores.csv")
# df2 = pd.read_csv("tuomas_r1.csv")
# df3 = pd.read_csv("xavier_r1.csv")
tilak = df1["Tilak"].to_numpy()
tuomas = df1["Tuomas"].to_numpy()
xavier = df1["Xavier"].to_numpy()

# -------- PARAMS -------- #
scores = {7: xavier, 9: tilak, 12: tuomas}
labels = [1,2,3,4]
grade_names = ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]
# grade_names = ["low", "medium", "high"]
read_name = "3"
# ------------------------ #
# %%
combs = [(7,9), (12,7), (9,12)] # (xavier, tilak), (tuomas, xavier), (tilak, tuomas). indices irrelevant
users = {"7": "Xavier", "9": "Tilak", "12": "Tuomas"}
for i in range(3):
    cf = confusion_matrix(scores[combs[i][0]], scores[combs[i][1]],)
    disp = ConfusionMatrixDisplay(cf, display_labels=labels)
    disp.plot()
    plt.xlabel(users[str(combs[i][1])])
    plt.ylabel(users[str(combs[i][0])])
plt.show()

# compute inter annotator agreement
for i in range(3):
    print(cohen_kappa_score(scores[combs[i][0]], scores[combs[i][1]]))

#%%
def plot_hist(scores, user, labels, readname=""):
    counts = np.unique(scores, return_counts=True)
    bar = plt.bar(labels, counts[1])
    percentages = return_percentages(scores)
    labels = [f"{count} ({percentage*100:.1f}%)" for count, percentage in zip(counts[1], percentages)]
    plt.bar_label(bar, labels=labels)
    # plt.bar_label(bar)
    plt.title(f'Rater {user} grade distribution, read: {readname}')

    plt.show()

def return_percentages(scores):
    counts = np.unique(scores, return_counts=True)[1]
    return counts/np.sum(counts)

# %%

for i in range(3):
    user = users[str(combs[i][0])]
    # counts, edges, bars = plt.hist(scores[combs[i][0]], bins=[1,2,3,4,5])
    # 
    grades = scores[combs[i][0]]
    percentages = return_percentages(grades)
    print(f"User {user} percentages: {percentages}")
    plot_hist(grades, f'{user}', grade_names, read_name)


all_valudes = np.concatenate(list(scores.values()))
percentages = return_percentages(all_valudes)
print(f"All grades distribution: {percentages}")
plot_hist(all_valudes, "All", grade_names, read_name)

# %%
