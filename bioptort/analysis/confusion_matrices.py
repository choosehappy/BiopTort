#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from bioptort.analysis.constants import RATER_NAMES


r1_name = RATER_NAMES[0]
r2_name = RATER_NAMES[1]
r3_name = RATER_NAMES[2]
# df1 = pd.read_csv("/media/jackson/backup/dp_data/tortuosity_study/read_csvs/score_tables/r4_scores.csv")
# df1 = pd.read_csv("/media/jackson/backup/dp_data/tortuosity_study/read_csvs/score_tables/clinically_relevant_measures/r2_XY_tort_no_duplicates_sufficient_tissue.csv")
# df1 = pd.read_csv('/media/jackson/backup/dp_data/tortuosity_study/read_csvs/score_tables/clinically_relevant_measures/r4_XY_tort.csv')
# df1 = pd.read_csv('/media/jackson/backup/dp_data/tortuosity_study/read_csvs/score_tables/clinically_relevant_measures/r2_Z_tort_no_duplicates_sufficient_tissue.csv')
df1 = pd.read_csv('/media/jackson/backup/dp_data/tortuosity_study/read_csvs/score_tables/clinically_relevant_measures/r4_Z_tort.csv')
r3_scores = df1[r3_name].to_numpy()
r2_scores = df1[r2_name].to_numpy()
r1_scores = df1[r1_name].to_numpy()
bt = df1["BT pred_grade"].to_numpy()

# -------- PARAMS -------- #
scores = {7: r1_scores, 9: r3_scores, 12: r2_scores, 1: bt}
# labels = [1,2,3,4]
# grade_names = ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]
# grade_names = ["low", "medium", "high"]

labels = [1, 2]
grade_names = ["Low XY Tortuosity", "High XY Tortuosity"]
grade_names = ["Low Z Tortuosity", "High Z Tortuosity"]


read_name = "3"
# ------------------------ #
# %%
combs = [(7,9), (12,7), (9,12), (1, 7), (1, 9), (1, 12)]
users = {"7": r1_name, "9": r3_name, "12": r2_name, "1": "BiopTort"}
for i in range(len(combs)):
    cf = confusion_matrix(scores[combs[i][0]], scores[combs[i][1]],)
    disp = ConfusionMatrixDisplay(cf, display_labels=labels)
    disp.plot()
    plt.xlabel(users[str(combs[i][1])])
    plt.ylabel(users[str(combs[i][0])])
plt.show()

# compute inter annotator agreement
for i in range(len(combs)):
    user1 = users[str(combs[i][0])]
    user2 = users[str(combs[i][1])]
    kappa_score = cohen_kappa_score(scores[combs[i][0]], scores[combs[i][1]])
    print(f"Cohen's kappa score between {user1} and {user2}: {kappa_score}")

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
