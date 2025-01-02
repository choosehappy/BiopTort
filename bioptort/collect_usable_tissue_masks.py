#%%
import os
from glob import glob
import shutil

#%%
output_dir = "/home/jackson/research/data/tortuosity_study/QC_results/read4_final_QC"


filepaths = glob(os.path.join(output_dir, "**/*mask_use.png"), recursive=True)

if not os.path.exists(os.path.join(output_dir, "usable_tissue_masks")):
    os.mkdir(os.path.join(output_dir, "usable_tissue_masks"))
# %%
for fp in filepaths:
    shutil.copy(fp, os.path.join(output_dir, "usable_tissue_masks", os.path.basename(fp)))
# %%
