#!/bin/bash

# Define the array of gauss_sigma values
sigma_values=(0.1 0.5 1 1.5 2 2.5 3 3.5)

# Loop through each gauss_sigma value
for gs in "${sigma_values[@]}"
do
    echo "Running with gauss_sigma = $gs"
    python3 BiopTort.py --im_path /home/jackson/research/data/tortuosity_study/tortuosity_training_set \
                        --mask_path /home/jackson/research/data/tortuosity_study/QC_results/training_set_final_run/usable_tissue_masks \
                        --gt_path /home/jackson/research/data/tortuosity_study/tortuosity_training_set_pngs \
                        --ppt_save "/home/jackson/research/data/tortuosity_study/powerpoints/training_set/gs_${gs}.pptx" \
                        --csv_save "/home/jackson/research/data/tortuosity_study/powerpoints/training_set/gs_${gs}.csv" \
                        -s --gauss_sigma $gs
done

echo "All runs completed."
