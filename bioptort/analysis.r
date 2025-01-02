library(irr)
library(raters)

r4_path <- "/media/jackson/backup/dp_data/tortuosity_study/read_csvs/score_tables/r4_scores.csv"
r2_path <- "/media/jackson/backup/dp_data/tortuosity_study/read_csvs/score_tables/r2_scores_no_duplicates_sufficient_tissue.csv"
r1_path <- "/media/jackson/backup/dp_data/tortuosity_study/read_csvs/score_tables/r1_no_duplicates_sufficient_tissue.csv"

# Read in the data from the csv file
data <- read.csv(uscap_path, header = TRUE)

# fleiss' Kappa
kappam.fleiss(data, exact = FALSE, detail = FALSE)
kappa(data)
