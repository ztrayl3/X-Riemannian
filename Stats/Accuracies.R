library(readr)
library(ggplot2)
library(dplyr)

LIME_load <- function(p) {
  file_names <- paste(p, dir(p), sep = '')  # list all csv files
  # return one big DF of all csvs from the path
  return(do.call(rbind, lapply(file_names, read_csv, col_types = cols(...1 = col_skip()))))
}

# Load all available LIME files
MS_RG_Within <- LIME_load("csv/MS_RG_W/")
MS_DL_Within <- LIME_load("csv/MS_DL_W/")
MS_Within <- rbind(MS_RG_Within, MS_DL_Within)

#MS_RG_Between <- LIME_load("csv/MS_RG_B/")
#MS_DL_Between <- LIME_load("csv/MS_DL_B/")
#MS_Between <- rbind(MS_RG_Between, MS_DL_Between)

#SS_RG_Within <- LIME_load("csv/SS_RG_W/")
#SS_DL_Within <- LIME_load("csv/SS_DL_W/")
#SS_Within <- rbind(SS_RG_Within, SS_DL_Within)

#SS_RG_Between <- LIME_load("csv/SS_RG_B/")
#SS_DL_Between <- LIME_load("csv/SS_DL_B/")
#SS_Between <- rbind(SS_RG_Between, SS_DL_Between)

source <- MS_Within  # choose your condition!

# Remove unnecessary information to save memory
source$Weight <- NULL
source$Time <- NULL
source$Channel <- NULL
source$Predicted <- NULL
source$Session <- NULL
source$Condition <- NULL
source <- source[!duplicated(source), ]

# Add a column for average classifier accuracy
means <- source %>% 
  group_by(Subject) %>% 
  summarise(Accuracy = mean(Accuracy),
            Classifier = "Mean")
source <- rbind(source, means)
source$Classifier <- factor(source$Classifier, levels = c("8csp+lda", "MDM", "tangentspace+LR", "DL", "Mean"))

# Plot classifier accuracy (per subject)
ggplot(source, aes(fill=Classifier, y=Accuracy, x=Subject)) + 
  geom_bar(position="dodge", stat="identity") +
  scale_fill_manual(values=c("#FECF72",
                             "#AA99AD",
                             "#B1D2C2",
                             "#CCBBB8",
                             "#000000")) +
  geom_hline(yintercept = 0.65)
