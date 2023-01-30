library(readr)
library(ggplot2)
library(eegUtils)
library(gridExtra)

adjust = FALSE
LIME_load <- function(p) {
  file_names <- paste(p, dir(p), sep = '')  # list all csv files
  # return one big DF of all csvs from the path
  return(do.call(rbind, lapply(file_names, read_csv, col_types = cols(...1 = col_skip()))))
}

# Load one of the files so we can grab the channel locations
example <- import_set("INRIA.set")
example <- electrode_locations(example, overwrite = TRUE)
chanlocs <- na.omit(example$chan_info)

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

# Structure:
#   +-----------------+------------------------------+------------------------------+
#   | Classifier      |          Left Hand           |          Right Hand          |
#   +-----------------+------------------------------+------------------------------+
#   | 8csp+lda        | Channel/Timepoint Importance | Channel/Timepoint Importance |
#   | MDM             | Channel/Timepoint Importance | Channel/Timepoint Importance |
#   | tangentspace+LR | Channel/Timepoint Importance | Channel/Timepoint Importance |
#   | ShallowConvNet  | Channel/Timepoint Importance | Channel/Timepoint Importance |
#   +-----------------+------------------------------+------------------------------+

# Collect channel importance by condition (see above structure)
p <- list()  # for capturing plots
i <- 1
for (hand in names(table(source$Predicted))) {
  for (classifier in names(table(source$Classifier))) {
    # Print selected combo
    t <- paste(classifier, hand, sep = " ")
    print(t)
    
    # Extract data
    temp <- subset(source, source$Predicted==hand & source$Classifier==classifier)  # select one hand - classifier combo
    temp.chans <- aggregate(Weight ~ Subject + Accuracy + Channel, data = temp, sum)  # sum channel importances across all timepoints
    accs <- data.frame(Subject=temp.chans$Subject,
                       Accuracy=temp.chans$Accuracy)  # make equal sized DF of just accuracy values
    temp.chans <- aggregate(Weight ~ Channel, data = temp, sum)  # limit data to ONLY channels and weights
    
    if (adjust) {
      # adjust feature weights based on classifier accuracy
      for (row in 1:nrow(temp.chans)) {
          temp.chans[row, ]$Weight <- temp.chans[row, ]$Weight * accs[row, ]$Accuracy
      }
    }
    temp.chans$Weight <- scale(temp.chans$Weight)  # normalize (mean 0, SD 1)
    
    # Plot topographical feature importance (across ALL subjects)
    names(temp.chans) <- c("electrode", "amplitude")
    p[[i]] <- topoplot(temp.chans, chanLocs = chanlocs, contour = FALSE, interp_limit = "head",
                       chan_marker = "name") + ggtitle(t)
    p[[i]]$guides$fill$title <- "Feature Importance"
    i <- i + 1
  }
}
do.call(grid.arrange, c(p, ncol=4, top = "Channel Importance (normalized), separated by predicted class (top = left, bottom = right)"))
