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
source <- LIME_load("csv/KernelTest/")

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
    temp.chans$Weight <- scale(temp.chans$Weight)  # normalize (mean 0, SD 1)
    
    if (adjust) {
      # adjust feature weights based on classifier accuracy
      for (row in 1:nrow(temp.chans)) {
        if (accs[row, ]$Accuracy < 0.65) {
          temp.chans[row, ]$Weight <- temp.chans[row, ]$Weight * 10
        } else {
          temp.chans[row, ]$Weight <- temp.chans[row, ]$Weight / 10
        }
      }
    }
    
    # Plot topographical feature importance (across ALL subjects)
    names(temp.chans) <- c("electrode", "amplitude")
    p[[i]] <- topoplot(temp.chans, chanLocs = chanlocs, contour = FALSE, interp_limit = "head",
                       chan_marker = "name") + ggtitle(t)
    p[[i]]$guides$fill$title <- "Feature Importance"
    i <- i + 1
  }
}
do.call(grid.arrange, c(p, ncol=4, top = "Channel Importance (normalized), separated by predicted class (top = left, bottom = right)"))
