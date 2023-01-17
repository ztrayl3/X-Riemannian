library(readr)
library(ggplot2)
library(eegUtils)
library(gridExtra)

# Load one of the files so we can grab the channel locations
example <- import_set("INRIA.set")
example <- electrode_locations(example, overwrite = TRUE)
chanlocs <- na.omit(example$chan_info)

# Load all available LIME files
p <- "csv/MS_DL_B/"
file_names <- paste(p, dir(p), sep = '')  # list all csv files
source <- do.call(rbind, lapply(file_names, read_csv, col_types = cols(...1 = col_skip())))  # load them into one DF

# Collect channel importance by condition
p <- list()  # for capturing plots
i <- 1
for (hand in names(table(source$Predicted))) {
  # Extract data
  temp <- subset(source, source$Predicted==hand)  # select one hand
  temp.chans <- aggregate(Weight ~ Channel, data = temp, sum)  # sum channel importances across all timepoints
  temp.chans$Weight <- scale(temp.chans$Weight)  # normalize (mean 0, SD 1)

  # Plot topographical feature importance (across ALL subjects)
  names(temp.chans) <- c("electrode", "amplitude")
  p[[i]] <- topoplot(temp.chans, chanLocs = chanlocs, contour = FALSE, interp_limit = "head",
                     chan_marker = "name") + ggtitle(hand)
  p[[i]]$guides$fill$title <- "Feature Importance"
  i <- i + 1
}

do.call(grid.arrange, c(p, ncol=1, top = "Channel Importance (normalized), separated by predicted class (top = left, bottom = right)"))
