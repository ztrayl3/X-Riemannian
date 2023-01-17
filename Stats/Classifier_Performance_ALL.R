library(readr)

# Load all available LIME files
a <- "csv/MS_DL_B/"
file_names_a <- paste(a, dir(a), sep = '')

b <- "csv/MS_RG_B/"
file_names_b <- paste(b, dir(b), sep = '')

file_names <- c(file_names_a, file_names_b)
source <- do.call(rbind, lapply(file_names, read_csv, col_types = cols(...1 = col_skip())))  # load them into one DF

# Plot classifier accuracy (per subject)
ggplot(source, aes(fill=Classifier, y=Accuracy, x=Subject)) + 
  geom_bar(position="dodge", stat="identity")