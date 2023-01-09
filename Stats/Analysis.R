library(readr)
library(ggplot2)
library(plotly)

source <- read_csv("A1_16730323238023405.csv", col_types = cols(...1 = col_skip()))

# Simple analysis of classifier accuracy
ggplot(source, aes(fill=Classifier, y=Accuracy, x=Subject)) + 
  geom_bar(position="dodge", stat="identity")

channels <- aggregate(Weight ~ Channel, data = source, sum)[order(-channels$Weight),]
channels$Channel <- factor(channels$Channel, levels = channels$Channel)
ggplot(channels, aes(y=Weight, x=Channel)) + 
  geom_bar(stat="identity")
