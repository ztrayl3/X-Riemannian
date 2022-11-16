library(readr)
library(ggplot2)
library(plotly)

source <- read_csv("A18_MDM.csv", col_types = cols(...1 = col_skip()))

# Look at timeseries importance
Timewise <- aggregate(source[, 4], list(source$Time), sum)
ggplotly(
  ggplot(Timewise, aes(x=Group.1, y=Weight)) +
    geom_area(fill="#69b3a2", alpha=0.5) +
    geom_line(color="#69b3a2") +
    ylab("Feature Weight (LIME)") +
    xlab("Time (in EEG samples)")
)

# Look at channel importance
Channelwise <- aggregate(source[, 4], list(source$Channel), sum)
Channelwise <- Channelwise[order(Channelwise$Group.1, decreasing = TRUE),]

ggplotly(
  ggplot(data=Channelwise, aes(x=reorder(Group.1, -Weight), y=Weight)) +
    geom_bar(stat="identity") + 
    xlab("Channel")
)

# Look at left and right, overlaid
Timewise.left <- subset(source, source$Predicted == "left")
Timewise.left <- aggregate(Timewise.left[, 4], list(Timewise.left$Time), sum)
Timewise.right <- subset(source, source$Predicted == "right")
Timewise.right <- aggregate(Timewise.right[, 4], list(Timewise.right$Time), sum)

ggplotly(
  ggplot(Timewise.left, aes(x=Group.1, y=Weight)) +
  geom_line(color="#69b3a2") +
  geom_line(data = Timewise.right, aes(x=Group.1, y=Weight)) +
  ylab("Feature Weight (LIME)") +
  xlab("Time (in EEG samples)")
)
