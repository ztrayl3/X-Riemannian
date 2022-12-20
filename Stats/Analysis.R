library(readr)
library(ggplot2)
library(plotly)

source <- read_csv("A18_1671123794633043.csv", col_types = cols(...1 = col_skip()))

# Look at timeseries importance
Timewise <- data.frame(time=source$Time, weight=source$Weight)
Timewise <- aggregate(weight~time, data=Timewise, FUN=sum)
ggplotly(
  ggplot(Timewise, aes(x=time, y=weight)) +
    geom_area(fill="#69b3a2", alpha=0.5) +
    geom_line(color="#69b3a2") +
    ylab("Feature Weight (LIME)") +
    xlab("Time (in EEG samples)") + 
    geom_smooth(method = "lm", formula = y ~ poly(x, 15), se = FALSE)
)

# Look at channel importance
Channelwise <- data.frame(channel=source$Channel, weight=source$Weight)
Channelwise <- aggregate(weight~channel, data=Channelwise, FUN=sum)
Channelwise <- Channelwise[order(Channelwise$channel, decreasing = TRUE),]

ggplotly(
  ggplot(data=Channelwise, aes(x=reorder(channel, -weight), y=weight)) +
    geom_bar(stat="identity") + 
    xlab("Channel")
)

# Look at left and right, overlaid
Timewise.left <- subset(source, source$Predicted == "left")
Timewise.left <- aggregate(Timewise.left[, 5], list(Timewise.left$Time), sum)
Timewise.right <- subset(source, source$Predicted == "right")
Timewise.right <- aggregate(Timewise.right[, 5], list(Timewise.right$Time), sum)

ggplotly(
  ggplot(Timewise.left, aes(x=Group.1, y=Weight)) +
  geom_line(color="#69b3a2") +
  geom_line(data = Timewise.right, aes(x=Group.1, y=Weight)) +
  ylab("Feature Weight (LIME)") +
  xlab("Time (in EEG samples)")
)
