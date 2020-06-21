library(tidyverse)

rm(list = ls(all.names = TRUE));

logfile_pred = "E:/ashwinWork/RNN_StockPrediction/stock-prediction-BrownHatMod/reinforcement/MU/logs/logfile_pred.csv";
logfile_test = "E:/ashwinWork/RNN_StockPrediction/stock-prediction-BrownHatMod/reinforcement/MU/logs/logfile_test.csv";
logfile_TEMA = "E:/ashwinWork/RNN_StockPrediction/stock-prediction-BrownHatMod/reinforcement/MU/logs/logfile_TEMA.csv";

predData <- read_csv(logfile_pred, col_names = TRUE);
testData <- read_csv(logfile_test, col_names = TRUE);
TEMAData <- read_csv(logfile_TEMA, col_names = TRUE);


# Line 1
plot(testData$x_test[1:length(predData$x_testPred)], testData$y_test[1:length(predData$x_testPred)], type = "b", frame = FALSE, pch = 20,
     col = "blue", xlab = "x", ylab = "y")

# Line 2
lines(predData$x_testPred, predData$y_predArr, pch = 20, col = "red", type = "b", lty = 2)


# Line 3
lines(TEMAData$x_TEMA,TEMAData$TEMAList, pch = 20, col = "magenta", type = "b", lty = 2)

# Add a legend to the plot
legend("topleft", legend=c("testData","predData","TEMAData"),
       col=c("blue","red","magenta"), lty = 1:2, cex=0.8)