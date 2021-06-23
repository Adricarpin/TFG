library(tfarima)
library(Metrics)


df <- read.table("precios_14_20.txt", header = T, sep=",", col.names = c("date", "price"))
rownames(df) <- df$date 
df$date <- NULL 


# June 2019 - December 2019
Y <- df[47449:52584,] 

# 2020
test <- df[52585:61368,] 



#IDENTIFICATION



#Range-median graph 
ide(Y, transf = list(list(), list(bc = T)), graphs = c("plot", "rm") ) 

#Barlett
n <- length(Y)
nj <- 48 
k <- n %/% nj
g <- rep(1:k, each = nj)
g <- g[1:n]
length(g)


h1.test <- bartlett.test(Y, g)
h2.test <- bartlett.test(log(Y), g)
h1.test
h2.test



#ACF and PACF
ide(Y, graphs = c ("plot", "acf", "pacf"), lag.max = 216, lags.at = 24)


# Regular difference
ide(Y, transf = list(d = 1), lag.max = 216, lags.at = c(24, 168))

# 1, 24 and 168 differences
i <- um(i = "(1-B1)(1-B24)(1-B168)")$i
ide(Y, transf = list(i = i), lag.max = 216, lags.at = c(24, 168))


ma <- um(ma = "(1 - 0.6B24)(1 - 0.6B168)")
display(list(ma), graphs = c("acf", "pacf"), lag.max = 400, byrow = T)



#ESTIMATION AND DIAGNOSTIC CHECKING


#ARIMA modeling with a short time series
Y1 <- ts(window(Y, end = 24*7*8), start = c(1,1), frequency = 24)


um1 <- um(Y1, ar = "(1-0.15B)(1-0.5B24)(1-0.5B168)", i = list(1), ma = "(1-0.1B24)(1-0.1B168)")
summary(um1)

diagchk(um1, lag.max = 168*3, lags.at = c(24, 168))



um2 <- modify(um1, ar = "(1:12)/24")
um2
summary(um2)
diagchk(um2, lag.max = 168*3, lags.at = c(24, 168))


#Fitting the ARIMA model to the long time series
Y <- ts(Y, start = c(1, 1), frequency = 24)
um3 <- fit(um2, Y)
summary(um3)

diagchk(um3)



#PREDICT

n <- length(Y)
Yf <- ts(df[47449:61368,]) #all data

predictions <- sapply(1:366, function(x) {
  ori <- n + (x-1)*24
  p <- predict(um3, Yf, ori = ori, n.ahead = 24)
  p$z[(ori+1):(ori+24)]
})


write.csv(predictions,"ARIMA_PRED.csv", row.names = FALSE)
















