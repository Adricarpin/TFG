library(tfarima)


# Read data
readtfg <- function(url, name){
  df <- read.table(url, 
                   header = T, sep=",", col.names = c("date", name))
  rownames(df) <- df$date 
  df$date <- NULL 
  return(df)
}

df1 <- readtfg("precios_14_20.txt", "price")
df2 <- readtfg("eolica_14_20.csv", "wind")
df3 <- readtfg("demanda_14_20.csv", "demand")

demand <- ts(df3[47449:52584,]) #June - December
demand_test <- ts(df3[52585:61368,])#2020

wind <- ts(df2[47449:52584,]) 
wind_test <- ts(df2[52585:61368,]) 

price <- ts(df1[47449:52584,])
price_test <- ts(df1[52585:61368,]) 


# Demand

umx1 <- um(demand, ar = list(1, "(1:12)/24"), i = list(1, c(1, 24), c(1, 168)), ma = "(1 - 0.1B24)(1-0.1B168)")
umx1
summary(umx1)
umx1$ar

diagchk(umx1)


# Predictions for demand
n <- length(demand)
demand2 <- ts(df3[47449:61368,]) #train + test
demandf <- sapply(1:366, function(x) {
  ori <- n + (x-1)*24
  p <- predict(umx1, demand2, ori = ori, n.ahead = 24)
  p$z[(ori+1):(ori+24)]
})
save(demandf, file = "demandf.Rda")
load("demandf.Rda")



# Wind

ide(wind, transf = list(d = 1))
umx2 <- um(wind, ar = list("(1-0.5B)", "(1:12)/24"), i = list(1, c(1, 24)), ma = "(1 - 0.1B24)")
umx2
summary(umx2)
umx2$ar
diagchk(umx2)


# Predictions for wind
n <- length(wind)
wind2 <- ts(df2[47449:61368,])
windf <- sapply(1:366, function(x) {
  ori <- n + (x-1)*24
  p <- predict(umx2, wind2, ori = ori, n.ahead = 24)
  p$z[(ori+1):(ori+24)]
})
windf
save(windf, file = "windf.Rda")
load("windf.Rda")


# TRANSFER FUNCTION MODEL

# Univariate model
umy <- um(price, ar = list(1, "(1:12)/24"), i = list(1, c(1, 24), c(1, 168)), ma = "(1 - 0.8B24)(1-0.7B168)")


y  <- residuals(umy)
x1 <- residuals(umx1)
x2 <- residuals(umx2)


# Cross-correlation functions
pccf(x1, y)
pccf(x2, y)
pccf(x1, x2)


X1 <- c(demand, as.vector(demandf)) 
tfx1 <- tf(X1, w0 = 0.0012) 


X2 <- c(wind, as.vector(windf))
tfx2 <- tf(X2, w0 = -0.0013, ar = "1 - 0.3B")



tfm1 <- tfm(inputs = list(tfx1, tfx2), noise = umy)
u <- residuals(tfm1)

diagchk(tfm1, lags.at= c(24, 168))

umu <- um(u, ma = c(2, 168))
diagchk(umu, lags.at= 24)


tfm2 <- modify(tfm1, ma = "1 - 0.1447080B168-0.1222818B336")


diagchk(tfm2, lags.at= 24)


# Predict 
n <- length(price)
Yf <- ts(df1[47449:61368,])
predictions <- sapply(1:366, function(x) {
  ori <- n + (x-1)*24
  p <- predict(tfm2, y = Yf, ori = ori, n.ahead = 24)
  p$z[(ori+1):(ori+24)]
})



length(price_test)
length(predictions)


write.csv(predictions,"TF.csv", row.names = FALSE)


