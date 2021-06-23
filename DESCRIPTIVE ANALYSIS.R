library(tfarima)

df <- read.table("precios_luz_14_19.txt",
                 header = TRUE, sep = "," , col.names = c("date", "price"))

head(df)

#price for  2019
p19 <- df$price[startsWith(df$date, "2019")]

#dates for 2019
d19 <- df$date[startsWith(df$date, "2019")]

#dates for the days
days <- substr(d19, 1, 10)

hours <- substr(d19, 12, 13)

#function to know the weekday 
dow <- weekdays(as.Date(days))

#days of the week
Dow <- weekdays(as.Date("2021-03-08")+0:6)

#hours of the day
h <- c(paste("0", 0:9, sep = ""), 10:23)

#matrix 365x24. prices for each hour
ph <- sapply(h, function(x) p19[hours == x])

m <- apply(ph, 2, mean)

#PLOT
plot.ts(m, xaxt = "n", ylab = "average price", xlab =  "hour", type = "o", pch = 16, cex = 0.5, main = "Day")
axis(side=1, at = 1:24, labels= h) 
abline(h=c(mean(m), mean(m)-sd(m), mean(m)-2*sd(m), mean(m)+sd(m)), col = "gray", lty = 2)


#WEEKS

#dh: prices for each hour of the week: 168 hours in a week, 52 weeks in a year
dh <- list()
for (d in Dow)
  for (h in hours[1:24])
    dh <- c(dh, list(p19[dow == d & hours == h]))

#price mean for each hour in a week
m <- sapply(dh, mean)



#PLOT
plot(m,type = "n", xaxt="n", ylab = "average price", xlab = "Day", main = "Week")
for (i in 0:6) {
  indx <- (i*24+1):((i+1)*24)
  lines(indx, m[indx])
  points(indx, m[indx], pch = 16, cex = 0.5)  
}
axis(side=1, at = 24*(1:7), labels= c("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"))


#SEASONS

#Dates for spring days
spring <- as.character(as.Date("2019-03-20")+(0:92))

#Prices for spring days
p19sp <- p19[days %in% spring]

#Name of the spring days 
dowsp <- dow[days %in% spring]

#spring hours
hourssp <- hours[days %in% spring]  

dh <- list()
for (d in Dow)
  for (h in hours[1:24])
    dh <- c(dh, list(p19sp[dowsp == d & hourssp == h]))

m <- sapply(dh, mean)


par(mfrow=c(2, 2))
plot(m,type = "n", xaxt="n", ylab = "average price", xlab = "hour", main = "Week (Spring)")
for (i in 0:6) {
  indx <- (i*24+1):((i+1)*24)
  lines(indx, m[indx])
  points(indx, m[indx], pch = 16, cex = 0.5)  
}
axis(side=1, at = 24*(1:7), labels= c("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"))



#Summer
summer <- as.character(as.Date("2019-06-21")+(0:92))
p19sm <- p19[days %in% summer]
dowsm <- dow[days %in% summer]
hourssm <- hours[days %in% summer]  

dh <- list()
for (d in Dow)
  for (h in hours[1:24])
    dh <- c(dh, list(p19sp[dowsm == d & hourssm == h]))
m <- sapply(dh, mean)

plot(m,type = "n", xaxt="n", ylab = "average price", xlab = "hour", main = "Week (Summer)")
for (i in 0:6) {
  indx <- (i*24+1):((i+1)*24)
  lines(indx, m[indx])
  points(indx, m[indx], pch = 16, cex = 0.5)  
}
axis(side=1, at = 24*(1:7), labels= c("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")) 



#Autumn
fall <- as.character(as.Date("2019-09-22")+(0:90))
p19f <- p19[days %in% fall]
dowf <- dow[days %in% fall]
hoursf <- hours[days %in% fall]  
dh <- list()
for (d in Dow)
  for (h in hours[1:24])
    dh <- c(dh, list(p19sp[dowf == d & hoursf == h]))
m <- sapply(dh, mean)
plot(m,type = "n", xaxt="n", ylab = "average price", xlab = "hour", main = "Week (Autumn)")
for (i in 0:6) {
  indx <- (i*24+1):((i+1)*24)
  lines(indx, m[indx])
  points(indx, m[indx], pch = 16, cex = 0.5)  
}
axis(side=1, at = 24*(1:7), labels= c("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")) 


#Winter
winter <- as.character(as.Date("2019-01-21")+(0:57))
winter <- c(winter,  as.character(as.Date("2019-12-21")+(0:10)))
p19w <- p19[days %in% winter]
doww <- dow[days %in% winter]
hoursw <- hours[days %in% winter]  
dh <- list()
for (d in Dow)
  for (h in hours[1:24])
    dh <- c(dh, list(p19sp[doww == d & hoursw == h]))
m <- sapply(dh, mean)
plot(m,type = "n", xaxt="n", ylab = "average price", xlab = "hour", main = "Week (Winter)")
for (i in 0:6) {
  indx <- (i*24+1):((i+1)*24)
  lines(indx, m[indx])
  points(indx, m[indx], pch = 16, cex = 0.5)  
}
axis(side=1, at = 24*(1:7), labels= c("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"))

