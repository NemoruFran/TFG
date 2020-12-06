#Libraries and context
library("dplyr") 
library(cluster)
library("MLmetrics")
source("GowerFunction.R")

setwd("/home/fran/TFG/RBF")

#Dataset preparation

df <- read.table("heart.csv", header=T, sep=",")

# ------- Procesado extra para testear ------------
#df <- df[-1]
#df[1,1] <- 20
# ------------- How much neurons? -----------------
dims <- dim(df)

#Approach 1 -> Poisson lambda fixada
n <- rpois(1,20)
n
#Approach 2 -> Poisson lambda depén de la quantitat màxima de dades
m <- rpois(1,dims[1]*0.05)
m

# ---------------- Which neurons? -----------------
Dissims <- GowerRBF(df) #Es una matriu triangular!!
Dissims[lower.tri(Dissims)] <- t(Dissims)[lower.tri(Dissims)] #Per completar la matriu de distancies
#Approach 1 -> distribució uniforme (random sampling)
chosen <- sample(1:dims[1],n)
chosen
#Approach 2 -> picking cluster centroids.
#clustering
dissimMatrix <- daisy(df, metric = "gower", stand=TRUE)
distMatrix <- dissimMatrix^2
h1 <- hclust(distMatrix,method="ward.D2")
c2 <- cutree(h1,m)

dd <- cbind(df,c2)

#getting the centroids

c0 <- colMeans(data.matrix(df[dd$c2 == 1, ]))

ds <- data.frame(c0)

for (i in 2:m)
{
  c0 <- colMeans(data.matrix(df[dd$c2 == i, ]))
  ds <- cbind(ds,c0)
}

#the centroids are on ds

#computar la media por cada variable y crear una row que se convierta en el centroide
# -------------- Data transformation --------------
#Approach 1 -> distribució uniforme
rbfdata <- Dissims[1:dims[1],chosen] #Aquí estàn guardades les dissimilituds de gower entre
                                     #totes les dades i els centroides.
#Approach 2 -> gower distance between centroids and the rest of the data
tds <- data.frame(t(ds))
numericdata <- data.frame(data.matrix(df))

types <- sapply(df,class)

for (i in 1:dims[2])
{
  if(types[i] == "integer")
  {
    tds[,i] <- sapply(tds[,i],as.integer)
    numericdata[,i] <- sapply(numericdata[,i],as.integer)
  }
  if(types[i] == "character")
  {
    tds[,i] <- trunc(tds[,i])
    tds[,i] <- sapply(tds[,i],as.character)
    numericdata[,i] <- sapply(numericdata[,i],as.character)
  }
  if(types[i] == "factor")
  {
    if(nlevels(df[,i]) == 2)
    {
      for(j in 1:n)
      {
        if(tds[j,i] < 1.5) tds[j,i] <- 1
        else tds[j,i] <- 2
      }
    }
    tds[,i] <- trunc(tds[,i])
    tds[,i] <- sapply(tds[,i],as.factor)
    numericdata[,i] <- sapply(numericdata[,i],as.factor)
  }
  if(types[i] == "logical")
  {
    for(j in 1:n)
    {
      if(tds[j,i]<0.5) tds[j,i] <- 0
      else tds[j,i] <- 1
    }
    tds[,i] <- sapply(tds[,i],as.logical)
    numericdata[,i] <- sapply(numericdata[,i],as.logical)
  }
}

fusedata <- rbind(numericdata,tds)

distances <- GowerRBF(fusedata)
distances[lower.tri(distances)] <- t(distances)[lower.tri(distances)]

extrasize <- c(1:m)
extrasize <- dims[1] + extrasize

rbfdata2 <- distances[1:dims[1],extrasize]

#----------------------- Sigmoid Hiperparameter ---------------------------

#Approach 1 -> Gower mean distance between random centroids
distclust <- Dissims[chosen,chosen]

x <- NULL

for (i in 1:n)
{
  sum <- 0
  for(j in distclust[i,])
  {
    sum <- sum+j
  }
  x <- append(x,sum/n)
}

x

#Approach 2 -> Dispersion index from a cluster

y <- NULL

for (i in 1:m)
{
  dissimsi <- Dissims[dd$c2 == i , ]
  y <- append(y,max(dissimsi))
}

y

#--------------------------- Neural Network ------------------------------- 

# Test approach
k1funcs <- NULL
for (i in x)
{
  func1 = function(x) 1/(1 + exp(-x/i))
  k1funcs <- append(k1funcs,func1)
}

rbfdatasig <- rbfdata
for (i in 1:n)
{
  rbfdatasig[,i] <- k1funcs[[i]](rbfdatasig[,i])
}


k2funcs <- NULL
for (j in y)
{
  func2 = function(x) 1/(1 + exp(-x/j))
  k2funcs <- append(k2funcs,func2)
}

rbfdatasig2 <- rbfdata2
for (i in 1:m)
{
  rbfdatasig2[,i] <- k2funcs[[i]](rbfdatasig2[,i])
}

rbfdatasig <- data.frame(rbfdatasig,target = df[,"target"])
rbfdatasig2 <- data.frame(rbfdatasig2,target = df[,"target"])

#Approach 1 -> Test with random centroids

smp_size <- floor(0.75 * dims[1])

train_ind <- sample(seq_len(dims[1]), size = smp_size)

train <- rbfdatasig[train_ind, ]
test <- rbfdatasig[-train_ind,-(n+1)]

trainpred <- numericdata[train_ind,"target"]
testpred <- numericdata[-train_ind, "target"]

formulastring <- "target ~ "
for (i in 1:n)
{
  if(i != n) strformula <- sprintf("X%d + ",i)
  else strformula <- sprintf("X%d",i)
  formulastring <- paste(formulastring,strformula,sep="")
}

model <- glm(formula = as.formula(formulastring), data = train, family = binomial)

summary(model)

probabilities <- predict(model,data.frame(test),type ="response")
probabilities
predicted.classes <- ifelse(probabilities > 0.5, 1 , 0)

#Error %
error_count <- 0

for (i in 1:length(testpred))
{
  if(predicted.classes[i] != testpred[i]) error_count <- error_count + 1
}

error_percent = error_count / length(testpred)
print(error_percent)

#F1 Score

f1.1 <- F1_Score(testpred,predicted.classes)
f1.1

#Approach 2 -> Test with cluster centroids

smp_size <- floor(0.75 * dims[1])

train_ind <- sample(seq_len(dims[1]), size = smp_size)

train2 <- rbfdatasig2[train_ind, ]
test2 <- rbfdatasig2[-train_ind,-(m+1) ]

trainpred <- numericdata[train_ind,"target"]
testpred <- numericdata[-train_ind, "target"]

formulastring2 <- "target ~ "
for (i in 1:m)
{
  if(i != m) strformula <- sprintf("X%d + ",i)
  else strformula <- sprintf("X%d",i)
  formulastring2 <- paste(formulastring2,strformula,sep="")
}

formulastring2

model2 <- glm(formula = as.formula(formulastring2),data = train2,family = binomial)

summary(model2)

probabilities2 <- predict(model2,data.frame(test2),type ="response")
predicted.classes2 <- ifelse(probabilities > 0.5, 1 , 0)

#Error %
error_count <- 0

for (i in 1:length(testpred))
{
  if(predicted.classes2[i] != testpred[i]) error_count <- error_count + 1
}

error_percent2 = error_count / length(testpred)
print(error_percent2)

# F1 Score

f1.2 <- F1_Score(testpred,predicted.classes2)
f1.2


#------------ Comparison with standard logistic regression --------------

smp_size <- floor(0.75 * dims[1])

train_ind <- sample(seq_len(dims[1]), size = smp_size)

train3 <- df[train_ind, ]
test3 <- df[-train_ind,-ncol(df)]

trainpred3 <- df[train_ind,"target"]
testpred3 <- df[-train_ind,"target"]



model3 <- glm(formula = formulastring3, data = train3,family = binomial)

summary(model3)

probabilities3 <- predict(model2,data.frame(test2),type ="response")
predicted.classes3 <- ifelse(probabilities2 > 0.5, 1 , 0)

#Error %
error_count3 <- 0

for (i in 1:length(testpred3))
{
  if(predicted.classes3[i] != testpred3[i]) error_count3 <- error_count3 + 1
}

error_percent3 = error_count3 / length(testpred3)

# F1 Score

f1.3 <- F1_Score(testpred3,predicted.classes3)
f1.3

#------------------------ Comparison with KNN ----------------------
k <- 5

predicted.classes4 <- NULL


for (i in 1:dims[1])
{
  dist <- order(Dissims[i,])
  dist <- dist[-1]
  
  kt <- df$target[dist[1:k]]
  n1s <- sum(kt[kt == 1])/length(kt)
  n0s <- sum(kt[kt == 0])/length(kt)
  
  if(n1s > n0s) predicted.classes4 <- append(predicted.classes4, 1)
  else predicted.classes4 <- append(predicted.classes4, 0)
}

predicted.classes4

error_count4 <- 0

for (i in 1:dims[1])
{
  if(predicted.classes4[i] != df$target[i]) error_count4 <- error_count4 + 1
}

error_percent4 = error_count4 / dims[1]
error_percent4

#F1 Score

f1.4 <- F1_Score(df$target,predicted.classes4)
f1.4


#-------------------- Comparison with normal RBF -------------------

