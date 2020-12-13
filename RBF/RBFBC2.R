#Libraries and context
library("dplyr") 
library(cluster)
library("MLmetrics")
library(mccr)
source("GowerFunction.R")

setwd("/home/fran/TFG/RBF")

#Dataset preparation

df <- read.table("heart.csv", header=T, sep=",")

# ------- Procesado extra para testear ------------
#df <- df[-1]
#df[1,1] <- 20
# ------------- How much neurons? -----------------
dims <- dim(df)

#Approach -> Poisson lambda depén de la quantitat màxima de dades
m <- rpois(1,dims[1]*0.05)
m

# ---------------- Which neurons? -----------------
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
#Approach 2 -> Dispersion index from a cluster

y <- NULL

for (i in 1:m)
{
  dissimsi <- Dissims[dd$c2 == i , ]
  y <- append(y,max(dissimsi))
}

length(y)

#--------------------------- Neural Network ------------------------------- 

rbfdatasig2 <- rbfdata2
for (j in 1:m)
{
  rbfdatasig2[,j] <- -rbfdatasig2[,j]/y[j]
}

rbfdatasig2 <- 1/(1 + exp(rbfdatasig2))

rbfdatasig2 <- data.frame(rbfdatasig2,target = df[,"target"])

#Approach 2 -> Test with cluster centroids

smp_size <- floor(0.75 * dims[1])

train_ind <- sample(seq_len(dims[1]), size = smp_size)

train2 <- rbfdatasig2[train_ind, ]
test2 <- rbfdatasig2[-train_ind,-(m+1) ]

trainpred2 <- numericdata[train_ind,"target"]
testpred2 <- numericdata[-train_ind, "target"]

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
error_count2 <- 0

for (i in 1:length(testpred2))
{
  if(predicted.classes2[i] != testpred2[i]) error_count2 <- error_count2 + 1
}

error_percent2 = error_count2 / length(testpred2)
print(error_percent2)

# F1 Score

f1.2 <- F1_Score(testpred,predicted.classes2)
f1.2

# Mathew's Score

mccr2 <- mccr(testpred,predicted.classes2)
mccr2
