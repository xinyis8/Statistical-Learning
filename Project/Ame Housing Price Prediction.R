data <- read.csv("~/Desktop/2019 spring/542/project/Ames_data.csv", header=TRUE)
load("~/Desktop/2019 spring/542/project/project1_testIDs.R")
# Deal with missing data

dim(data)
missing.n = sapply(names(data), function(x) length(which(is.na(data[, x]))))
which(missing.n > 0)  # 60th col: Garage_Yr_Blt
summary(data)
id = which(is.na(data$Garage_Yr_Blt))
length(id) 
data$Garage_Yr_Blt[id]=0 ## Set missing value to be 0


########## Split the data
j=1
train <- data[-testIDs[,j], ]
test <- data[testIDs[,j], ]
test.y <- test[, c(1, 83)]
test <- test[, -83]
write.csv(train,"train.csv",row.names=FALSE)
write.csv(test, "test.csv",row.names=FALSE)
write.csv(test.y, "test_y.csv",row.names=FALSE)

############### Remove Variables

j <- 1
test.dat <- as.data.frame(data[testIDs[,j], ])
colnames(test.dat)<-colnames(data)
train.dat <- as.data.frame(data[-testIDs[,j], ])
colnames(train.dat)<-colnames(data)
train.y <- log(train.dat$Sale_Price)
remove.var <- c('Street', 'Utilities',  'Condition_2', 'Roof_Matl', 'Heating',
                'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 
                'Longitude','Latitude',"PID", "Sale_Price")
train.x <- ProcessRemoveVars(train.dat, remove.var)
test.y <- log(test.dat$Sale_Price)
test.PID <- test.dat$PID
test.x <- ProcessRemoveVars(test.dat, remove.var)
################# Winsorization
winsor.vars <- c("Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", 
                 "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', 
                 "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", 
                 "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val")
train.x=ProcessWinsorization(train.x,0.95,winsor.vars)
#########test.x=ProcessWinsorization(test.x,0.95,winsor.vars)
#####################################
PreProcessingMatrixOutput <- function(train.data, test.data){
  # generate numerical matrix of the train/test
  # assume train.data, test.data have the same columns
  categorical.vars <- colnames(train.data)[which(sapply(train.data, 
                                                        function(x) is.factor(x)))]
  train.matrix <- train.data[, !colnames(train.data) %in% categorical.vars, drop=FALSE]
  test.matrix <- test.data[, !colnames(train.data) %in% categorical.vars, drop=FALSE]
  n.train <- nrow(train.data)
  n.test <- nrow(test.data)
  for(var in categorical.vars){
    mylevels <- sort(unique(train.data[, var]))
    m <- length(mylevels)
    tmp.train <- matrix(0, n.train, m)
    tmp.test <- matrix(0, n.test, m)
    col.names <- NULL
    for(j in 1:m){
      tmp.train[train.data[, var]==mylevels[j], j] <- 1
      tmp.test[test.data[, var]==mylevels[j], j] <- 1
      col.names <- c(col.names, paste(var, '_', mylevels[j], sep=''))
    }
    colnames(tmp.train) <- col.names
    colnames(tmp.test) <- col.names
    train.matrix <- cbind(train.matrix, tmp.train)
    test.matrix <- cbind(test.matrix, tmp.test)
  }
  return(list(train = train.matrix, test = test.matrix))
}
data.ready<-PreProcessingMatrixOutput(train.x,test.x)

########Necessary function

####ProcessRemoveVars
ProcessRemoveVars<-function(data,var){
  data.remove <- data[, !colnames(data) %in% var, drop=FALSE]
  return(data.remove)
  }


remove.var <- c('Street', 'Utilities',  'Condition_2', 'Roof_Matl', 'Heating',
                'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 
                'Longitude','Latitude',"PID", "Sale_Price")

#######Process Winsorization
ProcessWinsorization=function(data,percentile,vars.winsor){
for (var in vars.winsor ){
bound.winsorize=quantile(data[,colnames(data) %in% var],seq(0.0,0.95,0.01))[percentile*100]
for (i in 1:nrow(data)){
if (data[i,colnames(data) %in% var]>bound.winsorize)
{data[i,colnames(data) %in% var]=bound.winsorize}
  else {data[i,colnames(data) %in% var]=data[i,colnames(data) %in% var] }
                       }
}
  return(data)
}
winsor.vars <- c("Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", 
                 "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', 
                 "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", 
                 "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val")
#############################

library(glmnet)
set.seed(100)
cv.out <- cv.glmnet(as.matrix(data.ready$train), train.y, alpha = 1)
tmp <-predict(cv.out, s = cv.out$lambda.min, newx = as.matrix(data.ready$test))
sqrt(mean((tmp - log(test.y))^2))
