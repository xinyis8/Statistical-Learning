# setwd("~/Desktop/university/academic/542/project4")
set.seed(100)
# Load packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  "xgboost",
  "plyr"
)
# read data
  data <- read.csv("loan_stat542.csv")
  Y = data.frame("id"= data$id, "loan_status" = data$loan_status)
  data$loan_status = NULL
  
  n = dim(data)[1]
  p = dim(data)[2]

# find variables with missing value
  missing = numeric(p)
  for (i in 1:p){
    missing[i] = sum(is.na(data[,i]))
  }
  miss.var = colnames(data)[missing!=0]

# replace missing of numeric variables with the median, and categorical with
# the most frequent value
  for (var in miss.var) {
    id = which(is.na(data[,var]))
    if (class(data[,var])=="factor") {
      data[id,var] = names(which.max(table(data[,var])))
    }
    else{
      data[id,var] = median(data[,var], na.rm = TRUE)
    }
  }
  
# generate response variable
  Y$loan_status = as.factor(ifelse(Y$loan_status=="Fully Paid", 0, 1))

# train and test split
  test.id = read.csv("Project4_test_id.csv")
  for (i in 1:3){
    id = data.frame("id" = test.id[, i])
    train.id = data.frame("id" = setdiff(data$id, id$id))
    assign(paste0("Xtest", i), join(id, data)[,-1])
    assign(paste0("Ytest", i), join(id, Y)[,-1])
    assign(paste0("Xtrain", i), join(train.id, data)[,-1])
    assign(paste0("Ytrain", i), join(train.id, Y)[,-1])
  }
  


# create function to compute log loss
  logLoss = function(y, p){
    if (length(p) != length(y)){
      stop('Lengths of prediction and labels do not match.')
    }
    
    if (any(p < 0)){
      stop('Negative probability provided.')
    }
    
    p = pmax(pmin(p, 1 - 10^(-15)), 10^(-15))
    mean(ifelse(y == 1, -log(p), -log(1 - p)))
  }

# train models with 3 splits and compute average log loss
  xgb.model1<-xgboost(data=data.matrix(Xtrain1), label = Ytrain1, eta = 0.1, nrounds=100,verbose=0)
  tmp1<-predict(xgb.model1, data.matrix(Xtest1))
  tmp1 = pmax(tmp1-1, 0)
  loss1 = logLoss(as.numeric(Ytest1)-1,tmp1)
  out1 = data.frame(test.id[,1], tmp1)
  colnames(out1) = c("id", "prob")
  write.table(out1, file = "mysubmission_test1.txt", sep = ",", row.names = FALSE)
  
  xgb.model2<-xgboost(data=data.matrix(Xtrain2), label = Ytrain2, eta = 0.1, nrounds=100,verbose=0)
  tmp2<-predict(xgb.model2, data.matrix(Xtest2))
  tmp2 = pmax(tmp2-1, 0)
  loss2 = logLoss(as.numeric(Ytest2)-1,tmp2)
  out2 = data.frame(test.id[,2], tmp2)
  colnames(out2) = c("id", "prob")
  write.table(out2, file = "mysubmission_test2.txt", sep = ",", row.names = FALSE)
  
  xgb.model3<-xgboost(data=data.matrix(Xtrain3), label = Ytrain3, eta = 0.1, nrounds=100,verbose=0)
  tmp3 = predict(xgb.model3, data.matrix(Xtest3))
  tmp3 = pmax(tmp3-1, 0)
  loss3 = logLoss(as.numeric(Ytest3)-1,tmp3)
  out3 = data.frame(test.id[,3], tmp3)
  colnames(out3) = c("id", "prob")
  write.table(out3, file = "mysubmission_test3.txt", sep = ",", row.names = FALSE)

  # mean(c(loss1, loss2, loss3))

# train a classification model with complete dataset
  xgb.model = xgboost(data=data.matrix(data[, -1]), label = Y$loan_status, eta = 0.1, nrounds=100,verbose=0)

# Input test datasets and remove useless variables
  dataQ3 = read.csv("LoanStats_2018Q3.csv")
  dataQ4 = read.csv("LoanStats_2018Q4.csv")
  
  YtestQ3 = dataQ3$loan_status
  YtestQ3 = as.factor(ifelse(YtestQ3=="Fully Paid", 0, 1))
  YtestQ4 = dataQ4$loan_status
  YtestQ4 = as.factor(ifelse(YtestQ4=="Fully Paid", 0, 1))
  remove.var = colnames(dataQ3)[!colnames(dataQ3) %in% colnames(data)]
  ProcessRemoveVars = function(data,var){
    data.remove = data[, !colnames(data) %in% var, drop=FALSE]
    return(data.remove)
  }
  dataQ3 = ProcessRemoveVars(dataQ3, remove.var)
  dataQ4 = ProcessRemoveVars(dataQ4, remove.var)

# # missing value
  # missing = numeric(p)
  # for (i in 1:p){
  #   missing[i] = sum(is.na(dataQ3[,i]))
  #   missing[i] = missing[i] + sum(is.na(dataQ4[,i]))
  # }
  # miss.var = colnames(data)[missing!=0]
  # # sapply(data[, miss.var], class)
  # 
  # # replace missing of numeric variables with the median
  # idQ3 = which(is.na(dataQ3$dti))
  # idQ4 = which(is.na(dataQ4$dti))
  # dataQ3$dti[idQ3] = median(dataQ3$dti, na.rm = TRUE)
  # dataQ4$dti[idQ4] = median(dataQ4$dti, na.rm = TRUE)
  # 
  # dataQ3$emp_title[which(is.na(dataQ3$emp_title))] = names(which.max(table(dataQ3$emp_title)))

# make prediction with our model and test data
  YpredQ3 = predict(xgb.model, data.matrix(dataQ3[,-1]))
  YpredQ3 = pmax(YpredQ3-1, 0)
  lossQ3 = logLoss(as.numeric(YtestQ3)-1, YpredQ3)
  outQ3 = data.frame(dataQ3$id, YpredQ3)
  colnames(outQ3) = c("id", "prob")
  write.table(outQ3, file = "mysubmission_2018Q3.txt", sep = ",", row.names = FALSE)
  
  YpredQ4 = predict(xgb.model, data.matrix(dataQ4[,-1]))
  YpredQ4 = pmax(YpredQ4-1, 0)
  lossQ4 = logLoss(as.numeric(YtestQ4)-1, YpredQ4)
  outQ4 = data.frame(dataQ4$id, YpredQ4)
  colnames(outQ4) = c("id", "prob")
  write.table(outQ4, file = "mysubmission_2018Q4.txt", sep = ",", row.names = FALSE)
  
  
result.1<- read.csv("~/Desktop/2019 spring/542/project4/mysubmission_test1.txt")
result.2<-read.csv("~/Desktop/2019 spring/542/project4/mysubmission_test2.txt")
result.3<-read.csv("~/Desktop/2019 spring/542/project4/mysubmission_test3.txt")
loss.result1 = logLoss(as.numeric(Ytest1)-1,result.1[,2])
loss.result2 = logLoss(as.numeric(Ytest2)-1,result.2[,2])
loss.result3 = logLoss(as.numeric(Ytest3)-1,result.3[,2])
