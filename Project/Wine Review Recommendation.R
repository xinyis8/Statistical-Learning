# Relevant Packages
library(tm)
library(dplyr)
library(ggplot2)
library(data.table)
library(wordcloud)
library(wordcloud2)
library(glmnet)
library(tm)
library(gridExtra)
library(reshape2)
wine <- read.csv("~/Desktop/2019 Fall/STAT 542/Individual Project/wine-reviews/winemag-data-130k-v2.csv")
### Check missing values 
length(which((wine[,'country'])==''))
length(which(is.na(wine[,'country'])))
length(which((wine[,'price'])==''))
length(which(is.na(wine[,'price']))) # missing value 
length(which((wine[,'points'])==''))
length(which(is.na(wine[,'points'])))
length(which((wine[,'taster_twitter_handle'])==''))
length(which((wine[,'province'])=='')) 
length(which((wine[,'title'])=='')) 
length(which(is.na(wine[,'title'])))
length(which((wine[,'variety'])==''))
length(which((wine[,'winery'])=='')) 
length(which(is.na(wine[,'winery'])))
###### text mining 
e8 <- data.frame(doc_id=seq(1,nrow(wine),1),text=wine$description,stringsAsFactors = FALSE)
corpus <- VCorpus(DataframeSource(e8))
tryTolower <- function(x){
  # return NA when there is an error
  y = NA
  # tryCatch error
  try_error = tryCatch(tolower(x), error = function(e) e)
  # if not an error
  if (!inherits(try_error, 'error'))
    y = tolower(x)
  return(y)
}
clean.corpus<-function(corpus){
  corpus <- tm_map(corpus, content_transformer(tryTolower))
  corpus <- tm_map(corpus, removeWords, stopwords('english'))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removeNumbers)
  return(corpus)
}
newcorpus <- clean.corpus(corpus)
tdm<-TermDocumentMatrix(newcorpus, control=list(weighting=weightTf))
tdm=removeSparseTerms(tdm,0.997)
tdm.essay8 <- as.matrix(tdm)
g=as.data.frame(t(tdm.essay8))
m=g
m$price_add=wine$price
# price missing value: here, I use the mode of price in data set wine 
# to solve missing value of price variable 
m[which(is.na(m$price)),'price_add']=39
m$design_add=ifelse(wine$designation=='',0,1) # alcohol designation
m$point_add=wine$points
## Descriptive Statistics Data Visualization
summary(wine$price)
wine[which(is.na(wine$price)),'price']=39
hist(wine$price,
     main="Histogram of Wine Price",
     xlab="Wine Price",
     xlim=c(0,200),
     col="darkmagenta",
     breaks=1000
)
hist(wine$points,
     main="Histogram of Wine Points",
     xlab="Wine Points",
     xlim=c(80,100),
     col="darkgreen",
     breaks=100
)
which(is.na(wine$country))
wine[which(is.na(wine$country)),'country']='others'
country=as.character(wine$country)
country[is.na(country)]='others'
country_des=summary(country)
k=(table(country))
province=as.character(wine$province)
l=table(province)
p=table(wine$winery)
p=sort(p,decreasing=TRUE)
sfq <- data.frame(words=names(sort(rowSums(tdm.essay8),decreasing = TRUE)), freqs=sort(rowSums(tdm.essay8),decreasing = TRUE), row.names = NULL)
ggplot(sfq[1:30,], mapping = aes(x = reorder(words, freqs), y = freqs)) +
  geom_bar(stat= "identity", fill=rgb(0/255,191/255,196/255)) +
  coord_flip() +
  scale_colour_hue() +
  labs(x= "Words", title = "30 Most Frequenct Words (Wine Reviews)") +
  theme(panel.background = element_blank(), axis.ticks.x = element_blank(),axis.ticks.y = element_blank())
library(wordcloud)
wordcloud(sfq$words,sfq$freqs, min.freq = 1, max.words = 300, colors=blues9) 
m=g
m$price_add=wine$price # missing value = 39
m$design_add=ifelse(wine$designation=='',0,1) # alcohol designation
m$point_add=wine$points
m$variety_add=wine$variety  # add recomend
m$winery_add=wine$winery
m$province_add=wine$province
m$country_add=wine$country
m$region_1=wine$region_1
m$region_2=wine$region_2
m[which(is.na(m$price)),'price_add']=39
## Data Preparation for modeling 
data_total=model.matrix(~ .- 1, m)
## Divide total data set into train data and test data 
# 75% train data 
# 25% test data 
set.seed(0128)
id = sample(1:nrow(data_total),floor(nrow(data_total)*0.75),replace=FALSE) # traindata
train_m=m[id,]
test_m=m[-id,]
# linear model 
model_linear<-lm(point_add~.,as.data.frame(train_m))
summary(model_linear)
pred_linear=predict(model_linear, newdata = test_m)
point_test_true=test_m[,'point_add']
mse_linear = mean((pred_linear-point_test_true)^2)
print(mse_linear)
# MSE linear 3.33486561439597
Y_train=as.matrix(train_m[,1140])
X_train=as.matrix(train_m[,-1140])
X_test=as.matrix(test_m[,-1140])
Y_test=as.matrix(test_m[,1140])
# Lasso regression 
mycv = cv.glmnet(x=(X_train), y=(Y_train),nfolds = 10, alpha=1)
myfit_lasso_min=glmnet(x=(X_train), y=(Y_train), lambda = mycv$lambda.min, alpha=1)
pred_lasso_min=predict(myfit_lasso_min,X_test,type = "response")
mse_lasso = mean((pred_lasso_min-Y_test)^2)
print(mse_lasso) 
# lasso 3.33370606209833
# Ridge regression
mycv = cv.glmnet(x=(X_train), y=(Y_train),nfolds = 10, alpha=0)
myfit_ridge_min=glmnet(x=(X_train), y=(Y_train), lambda = mycv$lambda.min, alpha=0)
pred_ridge_min=predict(myfit_ridge_min,X_test,type = "response")
mse_ridge = mean((pred_ridge_min-Y_test)^2)
print(mse_ridge)
# ridge 3.33198850092889
library(xgboost)
xgb.model_1 <- xgboost(data = X_train, label = Y_train, eta = 0.2, nrounds = 100, verbose = 0)
tmp <- predict(xgb.model_1, X_test)
mse_xgb=mean((tmp - Y_test)^2)
# xgboost 3.2791
xgb.model_2 <- xgboost(data = X_train, label = Y_train, eta = 0.2, nrounds = 200, verbose = 0)
tmp <- predict(xgb.model_2, X_test)
mse_xgb_2=mean((tmp - Y_test)^2)
# 2.8307
xgb.model_3 <- xgboost(data = X_train, label = Y_train, eta = 0.2, nrounds = 300, verbose = 0)
tmp <- predict(xgb.model_3, X_test)
mse_xgb_3=mean((tmp - Y_test)^2)
# 2.6409
xgb.model_4 <- xgboost(data = X_train, label = Y_train, eta = 0.1, nrounds = 300, verbose = 0)
tmp <- predict(xgb.model_4, X_test)
mse_xgb_4=mean((tmp - Y_test)^2)
# 2.7812
xgb.model_5 <- xgboost(data = X_train, label = Y_train, eta = 0.3, nrounds = 300, verbose = 0)
tmp <- predict(xgb.model_5, X_test)
mse_xgb_5=mean((tmp - Y_test)^2)
# 2.8979
### Recommendation Part 

# recommendation 
# 1137
fruit_obs=c(which(m[,'fruit']==1),which(m[,'fruity']==1),which(m[,'fruitiness']==1),which(m[,'fruits']==1))
fruity=m[fruit_obs,]
pinot_noir_obs=which(fruity[,'variety_add']=='Pinot Noir')
pinot_noiry=fruity[pinot_noir_obs,]
price_sort_obs=which(pinot_noiry[,'price_add']<=20)
deal_recom=pinot_noiry[price_sort_obs,]
country_recom=deal_recom[,'country_add']
province_recom=deal_recom[,'province_add']
design_reom=deal_recom[,'design_add']
wineryy=deal_recom[,'winery_add']
print(dim(deal_recom))
winery_level_recom=deal_recom[,'winery_add']
winery_level=unique(winery_level_recom)
print(length(winery_level))
wine_recom=table(winery_level_recom)
wine_recom=sort(wine_recom,decreasing=TRUE)
NB=which(deal_recom[,'winery_add']=='Nuiton-Beaunoy')
NB_point=deal_recom[which(deal_recom[,'winery_add']=='Nuiton-Beaunoy'),'point_add']
NB_province=deal_recom[which(deal_recom[,'winery_add']=='Nuiton-Beaunoy'),'province_add']
NB_price=deal_recom[which(deal_recom[,'winery_add']=='Nuiton-Beaunoy'),'price_add']
NB_region_1=deal_recom[which(deal_recom[,'winery_add']=='Nuiton-Beaunoy'),'region_1']
NB_region_2=deal_recom[which(deal_recom[,'winery_add']=='Nuiton-Beaunoy'),'region_2']
wine_recom=as.data.frame(wine_recom)
y=wine_recom[which(wine_recom$Freq>=5),] 
# summary(NB_point)
# summary(NB_price)
VB=which(deal_recom[,'winery_add']=='Vignerons de Buxy')
VB_point=deal_recom[which(deal_recom[,'winery_add']=='Vignerons de Buxy'),'point_add']
VB_province=deal_recom[which(deal_recom[,'winery_add']=='Vignerons de Buxy'),'province_add']
VB_price=deal_recom[which(deal_recom[,'winery_add']=='Vignerons de Buxy'),'price_add']
VB_region_1=deal_recom[which(deal_recom[,'winery_add']=='Vignerons de Buxy'),'region_1']
VB_region_2=deal_recom[which(deal_recom[,'winery_add']=='Vignerons de Buxy'),'region_2']
VB_country=deal_recom[which(deal_recom[,'winery_add']=='Vignerons de Buxy'),'country_add']
VB_province=deal_recom[which(deal_recom[,'winery_add']=='Vignerons de Buxy'),'province_add']
AZ=which(deal_recom[,'winery_add']=='A to Z')
AZ_point=deal_recom[which(deal_recom[,'winery_add']=='A to Z'),'point_add']
AZ_province=deal_recom[which(deal_recom[,'winery_add']=='A to Z'),'province_add']
AZ_price=deal_recom[which(deal_recom[,'winery_add']=='A to Z'),'price_add']
AZ_region_1=deal_recom[which(deal_recom[,'winery_add']=='A to Z'),'region_1']
AZ_region_2=deal_recom[which(deal_recom[,'winery_add']=='A to Z'),'region_2']
AZ_country=deal_recom[which(deal_recom[,'winery_add']=='A to Z'),'country_add']
AZ_province=deal_recom[which(deal_recom[,'winery_add']=='A to Z'),'province_add']
# summary(AZ_point)
# summary(AZ_price)
CR=which(deal_recom[,'winery_add']=='Castle Rock')
CR_point=deal_recom[which(deal_recom[,'winery_add']=='Castle Rock'),'point_add']
CR_province=deal_recom[which(deal_recom[,'winery_add']=='Castle Rock'),'province_add']
CR_price=deal_recom[which(deal_recom[,'winery_add']=='Castle Rock'),'price_add']
CR_region_1=deal_recom[which(deal_recom[,'winery_add']=='Castle Rock'),'region_1']
CR_region_2=deal_recom[which(deal_recom[,'winery_add']=='Castle Rock'),'region_2']
CR_country=deal_recom[which(deal_recom[,'winery_add']=='Castle Rock'),'country_add']
CR_province=deal_recom[which(deal_recom[,'winery_add']=='Castle Rock'),'province_add']
FD=which(deal_recom[,'winery_add']=='Firesteed')
FD_point=deal_recom[which(deal_recom[,'winery_add']=='Firesteed'),'point_add']
FD_province=deal_recom[which(deal_recom[,'winery_add']=='Firesteed'),'province_add']
FD_price=deal_recom[which(deal_recom[,'winery_add']=='Firesteed'),'price_add']
FD_region_1=deal_recom[which(deal_recom[,'winery_add']=='Firesteed'),'region_1']
FD_region_2=deal_recom[which(deal_recom[,'winery_add']=='Firesteed'),'region_2']
FD_country=deal_recom[which(deal_recom[,'winery_add']=='Firesteed'),'country_add']
FD_province=deal_recom[which(deal_recom[,'winery_add']=='Firesteed'),'province_add']
COE=which(deal_recom[,'winery_add']=='Coelho')
COE_point=deal_recom[which(deal_recom[,'winery_add']=='Coelho'),'point_add']
COE_province=deal_recom[which(deal_recom[,'winery_add']=='Coelho'),'province_add']
COE_price=deal_recom[which(deal_recom[,'winery_add']=='Coelho'),'price_add']
COE_region_1=deal_recom[which(deal_recom[,'winery_add']=='Coelho'),'region_1']
COE_region_2=deal_recom[which(deal_recom[,'winery_add']=='Coelho'),'region_2']
COE_country=deal_recom[which(deal_recom[,'winery_add']=='Coelho'),'country_add']
COE_province=deal_recom[which(deal_recom[,'winery_add']=='Coelho'),'province_add']
LR=which(deal_recom[,'winery_add']=='Labour??-Roi')
LR_point=deal_recom[which(deal_recom[,'winery_add']=='Labour??-Roi'),'point_add']
LR_province=deal_recom[which(deal_recom[,'winery_add']=='Labour??-Roi '),'province_add']
LR_price=deal_recom[which(deal_recom[,'winery_add']=='Labour??-Roi'),'price_add']
LR_region_1=deal_recom[which(deal_recom[,'winery_add']=='Labour??-Roi'),'region_1']
LR_region_2=deal_recom[which(deal_recom[,'winery_add']=='Labour??-Roi'),'region_2']
LR_country=deal_recom[which(deal_recom[,'winery_add']=='Labour??-Roi'),'country_add']
LR_province=deal_recom[which(deal_recom[,'winery_add']=='Labour??-Roi'),'province_add']
WF=which(deal_recom[,'winery_add']=='Wakefield')
WF_point=deal_recom[which(deal_recom[,'winery_add']=='Wakefield'),'point_add']
WF_province=deal_recom[which(deal_recom[,'winery_add']=='Wakefield'),'province_add']
WF_price=deal_recom[which(deal_recom[,'winery_add']=='Wakefield'),'price_add']
WF_region_1=deal_recom[which(deal_recom[,'winery_add']=='Wakefield'),'region_1']
WF_region_2=deal_recom[which(deal_recom[,'winery_add']=='Wakefield'),'region_2']
WF_country=deal_recom[which(deal_recom[,'winery_add']=='Wakefield'),'country_add']
WF_province=deal_recom[which(deal_recom[,'winery_add']=='Wakefield'),'province_add']

plot_NB_point=data.frame(point=WF_point,winery='Nuiton-Beaunoy')
plot_VB_point=data.frame(point=WF_point,winery='Vignerons de Buxy')
plot_V_point=data.frame(point=WF_point,winery='Vignerons de Buxy')
plot_WF_point=data.frame(point=WF_point,winery='Wakefield')
plot_LR_point=data.frame(point=LR_point,winery='Labour??-Roi')

sfq_feature_NB=sfq_feature_NB[-1,]
sfq_feature_NB=sfq_feature_NB[-2,]
# sfq_feature_NB=sfq_feature_NB[-3,]
ggplot(sfq_feature_NB[1:20,], mapping = aes(x = reorder(words, freqs), y = freqs)) +
  geom_bar(stat= "identity", fill=rgb(0/255,191/255,196/255)) +
  coord_flip() +
  scale_colour_hue() +
  labs(x= "Words", title = "20 Most Frequenct Words (Nuiton-Beaunoy)") +
  theme(panel.background = element_blank(), axis.ticks.x = element_blank(),axis.ticks.y = element_blank())
sfq_feature_VB=sfq_feature_VB[-1,]
sfq_feature_VB=sfq_feature_VB[-2,]
ggplot(sfq_feature_VB[1:20,], mapping = aes(x = reorder(words, freqs), y = freqs)) +
  geom_bar(stat= "identity", fill=rgb(0/255,191/255,196/255)) +
  coord_flip() +
  scale_colour_hue() +
  labs(x= "Words", title = "20 Most Frequenct Words (Vignerons de Buxy)") +
  theme(panel.background = element_blank(), axis.ticks.x = element_blank(),axis.ticks.y = element_blank())

sfq_feature_AZ=sfq_feature_AZ[-1,]
sfq_feature_AZ=sfq_feature_AZ[-2,]
ggplot(sfq_feature_AZ[1:20,], mapping = aes(x = reorder(words, freqs), y = freqs)) +
  geom_bar(stat= "identity", fill=rgb(0/255,191/255,196/255)) +
  coord_flip() +
  scale_colour_hue() +
  labs(x= "Words", title = "20 Most Frequenct Words (A to Z)") +
  theme(panel.background = element_blank(), axis.ticks.x = element_blank(),axis.ticks.y = element_blank())
sfq_feature_CR=sfq_feature_CR[-1,]
sfq_feature_CR=sfq_feature_CR[-2,]
ggplot(sfq_feature_CR[1:20,], mapping = aes(x = reorder(words, freqs), y = freqs)) +
  geom_bar(stat= "identity", fill=rgb(0/255,191/255,196/255)) +
  coord_flip() +
  scale_colour_hue() +
  labs(x= "Words", title = "20 Most Frequenct Words (Castle Rock)") +
  theme(panel.background = element_blank(), axis.ticks.x = element_blank(),axis.ticks.y = element_blank())
sfq_feature_FD=sfq_feature_FD[-1,]
sfq_feature_FD=sfq_feature_FD[-2,]
ggplot(sfq_feature_FD[1:20,], mapping = aes(x = reorder(words, freqs), y = freqs)) +
  geom_bar(stat= "identity", fill=rgb(0/255,191/255,196/255)) +
  coord_flip() +
  scale_colour_hue() +
  labs(x= "Words", title = "20 Most Frequenct Words (Firesteed)") +
  theme(panel.background = element_blank(), axis.ticks.x = element_blank(),axis.ticks.y = element_blank())

sfq_feature_LR=sfq_feature_LR[-1,]
sfq_feature_LR=sfq_feature_LR[-8,]
sfq_feature_LR=sfq_feature_LR[-8,]
sfq_feature_LR=sfq_feature_LR[-10,]
sfq_feature_LR=sfq_feature_LR[-8,]
sfq_feature_LR=sfq_feature_LR[-7,]

ggplot(sfq_feature_LR[1:20,], mapping = aes(x = reorder(words, freqs), y = freqs)) +
  geom_bar(stat= "identity", fill=rgb(0/255,191/255,196/255)) +
  coord_flip() +
  scale_colour_hue() +
  labs(x= "Words", title = "20 Most Frequenct Words (Labour??-Roi)") +
  theme(panel.background = element_blank(), axis.ticks.x = element_blank(),axis.ticks.y = element_blank())
sfq_feature_WF=sfq_feature_WF[-1,]
sfq_feature_WF=sfq_feature_WF[-2,]
sfq_feature_WF=sfq_feature_WF[-7,]
sfq_feature_WF=sfq_feature_WF[-13,]

ggplot(sfq_feature_WF[1:20,], mapping = aes(x = reorder(words, freqs), y = freqs)) +
  geom_bar(stat= "identity", fill=rgb(0/255,191/255,196/255)) +
  coord_flip() +
  scale_colour_hue() +
  labs(x= "Words", title = "20 Most Frequenct Words (Wakefield)") +
  theme(panel.background = element_blank(), axis.ticks.x = element_blank(),axis.ticks.y = element_blank())


# Reference
# The text mining part is based on Prof. Kinson STAT 448 Class NLP Topic