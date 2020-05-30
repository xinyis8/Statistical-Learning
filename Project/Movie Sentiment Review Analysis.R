all<- read.table("Project2_data.tsv", stringsAsFactors = F,header = T,sep="")
all$review = gsub('<.*?>', ' ', all$review)
splits <- read.table("Project2_splits.csv",header=T)
train = all[-which(all$new_id%in%splits[,s]),]
test = all[which(all$new_id%in%splits[,s]),]

set.seed(416)
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  "text2vec",
  "slam",
  "pROC",
  "glmnet",
  "e1071",
  "tm",
  "dplyr",
  "MASS"
)

prep_fun = tolower
tok_fun = word_tokenizer
it_train = itoken(train$review,
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun)
it_test = itoken(test$review,
                 preprocessor = prep_fun, 
                 tokenizer = tok_fun)
stop_words = c("i", "me", "my", "myself", 
               "we", "our", "ours", "ourselves", 
               "you", "your", "yours", 
               "their", "they", "his", "her", 
               "she", "he", "a", "an", "and",
               "is", "was", "are", "were", 
               "him", "himself", "has", "have", 
               "it", "its", "of", "one", "for", 
               "the", "us", "this")
vocab = create_vocabulary(it_train,ngram = c(1L,4L), stopwords = stop_words)
pruned_vocab = prune_vocabulary(vocab,
                                term_count_min = 5, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)
bigram_vectorizer = vocab_vectorizer(pruned_vocab)
dtm_train = create_dtm(it_train, bigram_vectorizer)
dtm_test = create_dtm(it_test, bigram_vectorizer)


# Feature selection
v.size = dim(dtm_train)[2]
ytrain = train$sentiment

summ = matrix(0, nrow=v.size, ncol=4)
summ[,1] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==1, ]), mean)
summ[,2] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==1, ]), var)
summ[,3] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==0, ]), mean)
summ[,4] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==0, ]), var)
n1=sum(ytrain); 
n=length(ytrain)
n0= n - n1
myp = (summ[,1] - summ[,3])/
  sqrt(summ[,2]/n1 + summ[,4]/n0)
words = colnames(dtm_train)
id = order(abs(myp), decreasing=TRUE)[1:2800]
pos.list = words[id[myp[id]>0]]
neg.list = words[id[myp[id]<0]]
write(words[id], file="myvocab.txt")

myvocab = scan(file = "myvocab.txt", what = character())
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)
it_test = itoken(test$review,
                 preprocessor = tolower, 
                 tokenizer = word_tokenizer)
vocab = create_vocabulary(it_train,ngram = c(1L,4L))
vocab = vocab[vocab$term %in% myvocab, ]
bigram_vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, bigram_vectorizer)
dtm_test = create_dtm(it_test, bigram_vectorizer)

# Method 1: Logistic Regression
train_label<-train$sentiment
test_label<-test$sentiment
mycv = cv.glmnet(x=dtm_train, y=train_label, 
                 family='binomial',type.measure = "auc", 
                 nfolds = 10, alpha=0.1)
myfit = glmnet(x=dtm_train, y=train_label, 
               lambda = mycv$lambda.min, family='binomial', alpha=0.1)
pred.nec.1 = predict(myfit, dtm_test, type = "response")
roc_net.1 = roc(test_label, as.numeric(pred.nec.1))
out.lr = data.frame(test$new_id, as.numeric(pred.nec.1))
colnames(out.lr) = c("new_id", "prob")
write.table(out.lr, file = "mysubmission1.txt", sep = ",", row.names = FALSE)

# Method 2: Naive Bayes
traindata = cbind(as.data.frame(as.matrix(dtm_train), stringsAsFactors=False), ytrain)

testdata = as.data.frame(as.matrix(dtm_test), stringsAsFactors=False)

nb = naiveBayes(traindata, ytrain)

convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}
train.nb <- dtm_train %>%
  apply(MARGIN = 2, convert_counts)
test.nb <- dtm_test %>%
  apply(MARGIN = 2, convert_counts)

ytrain = as.factor(ytrain)
nb.model = naiveBayes(train.nb, ytrain)
Ypred = predict(nb.model, test.nb)
roc_net.2 = roc(test_label, as.numeric(Ypred))
out.nb = data.frame(test$new_id, Ypred)
colnames(out.nb) = c("new_id", "prob")
write.table(out.nb, file = "mysubmission2.txt", sep = ",", row.names = FALSE)

# Method 3: LDA
lda.model = lda(dtm_train, ytrain)
Ytest.pred=predict(lda.model, dtm_test)$class
roc_net.3 = roc(test_label, as.numeric(Ytest.pred))
out.lda = data.frame(test$new_id, Ytest.pred)
colnames(out.lda) = c("new_id", "prob")
write.table(out.lda, file = "mysubmission3.txt", sep = ",", row.names = FALSE)

