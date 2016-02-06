library(readr)
library(Metrics)
library(tm)
library(caret)

##TODO
#replace digits#gsub("\\d+","NUM",train_content)
#vector space model for query,title and description
#topic models, LDA
#dimensionality reduction, PCA, SVD, tSNE
##


## create custom summary function for caret
#ScoreQuadraticWeightedKappa() - function to compute
KappaSummaryFunction <- function(data, lev=null, model=null){
  #continue only when both columns are numeric
  stopifnot(is.numeric(data$obs),is.numeric(data$pred))
  
  obsv <- round(data$obs)
  predic <- round(data$pred)
  #upper and lower limit for prediction ratings
  predic[which(predic>4)] <- 4
  predic[which(predic<1)] <- 1
  
  out <- ScoreQuadraticWeightedKappa(obsv,predic,1,4)
  names(out) <- c("weightedkappa")
  out
}


train_file <- read_csv('train.csv')
test_file <- read_csv('test.csv')


#create content from query and title
train_content <- paste(train_file$query,train_file$product_title)
test_content <- paste(test_file$query,test_file$product_title)
all_content <- c(train_content,test_content)

#remove all special characters
all_content <- gsub("[^[:alnum:] ]", "", all_content)

#create a corpus object
train_corpus <- Corpus(VectorSource(train_content))
all_corpus <- Corpus(VectorSource(all_content))


#pre process docs, clean and stem
all_corpus <- tm_map(all_corpus,removePunctuation)
all_corpus <- tm_map(all_corpus,removeWords,stopwords("english"))
all_corpus <- tm_map(all_corpus,content_transformer(tolower))
#all_corpus <- tm_map(all_corpus,content_transformer(function(x) gsub("[^[:alnum:] ]", "", x)))
all_corpus <- tm_map(all_corpus,stripWhitespace)
all_corpus <- tm_map(all_corpus,stemDocument)

term_matrix <- DocumentTermMatrix(all_corpus)
#remove terms with sparsity greater than x value - dimensionality
dense_term_matrix <- removeSparseTerms(term_matrix,0.999)

dtmtx <- as.data.frame(as.matrix(dense_term_matrix))

#add the relevance term to data frame
corpus_matrix <- cbind(dtmtx,c(train_file$median_relevance,rep(-1,length(test_content))))

colnames(corpus_matrix)[length(colnames(corpus_matrix))] <- 
  "median_relevance"

#subset training data for model building
train_tfid <- corpus_matrix[1:length(train_content),]
test_tfid <- corpus_matrix[length(train_content)+1:nrow(corpus_matrix),]
##caret training

#trainSamples <- createDataPartition(train_tfid$median_relevance,p=0.8,list=FALSE)
#prod_train <- train_tfid[trainSamples,]
#prod_test <- train_tfid[-trainSamples,]

#create the sampling control 
fitControl <- trainControl(method="repeatedcv",number = 1,repeats=1,
                           summaryFunction = KappaSummaryFunction, allowParallel = TRUE)
#create the parameter search grid
gbmGrid <- expand.grid(interaction.depth=c(15,25),n.trees=c(500,1000),shrinkage=0.1,n.minobsinnode=10)

gbmFit2 <- train(median_relevance~.,data=train_tfid,method='gbm',metric="weightedkappa",tuneGrid=gbmGrid,trControl=fitControl,verbose=FALSE)


predictions <- predict(gbmFit2,newdata=prod_test)
submission <- cbind(test_file$id,round(predictions))
colnames(submission) <- c("id","prediction")
write.csv(submission,file="submission.csv",row.names =FALSE)

