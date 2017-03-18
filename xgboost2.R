library(xgboost)
library(caret)

setwd("~/Projects/xgboost_test")
train_file = "train_data"
if (!file.exists(train_file)) {
  download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file)
}
test_file = "test_data"
if (!file.exists(test_file)) {
  download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file)
}

headers = c("age", "workclass", "fnlwgt", "education", "education_num",
  "marital_status", "occupation", "relationship", "race", "gender",
  "capital_gain", "capital_loss", "hours_per_week", "native_country",
  "income_bracket")

income.train = read.csv(train_file, header = F, col.names = headers, strip.white=T)
income.train$income_bracket <- ifelse(income.train$income_bracket=="<=50K",0,1)

income.train$fnlwgt <- NULL
income.train$age <- NULL
income.train$capital_gain <- NULL
income.train$capital_loss <- NULL
income.train$hours_per_week <- NULL
income.train$race <- NULL
income.train$native_country <- NULL
income.train$education_num <- NULL

income.test = read.csv(test_file, header = F, col.names = headers, strip.white=T, skip=1)
income.test$income_bracket <- ifelse(income.test$income_bracket=="<=50K.",0,1)

income.test$fnlwgt <- NULL
income.test$age <- NULL
income.test$capital_gain <- NULL
income.test$capital_loss <- NULL
income.test$hours_per_week <- NULL
income.test$race <- NULL
income.test$native_country <- NULL
income.test$education_num <- NULL

fitControl <- trainControl(method = "cv", number = 2, repeats = 2, search = "random")
model <- train(factor(income_bracket)~., data = income.train, method = "xgbTree", trControl = fitControl)
income.test$prediction <- predict(model, income.test)
income.test$match <- income.test$prediction==income.test$income_bracket
cat('error of preds=', mean(income.test$match),'\n')
sqrt(mean((income.test$prediction-income.test$income_bracket)^2))

fitControl2 <- trainControl(method = "cv", number = 2, repeats = 2, search = "random")
model2 <- train(factor(income_bracket)~., data = income.train, method = "xgbLinear", trControl = fitControl2)
income.test$prediction2 <- predict(model2, income.test)
income.test$match2 <- income.test$prediction2==income.test$income_bracket
cat('error of preds=', mean(income.test$match2),'\n')
