library(xgboost)
library(Matrix)

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

train_spmx = sparse.model.matrix(income_bracket~.-1, data = income.train)

bst <- xgboost(data = train_spmx, label = income.train$income_bracket, max_depth = 9,
               eta = 1, nthread = 2, nrounds = 100, objective = "binary:logistic")

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

test_spmx = sparse.model.matrix(income_bracket~.-1, data = income.test)

income.test$predict <- ifelse(predict(bst, test_spmx)>0.5,1,0)
income.test$match <- income.test$predict==income.test$income_bracket

cat('error of preds=', mean(income.test$match),'\n')

importance <- xgb.importance(feature_names = colnames(train_spmx), model = bst)
print(xgb.plot.importance(importance_matrix = importance))

xgb.save(bst, 'xgboost1.save')
xgb.dump(bst, 'xgboost1.dump', with_states = TRUE)