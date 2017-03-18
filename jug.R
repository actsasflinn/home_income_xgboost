library(jug)
library(xgboost)
library(Matrix)

setwd("~/Projects/xgboost_test")
model <- xgb.load("xgboost1.save")

jug() %>%
  get("/model/(?<age>.*)/(?<workclass>.*)/(?<education>.*)/(?<marital_status>.*)/(?<occupation>.*)/(?<relationship>.*)/(?<gender>.*)", function(req, res, err){
    df <- data.frame(
      age = as.numeric(req$params$age),
      workclass = req$params$workclass,
      education = req$params$education,
      marital_status = req$params$marital_status,
      occupation = req$params$occupation,
      relationship = req$params$relationship,
      gender = req$params$gender
    )
    df$score <- predict(model, data.matrix(df))
    df$prediction <- df$score>0.5
    cat(str(df))
    res$json(df)
  }) %>%
  simple_error_handler_json() %>%
  serve_it()

  jug() %>%
    get("/model/(?<age>.*)/(?<workclass>.*)/(?<education>.*)/(?<marital_status>.*)/(?<occupation>.*)/(?<relationship>.*)/(?<gender>.*)/(?<income_bracket>.*)", function(req, res, err){
      return(paste("hello:", req$params$age, req$params$workclass, req$params$education))
    }) %>%
    simple_error_handler_json() %>%
    serve_it()

    df <- data.frame(
      age = "37",
      workclass = "Private",
      education = "Bachelors",
      marital_status = "Married-civ-spouse",
      occupation = "Exec-managerial",
      relationship = "Husband",
      gender = "Male",
      income_bracket = 1
    )
    l <- list(
      age = "37",
      workclass = "Private",
      education = "Bachelors",
      marital_status = "Married-civ-spouse",
      occupation = "Exec-managerial",
      relationship = "Husband",
      gender = "Male",
      income_bracket = 1
    )

    dtrain <- xgb.DMatrix(data = df, label = df$income_bracket)
    df2 <- data.frame(matrix(ncol = 8, nrow = 1))

    headers = c("age", "workclass", "education",  "marital_status", "occupation", "relationship", "gender", "income_bracket")
    dummy = read.csv("dummy", header = F, col.names = headers, strip.white=T)

    m.test <- hashed.model.matrix(f, income.test, 2^16)
    dtest <- xgb.DMatrix(m.test, label = income.test$income_bracket)
