# ======================================================
# Core LightGBM Training Pipeline for AKI Prediction
# De-identified / Reproducible Version for GitHub
# ======================================================

# ---- 1. Dependencies ----
# install.packages(c("lightgbm", "pROC"))
library(lightgbm)
library(pROC)

# ---- 2. Simulated De-identified Data ----
set.seed(42)
n <- 2000

# 模拟四个核心预测变量及结局 (脱敏演示数据)
model_data <- data.frame(
  age = rnorm(n, 60, 12),
  alb_cr_ratio = runif(n, 1, 15),
  sex_female = sample(c(0, 1), n, replace = TRUE),
  glucose = rnorm(n, 6.5, 2.0)
)

# 模拟 AKI 结局 (依赖于这四个变量)
model_data$aki <- rbinom(n, 1, plogis(
  -4 - 0.4 * model_data$alb_cr_ratio + 0.05 * model_data$age - 
  0.6 * model_data$sex_female + 0.15 * model_data$glucose
))

# ---- 3. Train/Test Split (70/30) ----
set.seed(123)
train_idx <- sample(1:nrow(model_data), 0.7 * nrow(model_data))
train_data <- model_data[train_idx, ]
test_data <- model_data[-train_idx, ]

features <- c("age", "alb_cr_ratio", "sex_female", "glucose")

X_train <- as.matrix(train_data[, features])
y_train <- train_data$aki
X_test <- as.matrix(test_data[, features])
y_test <- test_data$aki

# ---- 4. LightGBM Training ----
lgb_params <- list(
  objective = "binary",
  metric = "auc",
  num_leaves = 8,
  max_depth = 3,
  learning_rate = 0.03,
  feature_fraction = 0.7,
  bagging_fraction = 0.7,
  bagging_freq = 1,
  lambda_l1 = 2,
  lambda_l2 = 5,
  min_data_in_leaf = 50,
  verbose = -1,
  seed = 42
)

lgb_model <- lgb.train(
  params = lgb_params,
  data = lgb.Dataset(data = X_train, label = y_train),
  nrounds = 200
)

# ---- 5. Evaluation ----
pred_prob <- predict(lgb_model, X_test)
roc_obj <- roc(y_test, pred_prob, quiet = TRUE)
auc_val <- as.numeric(auc(roc_obj))
cat(sprintf("\nTest AUC: %.4f\n", auc_val))

# ---- 6. Feature Importance ----
importance <- lgb.importance(lgb_model, percentage = TRUE)
print(importance)
