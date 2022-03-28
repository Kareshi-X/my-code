from xgboost import XGBRegressor

xgb_model = XGBRegressor(max_depth=9,
                         n_estimators=500,
                         min_child_weight=1000,
                         colsample_bytree=0.7,
                         subsample=1,
                         eta=0.1,
                         seed=0)
xgb_model.fit(train_data_x,
              train_data_y,
              eval_metric="rmse",
              eval_set=[(train_validation_x,train_validation_y)],
              verbose=20,
              early_stopping_rounds=60)