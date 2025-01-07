import pandas as pd
from src.feature_engineering import feature_engineer, text_feature_engineering
from src.data_splitter import train_test_splitter, data_split
from src.model_building import Build_LSTM_model, model_builder
from src.model_evaluating import model_evaluator, model_evaluation

df = pd.read_csv('D:/Project-Fake_news_classifier/ProcessedData/fake_news_processed.csv')

# Initializing the feature engineer
engineer = feature_engineer(text_feature_engineering(max_len=1000, column='text'))
engineered_features = engineer.apply_engineering(df)

# Initializing the data splitter
splitter = data_split(train_test_splitter(test_size=0.2, random_state=42))
xtrain, xtest, ytrain, ytest = splitter.split_data(engineered_features, df['label'])

print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

# Initializing the model builder
builder = model_builder(builder=Build_LSTM_model())
model = builder.apply_model_builder()

# Training the model
model.fit(xtrain, ytrain, epochs=10, batch_size=128, validation_data=(xtest, ytest))

#initializing the model evaluator
evaluator = model_evaluator(model_evaluation())

# Evaluating the model
evaluator.evaluate_model(model, xtest, ytest)







