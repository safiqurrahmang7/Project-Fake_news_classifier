import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
from src.feature_engineering import feature_engineer, text_feature_engineering
from src.data_splitter import train_test_splitter, data_split
from src.model_building import Build_LSTM_model, model_builder
from src.model_evaluating import model_evaluator, model_evaluation
from src.metrics import classification, accuracy, confusion, model_evaluating
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models.signature import infer_signature
from mlflow_utils import create_experiment
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# MLflow setup
experiment_id= create_experiment(name = 'FakeNewsClassification', artifact_location ='LSTM_artifacts', tags = {'env':'dev'})

try:
    with mlflow.start_run(run_name='LSTM',experiment_id=experiment_id):  # Start MLflow run
        
        # Load dataset
        logging.info("Loading dataset...")
        df = pd.read_csv('/content/fake_news_processed.csv')
        if df.empty:
            raise ValueError("Dataset is empty. Please check the CSV file.")
        # df = df.loc[0:1000]  # For testing purposes

        # Log dataset parameters
        mlflow.log_param("dataset_path", "/content/fake_news_processed.csv")
        mlflow.log_param("dataset_size", len(df))

        # Feature engineering
        logging.info("Initializing feature engineering...")
        
        engineer = feature_engineer(text_feature_engineering(column='text'))
        engineered_features = engineer.apply_engineering(df)

        # Log feature engineering parameters
        mlflow.log_param("feature_column", "text")

        # Data splitting
        logging.info("Splitting data into train and test sets...")
        test_size = 0.2
        random_state = 42
        xtrain, xtest, ytrain, ytest = train_test_split(
            engineered_features, df['label'], test_size=test_size, random_state=random_state
        )
        logging.info(f"Train-Test split completed: {xtrain.shape}, {xtest.shape}, {ytrain.shape}, {ytest.shape}")

        # # Log splitting parameters
        # mlflow.log_param("test_size", test_size)
        # mlflow.log_param("random_state", random_state)
        mlflow.keras.autolog()
        # Model building
        logging.info("Building the LSTM model...")
        builder = model_builder(builder=Build_LSTM_model())
        model = builder.apply_model_builder()

        # Model training
        logging.info("Starting model training...")
        epochs = 10
        batch_size = 128
        history = model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, validation_data=(xtest, ytest), verbose=2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            ]
        )

        # # Log training parameters
        # mlflow.log_param("epochs", epochs)
        # mlflow.log_param("batch_size", batch_size)
    

        # Model evaluation
        logging.info("Evaluating the model...")
        evaluator = model_evaluator(model_evaluation)
        ypred = evaluator.evaluate_model(model, xtest)
        print(f"Raw predictions (first 5): {ypred[:5] if ypred is not None else 'None'}")

        #Evaluation metrics
        logging.info("Calculating accuracy metrics...")
        evaluation = model_evaluating(accuracy)
        accuracy = accuracy_score(ytest, ypred)

        # Classification report
        logging.info("Calculating classification report...")
        evaluation = model_evaluating(classification)
        classificationReport = evaluation.evaluate_model(ytest, ypred)

        # Confusion matrix
        logging.info("Calculating confusion matrix...")
        evaluation = model_evaluating(confusion)
        confusionMatrix = confusion_matrix(ytest, ypred)

        # # Log metrics
        # logging.info("Logging metrics...")
        # mlflow.log_metric("accuracy", accuracy)
        # for key, value in classificationReport.items():  # Assuming a dict structure for the report
        #     if isinstance(value, dict):
        #         mlflow.log_metric(f"precision_{key}", value["precision"])
        #         mlflow.log_metric(f"recall_{key}", value["recall"])
        #         mlflow.log_metric(f"f1-score_{key}", value["f1-score"])
        #     else:
        #         logging.info(f"Skipping key '{key}' as it is not a dictionary.")

        # # Save the model locally
        # model.save("models/lstm_model.keras")

        # # Log the model with MLflow
        # mlflow.keras.log_model(
        #     model=model,
        #     artifact_path="model",
        # )

        # Log the saved model file as an artifact
        logging.info("Logging the model file as an artifact...")
        mlflow.log_artifact("lstm_model.keras")
        
        # Save confusion matrix
        logging.info("Saving confusion matrix...")
        confusion_matrix_path = "artifacts/confusion_matrix.png"

        logging.info("Plotting confusion matrix...")
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(confusion_matrix_path)
        mlflow.log_artifact(confusion_matrix_path)

        accuracy_path = 'artifacts/accuracy.png'
        logging.info('Savig the train and test accuracy')
        plt.figure(figsize=(8,6))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['Train','Test'])
        plt.title('Distribution of ModelAccuracy')
        plt.savefig(accuracy_path)
        mlflow.log_artifact(accuracy_path)

        loss_path = 'artifacts/loss.png'
        logging.info('Saving the train and test loss')
        plt.figure(figsize=(10,6))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Distribution Of Train an Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['train','test'])
        plt.savefig(loss_path)
        mlflow.log_artifact(loss_path)

        logging.info("MLflow logging completed successfully.")

except Exception as e:
    logging.error(f"An error occurred: {e}")
