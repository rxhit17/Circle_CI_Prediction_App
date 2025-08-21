import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score ,recall_score ,f1_score , confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class Modeltraining:
    def __init__(self):
        self.processed_data_path = "artifacts/processed_data"
        self.model_path = "artifacts/model_path"
        os.makedirs(self.model_path, exist_ok=True)
        self.model = DecisionTreeClassifier(criterion= "gini", max_depth= 30 ,random_state=42)
        logger.info("model training starts ")

    def load_path(self):
        try:
            X_train = joblib.load(os.path.join(self.processed_data_path,"X_train.pkl"))
            y_train = joblib.load(os.path.join(self.processed_data_path,"y_train.pkl"))
            X_test = joblib.load(os.path.join(self.processed_data_path,"X_test.pkl"))
            y_test = joblib.load(os.path.join(self.processed_data_path,"y_test.pkl"))

            logger.info("data load successfully ")

            return X_train,X_test,y_test,y_train
        except Exception as e:
            logger.error(f"error while loading a  file {e}")
            raise CustomException("Failed to load data  ", e)
    

    def train_model(self,X_train,y_train):
        try:
            self.model.fit(X_train,y_train)
            joblib.dump(self.model,os.path.join(self.model_path , "model.pkl"))
            logger.info("model trained saved sucessfully..")
        except Exception as e:
            logger.error(f"error while model training  {e}")
            raise CustomException("Failed to train a model  ", e)
        

    def evaluate_model(self,X_test,y_test):
        try:
            y_pred = self.model.predict(X_test)

            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred,average="weighted")
            recall = recall_score(y_test,y_pred,average="weighted")
            f1 = f1_score(y_test,y_pred,average="weighted")

            logger.info(f"accuracy score = {accuracy} ")
            logger.info(f"precision score = {precision} ")
            logger.info(f"recall score = {recall} ")
            logger.info(f"f1 score = {f1} ")

            cm = confusion_matrix(y_test,y_pred)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm,annot= True, cmap="Blues" , xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
            plt.title("confusion Matrix")
            plt.xlabel("predicted label")
            plt.ylabel("actual label")
            confusion_matrix_path = os.path.join(self.model_path, "confusion_matrix.png")
            plt.savefig(confusion_matrix_path)
            plt.close()

            logger.info("Confusion matrix store successfully ")
        except Exception as e:
            logger.error(f"error while model evaluate   {e}")
            raise CustomException("Failed to evaluate model  ", e)
        

    def run(self):
        X_train,X_test,y_test,y_train = self.load_path()
        self.train_model(X_train,y_train)
        self.evaluate_model(X_test,y_test)

if __name__ == "__main__":
    train_model = Modeltraining()
    train_model.run()
    