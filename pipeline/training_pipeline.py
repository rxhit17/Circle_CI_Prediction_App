
from src.data_processing import DataProcessing
from src.model_training import Modeltraining



if __name__ == "__main__":
    data_processer = DataProcessing("artifacts/raw/data.csv")
    data_processer.run()

    train_model = Modeltraining()
    train_model.run()