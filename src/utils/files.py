import os
from datetime import datetime


def get_train_file_names(model_name: str):
    # find model filename for last run, and next free filename for model
    model_cnt = 0
    model_folder = os.path.join("results", model_name)
    while os.path.exists(model_folder + str(model_cnt)):
        model_cnt += 1

    model_folder += str(model_cnt) + f"_{datetime.now().strftime('%H%M%a%d%b')}"

    os.makedirs(model_folder)
    model_file = os.path.join(model_folder, "best.model")
    info_file = os.path.join(model_folder, "train.json")
    return model_folder, model_file, info_file
