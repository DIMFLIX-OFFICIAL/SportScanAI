import json
import numpy as np
from datetime import datetime
from catboost import CatBoostRegressor


class DataProcessor:
    def __init__(self, data_structure: dict) -> None:
        self.profile: dict = data_structure["profile"]
        self.session: np.ndarray = np.array(data_structure["sessions"])

    def process_profile(self) -> None:
        for column in ["antistress", "body_balance", "childrens_training", "flexibility",
                       "fun", "learn_swim", "lose_weight", "rehabilitation", "relief",
                       "strength"]:
            self.profile[f"personal_goals_{column}"] = int(column in self.profile["personal_goals"])

        # Закодируем признак пола
        self.profile["sex"] = int(
            self.profile["sex"]
                .replace("undefined", "0")
                .replace("female", "2")
                .replace("male", "1")
        )

        # Выделяем возраст
        epoch_day = int(datetime.now().timestamp() / 86400)

        # Получить возраст
        self.profile["age"] = int((epoch_day - self.profile["birth_date"]) / 365.25)

        # Удаляем признаки
        del(self.profile["personal_goals"], self.profile["birth_date"])

    def process_sessions(self) -> None:
        for index, session in enumerate(self.session):
            session["all_steps"] = session["steps"]["steps"]
            session["recording_day"] = session["steps"]["day"]
            session["meters_traveled"] = session["steps"]["meters"]
            session["overlap_dates"] = int(session["activity_day"] == session["recording_day"])

            if session["profile_id"] == self.profile["id"]:
                session.update(self.profile)

            data_steps = [step["steps"] for step in session["steps"]["samples"]]
            session["min_steps"] = min(data_steps)
            session["max_steps"] = max(data_steps)
            session["mediana_steps"] = data_steps[len(data_steps) // 2]

            for column in ["skllzz_without_artifacts", "skllzz_with_artifacts",
                           "start_millis", "stop_millis", "timezone", "steps",
                           "activity_day", "id", "profile_id", "hr_rest"]:
                del(session[column])

        self.session = np.array([list(session.values()) for session in self.session])


def load_data(file_path: str) -> dict:
    with open(file_path, "r") as f:
        data_structure = json.load(f)

    return data_structure


def predict_output(session: np.ndarray, model: CatBoostRegressor) -> np.ndarray:
    output = abs(model.predict(session))
    return output


def main() -> np.ndarray:
    data_structure = load_data("input_sample.json")
    data_processor = DataProcessor(data_structure)
    data_processor.process_profile()
    data_processor.process_sessions()
    model = CatBoostRegressor(task_type="GPU", devices='0:1').load_model("model.cbm")
    output = predict_output(data_processor.session, model)
    return output


if __name__ == "__main__":
    result = main()
    print(result)
