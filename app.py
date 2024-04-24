from typing import Dict, Any
import json
import numpy as np
from datetime import datetime
from catboost import CatBoostRegressor


class DataProcessor:
    def __init__(self, data_structure: Dict[str, Any]) -> None:
        self.profile: Dict[str, Any] = data_structure["profile"]
        self.session: np.ndarray = np.array(data_structure["sessions"])

    def process_profile(self) -> None:
        goals_columns = ["antistress", "body_balance", "childrens_training", "flexibility",
                         "fun", "learn_swim", "lose_weight", "rehabilitation", "relief",
                         "strength"]
        self.profile.update({f"personal_goals_{column}": int(column in self.profile["personal_goals"])
                             for column in goals_columns})

        # Закодируем признак пола
        sex_map = {"undefined": "0", "female": "2", "male": "1"}
        self.profile["sex"] = int(sex_map.get(self.profile["sex"], "0"))

        # Выделяем возраст
        epoch_day = int(datetime.now().timestamp() / 86400)
        self.profile["age"] = int((epoch_day - self.profile["birth_date"]) / 365.25)

        # Удаляем признаки
        for key in ["personal_goals", "birth_date"]:
            self.profile.pop(key, None)

    def process_sessions(self) -> None:
        for session in self.session:
            # Сначала добавляем необходимые ключи в session
            session["all_steps"] = session["steps"]["steps"]
            session["recording_day"] = session["steps"]["day"]
            session["meters_traveled"] = session["steps"]["meters"]

            # Теперь мы можем использовать добавленные ключи для обновления session
            session["overlap_dates"] = int(session["activity_day"] == session["recording_day"])

            if session["profile_id"] == self.profile["id"]:
                session.update(self.profile)

            data_steps = [step["steps"] for step in session["steps"]["samples"]]
            session.update({
                "min_steps": min(data_steps),
                "max_steps": max(data_steps),
                "mediana_steps": data_steps[len(data_steps) // 2]
            })

            for column in ["skllzz_without_artifacts", "skllzz_with_artifacts",
                           "start_millis", "stop_millis", "timezone", "steps",
                           "activity_day", "id", "profile_id", "hr_rest"]:
                session.pop(column, None)

        # После обработки всех сессий, преобразуем их в np.ndarray
        self.session = np.array([list(session.values()) for session in self.session])


def load_data(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)


def predict_output(session: np.ndarray, model: CatBoostRegressor) -> np.ndarray:
    return abs(model.predict(session))


def main() -> np.ndarray:
    data_structure = load_data("input.json")
    data_processor = DataProcessor(data_structure)
    data_processor.process_profile()
    data_processor.process_sessions()
    model = CatBoostRegressor(task_type="GPU", devices='0:1').load_model("model.cbm")
    return predict_output(data_processor.session, model)


if __name__ == "__main__":
    result = main()
    print(result)
