import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

def check_data_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, report_path: str = "drift_report.html") -> bool:
    """Генерирует HTML-отчет о дрифте признаков и возвращает True, если обнаружен дрифт."""
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference_data, current_data=current_data)

    # Сохраняем интерактивный HTML-отчет
    data_drift_report.save_html(report_path)

    # Извлекаем результат анализа
    result = data_drift_report.as_dict()
    dataset_drift = result["metrics"][0]["result"]["dataset_drift"]

    if dataset_drift:
        print(" ВНИМАНИЕ : Обнаружен Data Drift! Требуется переобучение модели.")
    else :
        print(" Распределение данных стабильно. Дрифт не обнаружен.")

    return dataset_drift

if __name__ == "__main__":
    columns = ["amount", "hours", "day_of_week", "time_delta", "user_avg_spend"]

    # Эталонные данные (нормальное поведение)
    ref_df = pd.DataFrame([[1500, 14, 2, 45, 2000] for _ in range(100)], columns=columns)

    # Новые данные (например, аномальный всплеск расходов ночью)
    curr_df = pd.DataFrame([[85000, 3, 1, 2, 1200] for _ in range(100)], columns=columns)

    check_data_drift(ref_df, curr_df)