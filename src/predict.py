import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

FEATURE_COLUMNS_PATH = "../models/feature_columns.joblib"
MODEL_PATH = "../models/xgb_model.json"

feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)


def predict_player_market_value(player: dict) -> float:
    # copiem dictionarul pentru a nu modifica datele originale
    data = player.copy()

    # calculam manual variabilele derivate (Feature Engineering)
    age = data.get("age", 25)
    minutes = data.get("minutes_played", 0)
    goals = data.get("goals", 0)
    assists = data.get("assists", 0)
    clean_sheets = data.get("clean_sheets", 0)
    sub_pos = data.get("sub_position", "")

    # Factor de varsta (scade pana la 29 de ani, apoi devine 0)
    age_factor = max(0, 29 - age)

    # contributions per 90 min
    if minutes > 0:
        data["contributions_per_90"] = ((goals + assists) / minutes) * 90
    else:
        data["contributions_per_90"] = 0.0

    # clean sheets se iau in calcul doar pentru pozitii defensive
    defensive_positions = [
        "Goalkeeper",
        "Centre-Back",
        "Left-Back",
        "Right-Back",
        "Defensive Midfield",
    ]
    if sub_pos not in defensive_positions:
        clean_sheets = 0
        data["clean_sheets"] = 0

    # scoruri ponderate cu varsta
    data["youth_minutes_score"] = minutes * age_factor
    data["contributions_score"] = (goals + assists) * age_factor
    data["clean_sheets_score"] = clean_sheets * age_factor

    # setam 0 default pentru Champions League daca nu exista in dictionar
    data.setdefault("ucl_minutes_played", 0)
    data.setdefault("ucl_goals", 0)
    data.setdefault("ucl_assists", 0)

    # transformam in DataFrame de 1 rand si aplicam One-Hot Encoding
    df = pd.DataFrame([data])
    df_encoded = pd.get_dummies(df)

    # aliniem coloanele cu cele invatate la antrenare (reindex adauga 0 pentru restul)
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # predictie
    prediction = model.predict(df_aligned)[0]

    # cota nu poate fi negativa in realitate
    return max(0.05, float(prediction))



# EXEMPLE DE TEST

if __name__ == "__main__":
    # test 1: Atacant tanar de top (Premier League + UCL)
    star_striker = {
        "age": 21,
        "height_in_cm": 188,
        "sub_position": "Centre-Forward",
        "foot": "right",
        "country_of_citizenship": "England",
        "domestic_competition_id": "GB1",
        "squad_size": 25,
        'teammates_mean_value': 40.0,
        "goals": 18,
        "assists": 7,
        "minutes_played": 2600,
        "international_caps": 6,
        "international_goals": 2,
        "clean_sheets": 0,
        "ucl_minutes_played": 600,
        "ucl_goals": 4,
        "ucl_assists": 2,

    }

    # test 2: Fundas experimentat din campionatul Olandei fara UCL
    veteran_defender = {
        "age": 31,
        "height_in_cm": 184,
        "sub_position": "Centre-Back",
        "foot": "left",
        "country_of_citizenship": "Netherlands",
        "domestic_competition_id": "NL1",
        "squad_size": 28,
        'teammates_mean_value': 40.0,
        "goals": 1,
        "assists": 1,
        "minutes_played": 2100,
        "international_caps": 0,
        "international_goals": 0,
        "clean_sheets": 8,
        "ucl_minutes_played": 0,
        "ucl_goals": 0,
        "ucl_assists": 0,
    }

    val_star = predict_player_market_value(star_striker)
    val_vet = predict_player_market_value(veteran_defender)

    print(f"Cota estimata Atacant Tanar: {val_star:.2f} M €")
    print(f"Cota estimata Fundas Veteran: {val_vet:.2f} M €")