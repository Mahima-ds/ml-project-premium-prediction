import pandas as pd
from joblib import load

# Load models
model_rest = load("artifacts/model_rest.joblib")
model_young = load("artifacts/model_young.joblib")

# Load scalers
scaler_rest = load("artifacts/scaler_rest.joblib")
scaler_young = load("artifacts/scaler_young.joblib")

# ---------- Risk Score Calculation ----------


def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0
    }

    medical_history = medical_history.lower().replace("&", "and").strip()

    if medical_history == "no disease":
        total_score = 0
    else:
        diseases = [d.strip() for d in medical_history.split("and")]
        total_score = sum(risk_scores.get(d, 0) for d in diseases)

    normalized_risk_score = total_score / 14
    return normalized_risk_score

# ---------- Input Preprocessing ----------


def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Number of Dependants':
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':
            df['income_lakhs'] = value
        elif key == 'Genetical Risk':
            df['genetical_risk'] = value

    # Add normalized risk score
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])
    print("⚠️ Normalized Risk Score:", df['normalized_risk_score'].values)

    print("📄 INPUT DF BEFORE SCALING:")
    print(df)

    df = handle_scaling(input_dict['Age'], df)

    print("📈 INPUT DF AFTER SCALING:")
    print(df)

    return df

# ---------- Handle Scaling ----------


def handle_scaling(age, df):
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df['income_level'] = None  # Dummy column to satisfy scaler shape
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop('income_level', axis='columns', inplace=True)
    return df

# ---------- Predict Function ----------


def predict(input_dict):
    print("📥 INPUT DICTIONARY:")
    print(input_dict)

    input_df = preprocess_input(input_dict)

    print("\n✅ DEBUG INFO")
    print(
        "👁️ Columns in scaler:",
        scaler_young['cols_to_scale'] if input_dict['Age'] <= 25 else scaler_rest['cols_to_scale']
    )
    print("📊 FINAL INPUT DF BEFORE PREDICTION:")
    print(input_df)
    print("📊 COLUMN ORDER:")
    print(input_df.columns.tolist())

    if input_dict['Age'] <= 25:
        print("🧠 Model used: model_young")
        prediction = model_young.predict(input_df)
    else:
        print("🧠 Model used: model_rest")
        prediction = model_rest.predict(input_df)

    print("🎯 FINAL PREDICTION:", prediction)

    return int(prediction)
