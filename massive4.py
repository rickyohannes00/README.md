import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

#Load Data
df = pd.read_csv(r"C:\Massive Project\filtered_fertilizer_recommendation_indo.csv")

#Encode Kolum Kategorial
tanah_encoder = LabelEncoder()
tanaman_encoder = LabelEncoder()
pupuk_encoder = LabelEncoder()

df['Jenis_Tanah'] = tanah_encoder.fit_transform(df['Jenis_Tanah'])
df['Jenis_Tanaman'] = tanaman_encoder.fit_transform(df['Jenis_Tanaman'])
df['Jenis_Pupuk'] = pupuk_encoder.fit_transform(df['Jenis_Pupuk'])

#Separate Features and Target
X = df[['Jenis_Tanah', 'Jenis_Tanaman']]
y = df['Jenis_Pupuk']

#Check class distribution
print("Class distribution before SMOTE:")
print(y.value_counts())

#Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Handle Imbalanced Data with adjusted k_neighbors
smote = SMOTE(random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

#Check class distribution after SMOTE
print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

#Split Data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

#Random Forest Hyperparameter Tuning
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_param_grid = {
    'n_estimators': [100, 250, 500],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 3, 4]
}
rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_

#XGBoost Hyperparameter Tuning 
xgb_param_grid = {
    'n_estimators': [100, 250, 500],
    'max_depth': [10, 15, 20],
    'learning_rate': [0.001, 0.01, 0.1],
    'subsample': [0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_grid_search = GridSearchCV(XGBClassifier(random_state=42), xgb_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)
best_xgb_model = xgb_grid_search.best_estimator_

#Define the base models
base_learners = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('xgb', XGBClassifier(random_state=42))
]

#Meta model
meta_model = LogisticRegression()

#Evaluasi Semua Model
rf_y_pred = best_rf_model.predict(X_test) #Evaluasi Random Forest
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("Random Forest Accuracy:", rf_accuracy)

xgb_y_pred = best_xgb_model.predict(X_test) #Evaluasi XGBoost Model
xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
print("XGBoost Accuracy:", xgb_accuracy)

#Ensemble Model
ensemble_model = VotingClassifier(estimators=[
    ('rf', best_rf_model),
    ('xgb', best_xgb_model)
], voting='soft')
ensemble_model.fit(X_train, y_train)
ensemble_pred = ensemble_model.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
print("Ensemble Model Accuracy:", ensemble_accuracy)

#Define Recommendation Function
def recommend_jenis_pupuk(jenis_tanah_str, jenis_tanaman_str):
    #Convert Inputs to Numeric Values
    jenis_tanah = tanah_encoder.transform([jenis_tanah_str])[0]
    jenis_tanaman = tanaman_encoder.transform([jenis_tanaman_str])[0]
    
    #Predict Jenis Pupuk
    prediction = best_rf_model.predict([[jenis_tanah, jenis_tanaman]])[0]
    recommended_jenis_pupuk = pupuk_encoder.inverse_transform([prediction])[0]
    
    return recommended_jenis_pupuk

# Fungsi untuk memeriksa apakah input valid untuk jenis tanah atau jenis tanaman
def validate_input(input_value, valid_values, prompt):
    while input_value not in valid_values:
        print(f"{input_value} tidak tersedia, ketik lagi.")
        input_value = input(prompt)
    return input_value

#User Input untuk Rekomendasi
jenis_tanah_example = input("Masukkan jenis tanah (contoh: Liat): ")
jenis_tanaman_example = input("Masukkan jenis tanaman (contoh: Beras): ")
recommended_jenis_pupuk = recommend_jenis_pupuk(jenis_tanah_example, jenis_tanaman_example)

print(f"Jenis Tanah: {jenis_tanah_example}")
print(f"Jenis Tanaman: {jenis_tanaman_example}")
print("Rekomendasi Pupuk:", recommended_jenis_pupuk)