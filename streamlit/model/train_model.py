import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def train(df):

    target = "Nutrient_Quality_Score"

    X = df.drop(target, axis=1)
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=300)
    model.fit(X_train, y_train)

    pickle.dump(model, open("model/soil_model.pkl", "wb"))
    pickle.dump(scaler, open("model/scaler.pkl", "wb"))

    print("Model trained and saved!")

    return model, scaler
