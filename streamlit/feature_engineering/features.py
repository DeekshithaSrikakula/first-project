def create_features(df):

    df["Moisture_Temp"] = df["Soil_Moisture_%"] * df["Soil_Temperature_C"]
    df["pH_Carbon"] = df["Soil_pH"] * df["Organic_Carbon_%"]

    return df
