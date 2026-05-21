from preprocessing.preprocess import load_and_clean_data
from feature_engineering.features import create_features
from model.train_model import train
from visualization.visualize import plot_correlation

# load data
df = load_and_clean_data("data/soildataset.csv")

# feature engineering
df = create_features(df)

# train model
model, scaler = train(df)

# visualize
plot_correlation(df)
