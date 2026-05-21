import matplotlib.pyplot as plt

def plot_correlation(df):

    plt.figure()
    corr = df.corr()
    plt.imshow(corr)
    plt.title("Correlation Matrix")
    plt.colorbar()
    plt.show()

def plot_feature_importance(model, features):

    plt.figure()
    plt.barh(features, model.feature_importances_)
    plt.title("Feature Importance")
    plt.show() 