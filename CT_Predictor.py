import pandas
import seaborn
import sklearn.model_selection as model_selection
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
import matplotlib.pyplot as pyplot
import copy

# Opening the File:
dataset = pandas.read_excel("dataset.xlsx")

# Creating a Heatmap:
seaborn.heatmap(dataset.corr())
pyplot.show()

# Dividing Dataset:
target_CT = dataset["critical_temp"]
sc_features = dataset.drop(["critical_temp"], axis=1)

# Defining Training and Testing Data:
sc_features_train, sc_features_test, target_CT_train, target_CT_test = model_selection.train_test_split(sc_features, target_CT, test_size=0.2)

# Applying Random Forest Regressor Algorithm:
RFR = ensemble.RandomForestRegressor(n_estimators=200)
RFR.fit(sc_features_train, target_CT_train)

print("\nCoefficient of Determination of Prediction => ", RFR.score(sc_features_train, target_CT_train))

# Prediction of Critical Temperatures:
prediction_RFR = RFR.predict(sc_features_test)

# Results:
results_RFR = copy.deepcopy(sc_features_test)
results_RFR["Predicted CT"] = prediction_RFR
results_RFR["Expected CT"] = target_CT_test

print("\n-------------------------------------------------------------------------------------")
print(results_RFR.head(25))

# Plotting a Scatter Graph:
pyplot.scatter(target_CT_test, prediction_RFR)
pyplot.show()

# Metrics:
print("\n\n--------------------------------------------------------------------")
print("\nRandom Forest Regressor Score => ", RFR.score(sc_features_test, target_CT_test))
print("Mean Absolute Error => ", metrics.mean_absolute_error(target_CT_test, prediction_RFR))
print("Mean Squared Error => ", metrics.mean_squared_error(target_CT_test, prediction_RFR))
