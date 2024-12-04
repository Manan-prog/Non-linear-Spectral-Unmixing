import os
import joblib
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from sklearn.model_selection import train_test_split , RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image as im
from rasterio.plot import show
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gist_gray"

# Load the CSV file as a NumPy array of the training data for multiple land cover classes
Crops = np.genfromtxt('Path to Crops.csv', delimiter=',', names=None)
Bare_Soil = np.genfromtxt('Path to BareSoil.csv', delimiter=',')
Marsh = np.genfromtxt('Path to Marsh.csv', delimiter=',')
Water = np.genfromtxt('Path to Water.csv', delimiter=',')
Salt = np.genfromtxt('Path to Salt.csv', delimiter=',')

# Removing the header from all the columns
Crops = np.delete(Crops, 0, axis=0)
Bare_Soil = np.delete(Bare_Soil, 0, axis=0)
Marsh = np.delete(Marsh, 0, axis=0)
Water = np.delete(Water, 0, axis=0)
Salt = np.delete(Salt, 0, axis=0)

# Print the shape of the array
print(Crops.shape)
print(Bare_Soil.shape)
print(Marsh.shape)
print(Water.shape)
print(Salt.shape)

# The above step is crucial as number of sample from each category will indicate the sampling approach in the next steps.
# As the salt class had the least number of training samples, we lead with it.

# Taking random rows from the training data set
n = Salt.shape[0]

np.random.seed(66)

# Generate a fixed random permutation of the indices
indices= np.random.permutation(Crops.shape[0])

# Extract the first 290 rows using the random indexing
Crops_random_rows = Crops[indices[:5*n], :]

# Generate a fixed random permutation of the indices
indices= np.random.permutation(Bare_Soil.shape[0])
# Extract the first 290 rows using the random indexing
Bare_Soil_random_rows = Bare_Soil[indices[:4*n], :]

# Generate a fixed random permutation of the indices
indices= np.random.permutation(Marsh.shape[0])
# Extract the first 290 rows using the random indexing
Marsh_random_rows = Marsh[indices[:3*n], :]

# Generate a fixed random permutation of the indices
indices= np.random.permutation(Water.shape[0])
# Extract the first 290 rows using the random indexing
Water_random_rows = Water[indices[:4*n], :]

# -- creating testing array
feat = np.concatenate((Crops_random_rows, Bare_Soil_random_rows,Marsh_random_rows, Water_random_rows, Salt))

feat_norm = feat

# -- creating target array
targ = np.concatenate((np.full(5*n, 'Crops'), np.full(4*n, 'Bare_Soil'), np.full(3*n, 'Marsh'), 
                       np.full(4*n, 'Water'), np.full(n, 'Salt')))

# Initialize the Random Forest Classifier with balanced subsample class weighting
model = RandomForestClassifier(class_weight="balanced_subsample")

# Define the number of folds for cross-validation
num_folds = 5

# Set up k-fold cross-validation with shuffling and a fixed random seed for reproducibility
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform cross-validation and compute accuracy scores for each fold
fold_scores = cross_val_score(model, feat, targ, cv=kf)

# Output the accuracy scores for each fold
for fold, score in enumerate(fold_scores):
    print(f"Fold {fold + 1} Accuracy: {score}")

# Calculate the mean accuracy across all folds
mean_accuracy = fold_scores.mean()
print("Mean Accuracy:", mean_accuracy)

# Define the number of times to train the model
num_iterations = 10

# Create a list to store the trained models
trained_models = []

# Perform training and saving models
for _ in range(num_iterations):
    
  # Create a training/testing sample. We define the random state parameter here to quantify model structure uncertainty
  feat_tr, feat_te, targ_tr, targ_te = train_test_split(feat_norm, targ, test_size=0.2, random_state=42, stratify=targ)

  print("number of training examples : {0}".format(targ_tr.size))
  print("number of testing examples  : {0}".format(targ_te.size))

  
  # Instantiate the Random Forest Classifier model, random state not defined to have different model strucutre
  model = RandomForestClassifier(class_weight="balanced_subsample")
  
  # Fit the model to the training data
  model.fit(feat_tr, targ_tr)
  
  # Save the trained model
  trained_models.append(model)
  joblib.dump(model, "G:/My Drive/Neural_SpectralUnmixing/RandomForestModles_UncertaintyAnalysis/ModelUncertainty/Salt/SaltModel_Uncertainty" + str(len(trained_models)).zfill(2) + ".joblib")
  
  # Predict the training and testing sets
  pred_tr = model.predict(feat_tr)
  pred_te = model.predict(feat_te)

  # Print the accuracy on the training and testing set
  acc_tr = accuracy_score(targ_tr, pred_tr)
  acc_te = accuracy_score(targ_te, pred_te)

  print("training accuracy : {0}".format(acc_tr))
  print("testing accuracy : {0}".format(acc_te))
  print(classification_report(targ_te, pred_te))
  ConfusionMatrixDisplay.from_estimator(model, feat_te,targ_te)
  importances = model.feature_importances_
  print(importances)

# Create a list of names of the raster files over which spectral unmixing and uncertanity analysis has to be performed
file_addresses = [A,B,C,..] # A, B, C represent the path to raster files

# Loop through each file to apply spectral unmixing using 10 trained models
for i, fname in enumerate(file_addresses):
    # Read the image data and normalize it
    data = np.asarray(imageio.imread(fname))
    norm = np.ma.array(data, mask=np.isnan(data))  # Mask NaN values
    norm[np.isnan(norm)] = -99999  # Replace NaN values with a placeholder

    # Reshape the image data into a 2D array for model input
    norm_reshaped = norm.reshape(norm.shape[0] * norm.shape[1], norm.shape[2])
    
    # Initialize a list to store predictions from all models
    pred_imgs = []
    
    for model in trained_models:
        # Predict probabilities for each pixel using the current model
        predict_proba_img = model.predict_proba(norm_reshaped)
        predict_proba_img_reshaped = predict_proba_img.reshape(norm.shape[0], norm.shape[1], 5)

        # Extract the abundance of a specific class (e.g., salt - index 3)
        # To extract the abundance of another class, simple change the index number
        salt_abundance = im.fromarray(predict_proba_img_reshaped[:, :, 3])
        salt_array = np.array(salt_abundance).reshape(salt_abundance.size[1], salt_abundance.size[0], 1)
        pred_imgs.append(salt_array)
    
    # Compute mean and standard deviation of predictions across all models
    mean_img = np.mean(pred_imgs, axis=0)
    std_img = np.std(pred_imgs, axis=0)

    # Read the metadata from the input raster
    with rasterio.open(fname) as src:
        meta = src.meta.copy()
    crs = meta["crs"]
    transform = meta["transform"]

    # Save the standard deviation as a raster file
    output_fname = f"G:/My Drive/Neural_SpectralUnmixing/RandomForestModels_UncertaintyAnalysis/ModelUncertainty/Salt/Salt_2022_StDev{i+1:010}.tif"
    with rasterio.open(output_fname, "w", driver="GTiff", height=std_img.shape[0], width=std_img.shape[1], count=1, dtype=std_img.dtype, crs=crs, transform=transform) as dst:
        dst.write(np.moveaxis(std_img, [2], [0]))

    # Save the mean as a raster file
    output_fname = f"G:/My Drive/Neural_SpectralUnmixing/RandomForestModels_UncertaintyAnalysis/ModelUncertainty/Salt/Salt_2022_Mean{i+1:010}.tif"
    with rasterio.open(output_fname, "w", driver="GTiff", height=mean_img.shape[0], width=mean_img.shape[1], count=1, dtype=mean_img.dtype, crs=crs, transform=transform) as dst:
        dst.write(np.moveaxis(mean_img, [2], [0]))
