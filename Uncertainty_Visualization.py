import rasterio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import linregress, pearsonr

# Read raster files
mean_file = 'path to mean TIF file'
std_dev_file = 'path to the standard deviation TIF file'

# Open raster files using rasterio
with rasterio.open(mean_file) as mean_src, rasterio.open(std_dev_file) as std_dev_src:
    # Read raster data as arrays
    mean_raster = mean_src.read(1)
    std_dev_raster = std_dev_src.read(1)

# Flatten the arrays to 1D
mean_values = mean_raster.flatten()
std_dev_values = std_dev_raster.flatten()


# Create a boolean mask for NaN values
Mean_mask = np.isnan(mean_values)
StDev_mask = np.isnan(std_dev_values)

# Remove NaN and retain only non-negative values
Masked_Mean = mean_values[~Mean_mask]
Masked_Mean = Masked_Mean[Masked_Mean>=0]
Masked_StDev = std_dev_values[~StDev_mask]
Masked_StDev = Masked_StDev[Masked_StDev>=0]

# Perform clustering
kmeans = KMeans(n_clusters=4, random_state=0)
labels = kmeans.fit_predict(np.column_stack((Masked_Mean, Masked_StDev)))

# Get cluster centers and sort by Mean
cluster_centers = kmeans.cluster_centers_
sorted_indices = np.argsort(cluster_centers[:, 0])
sorted_labels = np.zeros_like(labels)

for new_label, old_label in enumerate(sorted_indices):
    sorted_labels[labels == old_label] = new_label

# Define custom legend labels in the correct order
legend_labels = ["Small", "Medium", "High", "Highest"]

# Plotting
scatter = plt.scatter(Masked_Mean, Masked_StDev, s=1, c=sorted_labels, cmap='viridis', alpha=0.7)

# Add trendline
slope, intercept, r_value, _, _ = linregress(Masked_Mean, Masked_StDev)
plt.plot(Masked_Mean, intercept + slope * Masked_Mean, color='red', label='Trendline')

# Add Pearson correlation
pearson_corr, _ = pearsonr(Masked_Mean, Masked_StDev)
plt.text(0.05, 0.22, f'Pearson Corr: {pearson_corr:.3f}', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

# Add x value display at the start and end of clusters
for i in range(4):
    cluster_points = Masked_Mean[sorted_labels == i]
    if len(cluster_points) > 0:
        plt.annotate(f'{cluster_points.min():.2f}', (cluster_points.min(), 0), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.annotate(f'{Masked_Mean[sorted_labels == 3].max():.2f}', (Masked_Mean[sorted_labels == 3].max(), 0), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
      
# Final touches
plt.xlim(0, 1.1)
plt.ylim(-0.01, 0.25)
plt.xlabel('Mean')
plt.ylabel('Standard Deviation')
plt.title('Scatter Plot with Clusters - Marsh')

# Update legend with custom labels
handles, _ = scatter.legend_elements()
plt.legend(handles, legend_labels, title='Clusters')

# save the figure
plt.savefig('path to save the figure', dpi=300)
plt.show()
