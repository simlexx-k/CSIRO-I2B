Dataset Description
Competition Overview
In this competition, your task is to use pasture images to predict five key biomass components critical for grazing and feed management:

Dry green vegetation (excluding clover)
Dry dead material
Dry clover biomass
Green dry matter (GDM)
Total dry biomass
Accurately predicting these quantities will help farmers and researchers monitor pasture growth, optimize feed availability, and improve the sustainability of livestock systems.

Files
test.csv

sample_id — Unique identifier for each prediction row (one row per image–target pair).
image_path — Relative path to the image (e.g., test/ID1001187975.jpg).
target_name — Name of the biomass component to predict for this row. One of: Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g.
The test set contains over 800 images.

train/

Directory containing training images (JPEG), referenced by image_path.
test/

Directory reserved for test images (hidden at scoring time); paths in test.csv point here.
train.csv

sample_id — Unique identifier for each training sample (image).
image_path — Relative path to the training image (e.g., images/ID1098771283.jpg).
Sampling_Date — Date of sample collection.
State — Australian state where sample was collected.
Species — Pasture species present, ordered by biomass (underscore-separated).
Pre_GSHH_NDVI — Normalized Difference Vegetation Index (GreenSeeker) reading.
Height_Ave_cm — Average pasture height measured by falling plate (cm).
target_name — Biomass component name for this row (Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, or Dry_Total_g).
target — Ground-truth biomass value (grams) corresponding to target_name for this image.
sample_submission.csv

sample_id — Copy from test.csv; one row per requested (image, target_name) pair.
target — Your predicted biomass value (grams) for that sample_id.
What you must predict
For each sample_id in test.csv, output a single numeric target value in sample_submission.csv. Each row corresponds to one (image_path, target_name) pair; you must provide the predicted biomass (grams) for that component. The actual test images are made available to your notebook at scoring time.

