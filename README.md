# CSIRO-Image2Biomass_Prediction

Predict biomass using the provided pasture images

![Alt text for the image](C:\Users\Dell\Downloads\510010962-b16b9628-a054-4e40-be65-ed0fc0600b7c.png)


https://www.kaggle.com/competitions/csiro-biomass
Overview

Build models that predict pasture biomass from images, ground-truth measurements, and publicly available datasets. Farmers will use these models to determine when and how to graze their livestock.

Description

Farmers often walk into a paddock and ask one question: “Is there enough grass here for the herd?” It sounds simple, but the answer is anything but. Pasture biomass - the amount of feed available - shapes when animals can graze, when fields need a break, and how to keep pastures productive season after season.

Estimate incorrectly, and the land suffers; feed goes to waste, and animals struggle. Get it right and everyone wins: better animal welfare, more consistent production, and healthier soils.

Current methods make this assessment more challenging than it could be. The old-school “clip and weigh” method is accurate but slow and impossible at scale. Plate meters and capacitance meters can provide quicker readings, but are unreliable in variable conditions. Remote sensing enables broad-scale monitoring, but it still requires manual validation and can’t separate biomass by species.

This competition challenges you to bring greener solutions to the field: build a model that predicts pasture biomass from images, ground-truth measures, and publicly available datasets. You’ll work with a professionally annotated dataset covering Australian pastures across different seasons, regions, and species mixes, along with NDVI values to enhance your models.

If you succeed, you won’t just improve estimation methods. You’ll help farmers make smarter grazing choices, enable researchers to track pasture health more accurately, and drive the agriculture industry toward more sustainable and productive systems.

Evaluation Scoring The model performance is evaluated using a globally weighted coefficient of determination (R²) computed over all (image, target) pairs together. Each row is weighted according to its target type using the following weights:

Dry_Green_g: 0.1

Dry_Dead_g: 0.1

Dry_Clover_g: 0.1

GDM_g: 0.2

Dry_Total_g: 0.5

This means that instead of calculating R² separately for each target and then averaging, a single weighted R² is computed using all rows combined, with the above per-row weights applied.
