
# This file describes the organization and format of the **2023 satellite and ground truth data (only 2 locations: Lincoln, NE, Missouri Valley, IA)**. In addition, we provide an example code for searching and retrieving images from individual hybrid corn research plots as well as calculating the wavelength intensity values and indices as used in Shrestha et al [https://doi.org/10.5061/dryad.905qftttm](https://doi.org/10.5061/dryad.905qftttm).

## Directory Layout:


*   **Satellite**
    *   Two directories correspond to Lincoln, NE, Missouri Valley, IA from which satellite imagery was collected.
    *   Four directories numbered sequentially "TP1", "TP2", "TP3", and "TP4" corresponding to the order of satellite images collected at each location. Note that the timing of time points with the same number are not identical across locations, and are not guaranteed to be similar. The time of acquisition for each time point at each location is provided in the file **DateofCollection.xlsx** in the folder **"GroundTruth"**.
    *   .TIF formatted images segmented for each plot at a given location at a given time point. Rows often were at orientations other than 0 or 90 degrees in satellite images. Segmented images consist of the minimum bounding box encompassing the plot of interest, with real pixel data for all plot pixels and zero values entered for all pixels within the minimum bounding box but not part of the plot of interest.
    *   Plot image names follow the format location-time-experiment_range_row.tif (Lincoln-TP1-hybirds_2_2.TIF). An example implementation of how to map between image names, plot IDs, and ground truth information is provided later in this document. The necessary information to map between ground truth, plot ID, and images is provided in the file **train_HIPS_HYBRIDS_2023_V2.3.csv** located in the **"GroundTruth"** folder.
    *   Each image contains data on six bands per pixel: near-infrared, red edge, red, green, blue, and deep blue.


*   **GroundTruth**
    *   **train_HIPS_HYBRIDS_2023_V2.3.csv** contains record for hybrid maize grown at each of the two locations. Each record includes information on the field, location within the field (row and column), the hybrid genotype planted in that plot, and a set of ground truth data collected from that plot. The individual ground truth measurements are defined below
        *   plantingDate: the date the plot was planted.
        *   totalStandCount: The number of living plants observed in the middle two rows of the four-row plot. Note that plots are variable length between locations, so this needs to be corrected for plot size to calculate the density of plants per unit area.
        *   daysToAnthesis: The difference between the planting date and the first date where at least 50% of living plants in the plot had visible anthers present on their tassels (syn. "male flowering"). May not be present for all locations.
        *   GDDToAnthesis: This is a method for quantifying flowering time that corrects for the fact plants grow faster on warm days and slower in the cold. GDD stands for growing degree days. The number of growing degree days per day was calculated using temperatures in Fahrenheit with a crop base temperature of 50 degrees and a crop maximum temperature of 86 degrees.
        *   yieldPerAcre: estimated grain yield per plot. This measurement starts with the direct measurement of grain pass per plot and grain moisture percentage and then corrects for variation in moisture content and differences in plot size across locations to calculate bushels per acre yield at a standardized 15.5% moisture content.
    *   **DateofCollection.xlsx** translates location + TP1/2/3/etc into the specific date of image collection for Satellite images collected at all locations included.


*   **Documentation**
    *   Documentation.ipynb: example code for extracting images, calculating different indices, and linking images/indices to the ground truth data.
    *   Readme.md: explaination of data types and layout.