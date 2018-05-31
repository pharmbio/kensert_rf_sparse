# Datasets
The data in “kensert_rf_sparse/Datasets/” is obtained from the tox21 challenge (https://tripod.nih.gov/tox21/challenge/data.jsp) and from the paper “Benchmark Data Set for in Silico Prediction of Ames Mutagenicity” by Hansen and co-workers. The data has been normalized, i e. standardized using the standardization procedure IMI eTOX project standardizer (version 0.1.7. https://pypi.python.org/pypi/standardiser) and MolVS standardizer (version 0.0.9. https://pypi.python.org/pypi/MolVS).

# How to run the grid search
Versions:
Python 3.63;
Rdkit 2017-09.1;
pandas 0.20.3;
numpy 1.13.3;
memory_profiler 0.48.0;
scikit-learn 0.19.1;
scipy 0.19.1.

## Generate Morgan fingerprints
First, (1) RDKit, (2) Pandas and (3) Numpy have to be installed:
(1) http://www.rdkit.org/docs/Install.html;
(2) "pip install pandas" for example;
(3) "pip install numpy" for example.

gen_morgan.py should now run, and will output 84 .csv files; 21 different Morgan fingerprints for each of the four SDFs.

## Run grid search
Before running either RFC_grid_search.py, SVC_grid_search.py or FEST_grid_search.py, file names inside these scripts have to be changed to the file names of the generated datasets,

and five packages have to be installed:
(1) memory_profiler: "pip install -U memory_profiler" for example;
(2) scipy: "pip install scipy" for example;
(3) Scikit-learn: "pip install -U scikit-learn" for example;
(4) pandas;
(5) numpy.

Additionally, before running FEST_grid_search.py, FEST needs to be compiled:
(1) http://lowrank.net/nikos/fest/;
(2) download;
write:
(3) tar -zxvf fest.tar.gz;
(4) cd fest;
(5) make;
finally:
(6) move "festlearn" and "festclassify" to same folder as FEST_grid_search.py.

All Grid searches should now be able to run.

## Grid search outputs
When modelling is completed for a dataset, results will be "appended" to a .csv file in main folder.
