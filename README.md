## Repository Introduction
*   This repository is for the paper `Prediction of Molecular Critical Properties Based on Boruta Feature Selection: Balancing Accuracy, Applicability and Explainability`. It contains related contents such as test set data, models, and test results.
*   Because the ThermoData Engine (TDE) is a commercial database, this study is for academic purposes only. Without the permission of TDE, the critical data points of the test dataset cannot be published. Therefore, the test dataset only contains molecular names and SMILES strings, and the experimental critical data points have been manually hidden. Since we provide the original calculation code, the AARD may appear as 'nan' during the calculation due to the missing data, but this does not affect the normal execution of the code.

## Folder Description
*   The `data` folder contains feature normalization parameters.
*   The `homo` folder contains the prediction results of critical points for homologues.
      - `PURE-homo-150.xlsx` contains four classes of PONA hydrocarbon homologues, with 150 points for each class.
*   The `image` folder contains the performance plots of the models on the training and validation sets.
*   The `model` folder contains the parameters of the trained models.
*   The `newexp` folder contains the prediction performance of the models tested on new substances.
      - `TEST_A.xlsx` contains the test data for Test(A).
      - `TEST_AB.xlsx` contains the test data for Test(A) and Test(B).

## Description of .ipynb Files
*   `GNN_predict.ipynb` uses the GNN model to make predictions on Test(A).
*   `GRP_Pc_2Dpredict.ipynb` uses the GRP (2D) model to predict $P_c$ on Test(A). The internal outputs are the prediction metrics on Test(A), Test(ALL), and Test(B), respectively.
*   `GRP_Tc_2Dpredict.ipynb` uses the GRP (2D) model to predict $T_c$ on Test(A). The internal outputs are the prediction metrics on Test(A), Test(ALL), and Test(B), respectively.
*   `RF_Pc_2Dpredict.ipynb` uses the RF model to predict $P_c$ on Test(A). The internal outputs are the prediction metrics on Test(A), Test(ALL), and Test(B), respectively.
*   `RF_Tc_2Dpredict.ipynb` uses the RF model to predict $T_c$ on Test(A). The internal outputs are the prediction metrics on Test(A), Test(ALL), and Test(B), respectively.
*   `SVR_Pc_2Dpredict.ipynb` uses the SVR model to predict $P_c$ on Test(A). The internal outputs are the prediction metrics on Test(A), Test(ALL), and Test(B), respectively.
*   `SVR_Tc_2Dpredict.ipynb` uses the SVR model to predict $T_c$ on Test(A). The internal outputs are the prediction metrics on Test(A), Test(ALL), and Test(B), respectively.
*   `predict_3D.ipynb` contains the prediction code for all 3D models on the test set. It internally outputs the prediction metrics of the GPR, RF, and SVR models on Test(ALL) and Test(A), respectively.

## Test Data Description
*   All test sets are derived from the ThermoData Engine (TDE) database and consist of two parts: Test(A) and Test(B).
*   Test(A) contains 148 molecules, and Test(B) contains 38 molecules.
*   The entire test set was obtained through manual individual queries, as the TDE software does not allow batch queries or batch data exports.
*   All data in the test set use the critical data points evaluated and accepted by TDE. If there are multiple experimental data points, their average value is calculated and used.
*   Because TDE is a commercial database, this study is for academic purposes only. Without the permission of TDE, the critical data point values of the test dataset cannot be published.