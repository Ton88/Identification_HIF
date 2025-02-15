## General Explanation of the Simulations

This repository focuses on the simulation and analysis of high-impedance fault detection in power distribution systems. The experiments are based on IEEE standard bus systems, specifically tailored for scenarios involving high-impedance faults (HIFs). These faults are critical in power system studies due to their low fault current, making them challenging to detect with conventional methods.

### Simulation Scope

The simulations are structured as follows:

1. **Power System Models**:
   - Original IEEE bus systems are modeled in MATLAB Simulink, as found in the file `SIMU_2024.rar`. These models serve as the foundation for the experiments, simulating realistic power distribution scenarios.

2. **Fault Scenarios**:
   - High-impedance faults are artificially introduced into the system to study their impact and develop computational techniques for detection.

3. **Data Processing**:
   - After simulating the faults, the results are processed using Wavelet Transformations and other advanced signal processing techniques to identify unique fault characteristics.

4. **Machine Learning Analysis**:
   - The processed data is analyzed using machine learning algorithms, including KNN, MLP, Decision Trees, Random Forest, and SVM, to evaluate the effectiveness of computational methods for fault detection.

**Additional Instructions**

Detailed instructions on how to perform the simulations can be found within the respective folders of this repository. Make sure to check the Python_Workflow or Orange_Workflow and Processed_Data directories for more information on setting up and running the simulations.

This repository simplifies power system simulation execution using cloud resources, eliminating the need for local computational power.



### Source of Models

The original bus system models included in `SIMU_2024.rar` were obtained from the official [MATLAB website](https://www.mathworks.com/products/matlab.html). These models are widely recognized for academic and research purposes.

### License and Citation

The authors of the original bus system models have made them available under a free license with the condition that the following works are cited in any research or publication that uses these models:

1. **Paper 1**:
   > A. Suresh, R. Bisht and S. Kamalasadan, "A Coordinated Control Architecture With Inverter-Based Resources and Legacy Controllers of Power Distribution System for Voltage Profile Balance," *IEEE Transactions on Industry Applications*, vol. 58, no. 5, pp. 6701-6712, Sept.-Oct. 2022, doi: [10.1109/TIA.2022.3183030](https://doi.org/10.1109/TIA.2022.3183030).

2. **Paper 2**:
   > A. Suresh, R. Bisht and S. Kamalasadan, "ADMM Based LQR for Voltage Regulation Using Distributed Energy Resources," *2020 IEEE International Conference on Power Electronics, Drives and Energy Systems (PEDES)*, 2020, pp. 1-6, doi: [10.1109/PEDES49360.2020.9379625](https://doi.org/10.1109/PEDES49360.2020.9379625).

