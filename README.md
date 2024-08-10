# GMSL Prediction Model

This repository contains a trained machine learning model for predicting Global Mean Sea Level (GMSL) values for the next 20 years, using historical data and associated uncertainty. The model is saved using joblib and can be loaded and executed on any system following the steps outlined below.
Prerequisites
# 1. Python
* Ensure that Python 3.x is installed on your system. You can download it from python.org.

# 2. Virtual Environment (Optional but Recommended)

* It's recommended to create a virtual environment to manage dependencies.
Create a Virtual Environment
# Activate the Virtual Environment

# Bash
    python -m venv gmsl_env

# Windows:
    gmsl_env\Scripts\activate

# 3. Install Required Libraries

* Install the required Python libraries using pip.

# Bash

    pip install pandas numpy scikit-learn matplotlib joblib


# How to Load and Execute the Model
## 1. Clone the Repository

Clone this repository to your local machine.

# Bash

    git clone <repository-url>
    cd <repository-directory>

# 2. Place Files in the Directory

Ensure that #sea-tide-level.pkl# and #sea.csv# are in the same directory as the Python script.

# 3. Run the Python Script

With the virtual environment activated, run the script to load the model, make predictions, and visualize the results.

# bash

    python sea-level.py

# 4. View the Output

The script will output:

    Predicted GMSL values for the next 20 years.
    Visualization of historical GMSL data, predictions, and uncertainty.

# 5. Troubleshooting

    EOFError: If you encounter an EOFError while loading the model, ensure that the model file is not corrupted or incomplete. Try re-downloading the file if necessary.
    Library Compatibility: Ensure that the versions of Python and the installed libraries are compatible with the ones used to create the model.

Customizing the Script

You can modify the script to:

    Change the range of years for predictions.
    Adjust the visualization or add additional features.
    Experiment with different models or data.

License
