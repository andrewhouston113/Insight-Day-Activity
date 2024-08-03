![splash](splash.png)

## Overview

Welcome to the Barts Life Sciences Data Science Insight Day repository! This repository contains a Python script designed to offer participants a hands-on experience with data science concepts. The script walks through various stages of a typical data science workflow, from data loading and exploration to visualisation and predictive modeling

## Repository Contents

- **Python Script (`diabetes_analysis.py`)**: This script contains all the necessary code for loading the dataset, performing exploratory data analysis (EDA), visualizing data, and building a basic logistic regression model to predict diabetes outcomes.
- **Python Notebook (`diabetes_analysis.ipynb`)**: This is a notebook version of the script.

## Getting Set-Up

### Pre-requisites
To complete this activity ensure you have the following software installed:

- Anaconda (https://www.anaconda.com/)
- Visual Studio Code (https://code.visualstudio.com/)
- Git (https://git-scm.com/download/win)

### Cloning the Repository
To get the relevant files onto your computer, open the command line and enter the following commands:

```bash
cd <FILE PATH OF WHERE YOU WANT THE REPOSITORY>
git clone https://github.com/andrewhouston113/Insight-Day-Activity
```

### Setting up your Environment
It is good practice to always work within a virtual environment. To set up your virtual environment, open the command line and enter the following command:

```bash
conda create --name bls_insight_day python=3.9
conda activate bls_insight_day
```

To run the script, ensure you have the following Python libraries installed:

- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `ipykernel`

You can install these libraries using pip in the command line:

```bash
pip install pandas matplotlib seaborn scikit-learn ipykernel
```

To work with the notebook in Visual Studio Code, we need to create our kernel. Simply copy paste this code into the command line:

```bash
python -m ipykernel install --user --name=bls_insight_day --display-name "bls_insight_day"
```

You can now close the command line, open Visual Studio Code and begin the task.
