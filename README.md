# Feature-Extraction-from-Random-forest

This repository contains a Python script called ```random_forest.py``` that extracts important features and associated decision trees from a Random Forest model. The script retrieves downstream features of important features from a decision tree. Additionally, for bioinformatics analysis, a Python script called `string_api.py` is included that extracts interactions associated with important features (here, genes) from the STRING database. (https://string-db.org/cgi/help?sessionId=bcOVN42GVzPj&subpage=api)

## Usage
For the random forest an input matrix file that contains features as columns and predictor variable as rows.
To find the interaction genes from STRING database, input a list of genes of interest.
