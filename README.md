# Feature-focused-Random-forest

A Repo to extract important features and associated decision trees that contribute to the prediction variables from a Random Forest model. The repo contains a python script ```random_forest.py``` written to retrieve associated features of important features that are downstream of a decision tree from a random forest model. Furthermore from the bioinformatics analysis perespective a python script ```string_api.py``` to extract interactions assoicated with important features(here genes) from STRING database. (https://string-db.org/cgi/help?sessionId=bcOVN42GVzPj&subpage=api)

## Usage
For the random forest an input matrix file that contains features as columns and predictor variable as rows.
To find the interaction genes from STRING database, input a list of genes of interest.
