# Dataframe import
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

# Tree Visualisation
from sklearn.tree import export_graphviz

# from IPython.display import Image
import graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text

# Read feature matrix file
feature_matrix = pd.read_csv(
    "/Users/manuelphilip/Documents/Random_forest/mel_scaled_data.csv", sep=","
)

# Remove the catogirical variables and assign it to new dataframe
feature_matrix = feature_matrix.drop("response", axis=1)
feature_matrix = feature_matrix.drop("cell_barcode", axis=1)
feature_matrix = feature_matrix.drop("response", axis=1)
feature_colms = list(feature_matrix.columns)
# Assign predictor variable
pedictor_variable = feature_matrix["response"]

my_features = list(feature_matrix.columns)
feature_list = list(feature_matrix.columns)
my_features = np.array(my_features)

(
    feature_matrix_train,
    feature_matrix_test,
    pedictor_variable_train,
    pedictor_variable_test,
) = train_test_split(feature_matrix, pedictor_variable, test_size=0.2)

# Apply Random forest
model1 = RandomForestClassifier(n_estimators=2000)
model1.fit(feature_matrix_train, pedictor_variable_train)

# Get important features order by highest score
feature_importances = pd.Series(
    model1.feature_importances_, index=feature_matrix.columns, name="Values"
).sort_values(ascending=False)
features_dataframe = feature_importances.to_frame().reset_index()
features_list = list(features_dataframe["index"])
imp_feat = features_list[:99]

# Get associated features and number of samples of each individual model from important features
models = {}
associated_features = []
for each_model in model1.estimators_:
    features = [feature_colms[each_model] for each_model in each_model.tree_.feature]
    sample_count = [
        int(sample_index) for sample_index in each_model.tree_.n_node_samples
    ]
    associated_features.append(features)
    associated_features.append(sample_count)
    models[each_model] = associated_features
    associated_features = []


# Function for getting features and samples based on sample count defined by user
def get_associated_features(feat, sample_count, important_features):
    usr_defined_feat_with_sam_count = []
    usr_defined_feats = []
    ind_decision_tree = None
    samples = []
    all_asstd_feat_with_sam_count = []
    all_asstd_feats = []
    model_key_list = []
    depth = []
    for ind_decision_tree, model_val in models.items():
        feature = []
        feat_key = False
        important_feat_key = False
        for each_feature in model_val:
            samples = []
            if each_feature[0] in important_features:
                feature = each_feature[0 : len(each_feature)]
                important_feat_key = True
                if each_feature[0] == feat:
                    feat_key = True
            if type(each_feature[0]) == int and important_feat_key == True:
                samples = [
                    sample
                    for sample in each_feature[0 : len(each_feature)]
                    if sample >= sample_count
                ]
            if feat_key == True:
                if len(samples) > 0:
                    usr_defined_feat_with_sam_count.append(feature[0 : len(samples)])
                    usr_defined_feat_with_sam_count.append(samples)
                    usr_defined_feats.extend(feature[0 : len(samples)])
                    model_key_list.append(ind_decision_tree)
                    depth.append(samples)
            if important_feat_key == True:
                if len(samples) > 0:
                    all_asstd_feat_with_sam_count.append(feature[0 : len(samples)])
                    all_asstd_feat_with_sam_count.append(samples)
                    all_asstd_feats.extend(feature[0 : len(samples)])
    return (
        usr_defined_feat_with_sam_count,
        usr_defined_feats,
        all_asstd_feat_with_sam_count,
        all_asstd_feats,
        model_key_list,
        samples,
        depth,
    )


# Function to get unique list of all the features from the important feature/features
def get_total_genes(usr_defined_feats, all_asstd_feats):
    usr_def_total_feats = list(set(usr_defined_feats))
    all_total_feats = list(set(all_asstd_feats))
    return usr_def_total_feats, all_total_feats


# Function to get the corresponding tree for all the imporatant features
def get_trees(model_key, depth, feat, path):
    k = 0
    for model in model_key:
        file = path + str(feat) + "_" + str(len(depth[k])) + "_" + str(k) + ".dot"
        export_graphviz(
            model,
            out_file=file,
            feature_names=feature_matrix.columns,
            class_names=model1.classes_,
            rounded=True,
            proportion=False,
            precision=2,
            max_depth=len(depth[k]),
            filled=True,
        )
        k = k + 1


abl2 = get_associated_features(
    feat="ABL2", sample_count=1500, important_features=imp_feat
)
abl2[3]
comm_genes = get_total_genes(abl2[1], abl2[3])
type(comm_genes)
comm_genes[0]
list(set(imp_feat).intersection(comm_genes[1]))
tress_view = get_trees(
    model_key=abl2[4],
    depth=abl2[6],
    feat="ABL2",
    path="/Users/manuelphilip/Documents/Random_forest/trees/new_trees/imp_trees/test/",
)
