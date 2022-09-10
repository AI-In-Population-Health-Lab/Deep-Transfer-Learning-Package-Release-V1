import pandas as pd
import math
from dataset import Dataset
from collections import Counter



def parse_csv(csv_path, label_key):
    """
    Parses the .csv into a features dataframe and labels dataframe. Transforms
    categorical variables into binary dummy variables and labels into ordinal
    indices (for cross entropy loss).
    args:
        csv_path (string): path to the data as .csv
        label_key (string): column name giving the the data label
    return:
        df_features (dataframe): features
        df_labels (dataframe): labels
    """
    df = pd.read_csv(csv_path)
    #df = df.sample(frac=1).reset_index(drop=True) # shuffle the data
    df[label_key] = pd.Categorical(df[label_key]) # set label to Categorical for separate processing
    df[label_key + "-codes"] = df[label_key].cat.codes # add codes column for label (e.g. 0, 1, 2)
    df_features = df.loc[:, ~df.columns.str.startswith(label_key)]
    df_features = df_features.loc[:, ~df_features.columns.str.startswith("ID")]
    df_features = df_features.loc[:, ~df_features.columns.str.startswith("admit_date")]
    df_features = df_features.loc[:, ~df_features.columns.str.startswith("admityear")]
    df_labels = df.loc[:, label_key + "-codes"]
    return df_features, df_labels

def prepare_datasets(source_train_path, target_train_path, target_test_path, label_key, validation_split):
    """
    Prepares source train, target train, target validation, and target test 
    data. The target validation data is taken from the target training data.
    args:
        label_key (string): column name giving the data label
        validation_split (float): percentage of target train to use as validation data
    return:
        source_train (Dataset), target_train (Dataset), target_val (Dataset), 
        target_test (Dataset)
    """
    # parse .csv's, split into features and labels, and convert categorical
    # features into dummy variables
    source_train_features, source_train_labels = parse_csv(source_train_path, label_key)
    source_train_features_dummies = pd.get_dummies(source_train_features) # TODO: assumes all features are categorical?


    target_train_features_overall, target_train_labels_overall = parse_csv(target_train_path, label_key)
    target_train_features_dummies_overall = pd.get_dummies(target_train_features_overall)

    target_test_features, target_test_labels = parse_csv(target_test_path, label_key)
    target_test_features_dummies = pd.get_dummies(target_test_features)

 
    # ensure that all feature data have the same set of columns
    all_feature_cols = source_train_features_dummies.keys().union(target_train_features_dummies_overall.keys().union(target_test_features_dummies.keys()))

    source_train_diff_cols = all_feature_cols.difference(source_train_features_dummies.keys())
    for col in source_train_diff_cols:
        source_train_features_dummies[col] = [0 for _ in range(source_train_features_dummies.shape[0])]
    
    target_train_diff_cols = all_feature_cols.difference(target_train_features_dummies_overall.keys())
    for col in target_train_diff_cols:
        target_train_features_dummies_overall[col] = [0 for _ in range(target_train_features_dummies_overall.shape[0])]

    target_test_diff_cols = all_feature_cols.difference(target_test_features_dummies.keys())
    for col in target_test_diff_cols:
        target_test_features_dummies[col] = [0 for _ in range(target_test_features_dummies.shape[0])]
    
    # split target data into training and validation, last few as validation set

    target_val_size = math.floor(len(target_train_features_dummies_overall.index) * validation_split)
    target_train_size = len(target_train_features_dummies_overall.index) - target_val_size
    target_train_features_dummies = target_train_features_dummies_overall.loc[0:target_train_size, :]
    target_train_labels = target_train_labels_overall.loc[0:target_train_size]
    target_val_features_dummies = target_train_features_dummies_overall.loc[(target_train_size + 1):, :]
    target_val_labels = target_train_labels_overall.loc[(target_train_size + 1):]

    print("target_train_labels", target_train_labels.shape)

    # wrap features and labels in Dataset objects
    source_train_dataset = Dataset(source_train_features_dummies, source_train_labels)
    target_train_dataset = Dataset(target_train_features_dummies, target_train_labels)
    target_val_dataset = Dataset(target_val_features_dummies, target_val_labels)
    target_test_dataset = Dataset(target_test_features_dummies, target_test_labels)

    return source_train_dataset, target_train_dataset, target_val_dataset, target_test_dataset


def prepare_datasets_returnSourceVal(source_train_path, target_train_path, target_test_path, label_key, validation_split):
    """
    Prepares source train, target train, target validation, and target test
    data. The target validation data is taken from the source training data.
    args:
        label_key (string): column name giving the data label
        validation_split (float): percentage of target train to use as validation data
    return:
        source_train (Dataset), target_train (Dataset), target_val (Dataset),
        target_test (Dataset)
    """
    # parse .csv's, split into features and labels, and convert categorical
    # features into dummy variables
    source_train_features, source_train_labels = parse_csv(source_train_path, label_key)
    source_train_features_dummies = pd.get_dummies(source_train_features)  # TODO: assumes all features are categorical?

    target_train_features_overall, target_train_labels_overall = parse_csv(target_train_path, label_key)
    target_train_features_dummies_overall = pd.get_dummies(target_train_features_overall)



    target_test_features, target_test_labels = parse_csv(target_test_path, label_key)
    target_test_features_dummies = pd.get_dummies(target_test_features)

    # ensure that all feature data have the same set of columns
    all_feature_cols = source_train_features_dummies.keys().union(
        target_train_features_dummies_overall.keys().union(target_test_features_dummies.keys()))

    source_train_diff_cols = all_feature_cols.difference(source_train_features_dummies.keys())
    for col in source_train_diff_cols:
        source_train_features_dummies[col] = [0 for _ in range(source_train_features_dummies.shape[0])]

    target_train_diff_cols = all_feature_cols.difference(target_train_features_dummies_overall.keys())
    for col in target_train_diff_cols:
        target_train_features_dummies_overall[col] = [0 for _ in range(target_train_features_dummies_overall.shape[0])]

    target_test_diff_cols = all_feature_cols.difference(target_test_features_dummies.keys())
    for col in target_test_diff_cols:
        target_test_features_dummies[col] = [0 for _ in range(target_test_features_dummies.shape[0])]

    source_train_features_dummies_overall = source_train_features_dummies
    source_train_labels_overall = source_train_labels

    

    target_test_dataset = Dataset(target_test_features_dummies, target_test_labels)
    ## newly added target train dataset
    target_train_dataset = Dataset(target_train_features_dummies_overall, target_train_labels_overall)

    source_val_size = math.floor(len(source_train_features_dummies_overall.index) * validation_split)
    source_train_size = len(source_train_features_dummies_overall.index) - source_val_size
    source_train_features_dummies = source_train_features_dummies_overall.loc[0:source_train_size, :]
    source_train_labels = source_train_labels_overall.loc[0:source_train_size]
    source_val_features_dummies = source_train_features_dummies_overall.loc[(source_train_size + 1):, :]
    source_val_labels = source_train_labels_overall.loc[(source_train_size + 1):]

    # wrap features and labels in Dataset objects
    source_train_dataset = Dataset(source_train_features_dummies, source_train_labels)
    source_val_dataset = Dataset(source_val_features_dummies, source_val_labels)

    print("source_train_feature_dummy:" + str(source_train_features_dummies.shape))
    print("source_val_feature_dummy:"+ str(source_val_features_dummies.shape))
    print("target_test_feature_dummy:" + str(target_test_features_dummies.shape))

    return source_train_dataset, source_val_dataset, target_test_dataset, target_train_dataset


def prepare_datasets_stratify(source_train_path, target_train_path, target_test_path, label_key, validation_split):
    """
    Prepares source train, target train, target validation, and target test
    data. The target validation data is taken from the target training data.
    When preparing the validation data, using stratify approach for each class to do the split.
    args:
        label_key (string): column name giving the data label
        validation_split (float): percentage of target train to use as validation data
    return:
        source_train (Dataset), target_train (Dataset), target_val (Dataset),
        target_test (Dataset)
    """
    # parse .csv's, split into features and labels, and convert categorical
    # features into dummy variables
    source_train_features, source_train_labels = parse_csv(source_train_path, label_key)
    source_train_features_dummies = pd.get_dummies(source_train_features)  # TODO: assumes all features are categorical?
    target_train_features_overall, target_train_labels_overall = parse_csv(target_train_path, label_key)
    target_train_features_dummies_overall = pd.get_dummies(target_train_features_overall)
    target_test_features, target_test_labels = parse_csv(target_test_path, label_key)
    target_test_features_dummies = pd.get_dummies(target_test_features)
    # ensure that all feature data have the same set of columns
    all_feature_cols = source_train_features_dummies.keys().union(
        target_train_features_dummies_overall.keys().union(target_test_features_dummies.keys()))
    source_train_diff_cols = all_feature_cols.difference(source_train_features_dummies.keys())
    for col in source_train_diff_cols:
        source_train_features_dummies[col] = [0 for _ in range(source_train_features_dummies.shape[0])]
    target_train_diff_cols = all_feature_cols.difference(target_train_features_dummies_overall.keys())
    for col in target_train_diff_cols:
        target_train_features_dummies_overall[col] = [0 for _ in range(target_train_features_dummies_overall.shape[0])]
    target_test_diff_cols = all_feature_cols.difference(target_test_features_dummies.keys())
    for col in target_test_diff_cols:
        target_test_features_dummies[col] = [0 for _ in range(target_test_features_dummies.shape[0])]

    # now, we change to stratified split
    num_target_train = len(target_train_features_dummies_overall.index)
    target_train_features_dummies = pd.DataFrame(columns=target_train_features_dummies_overall.columns)
    target_train_labels = pd.Series()
    target_val_features_dummies = pd.DataFrame(columns=target_train_features_dummies_overall.columns)
    target_val_labels = pd.Series()
    for key in Counter(target_train_labels_overall).keys():
        temp_target_train_features_dummies_overall = target_train_features_dummies_overall.copy(deep=True)
        temp_target_train_labels_overall = target_train_labels_overall.copy(deep=True)
        for temp_index in range(num_target_train-1,-1,-1):
            if temp_target_train_labels_overall[temp_index]!=key:
                temp_target_train_labels_overall.drop(index=temp_index,inplace=True)
                temp_target_train_features_dummies_overall.drop(index=temp_index,inplace=True)
        temp_target_train_features_dummies, temp_target_train_labels, temp_target_val_features_dummies, temp_target_val_labels \
            = split(temp_target_train_features_dummies_overall,temp_target_train_labels_overall,validation_split)
        target_train_features_dummies = target_train_features_dummies.append(temp_target_train_features_dummies)
        target_train_labels = target_train_labels.append(temp_target_train_labels)
        target_val_features_dummies = target_val_features_dummies.append(temp_target_val_features_dummies)
        target_val_labels = target_val_labels.append(temp_target_val_labels)

    # wrap features and labels in Dataset objects
    source_train_dataset = Dataset(source_train_features_dummies, source_train_labels)
    target_train_dataset = Dataset(target_train_features_dummies, target_train_labels)
    target_val_dataset = Dataset(target_val_features_dummies, target_val_labels)
    target_test_dataset = Dataset(target_test_features_dummies, target_test_labels)

    #print("target_train_labels",target_train_labels.shape)
    #print("target_train_dataset",Counter(target_train_labels))
    return source_train_dataset, target_train_dataset, target_val_dataset, target_test_dataset

def prepare_datasets_stratify_returnSourceVal(source_train_path, target_train_path, target_test_path, label_key, validation_split):
    """
    Prepares source train, target train, target validation, and target test
    data. The target validation data is taken from the target training data.
    args:
        label_key (string): column name giving the data label
        validation_split (float): percentage of target train to use as validation data
    return:
        source_train (Dataset), target_train (Dataset), target_val (Dataset),
        target_test (Dataset)
    """
    # parse .csv's, split into features and labels, and convert categorical
    # features into dummy variables
    source_train_features, source_train_labels = parse_csv(source_train_path, label_key)
    source_train_features_dummies = pd.get_dummies(source_train_features)  # TODO: assumes all features are categorical?
    target_train_features_overall, target_train_labels_overall = parse_csv(target_train_path, label_key)
    target_train_features_dummies_overall = pd.get_dummies(target_train_features_overall)
    target_test_features, target_test_labels = parse_csv(target_test_path, label_key)
    target_test_features_dummies = pd.get_dummies(target_test_features)
    # ensure that all feature data have the same set of columns
    all_feature_cols = source_train_features_dummies.keys().union(
        target_train_features_dummies_overall.keys().union(target_test_features_dummies.keys()))
    source_train_diff_cols = all_feature_cols.difference(source_train_features_dummies.keys())
    for col in source_train_diff_cols:
        source_train_features_dummies[col] = [0 for _ in range(source_train_features_dummies.shape[0])]
    target_train_diff_cols = all_feature_cols.difference(target_train_features_dummies_overall.keys())
    for col in target_train_diff_cols:
        target_train_features_dummies_overall[col] = [0 for _ in range(target_train_features_dummies_overall.shape[0])]
    target_test_diff_cols = all_feature_cols.difference(target_test_features_dummies.keys())
    for col in target_test_diff_cols:
        target_test_features_dummies[col] = [0 for _ in range(target_test_features_dummies.shape[0])]

    # now, we change to stratified split
    source_train_features_dummies_overall = source_train_features_dummies.copy(deep=True)
    source_train_labels_overall = source_train_labels.copy(deep=True)
    num_source_train = len(source_train_features_dummies_overall.index)
    source_train_features_dummies = pd.DataFrame(columns=source_train_features_dummies_overall.columns)
    source_train_labels = pd.Series()
    source_val_features_dummies = pd.DataFrame(columns=source_train_features_dummies_overall.columns)
    source_val_labels = pd.Series()
    for key in Counter(source_train_labels_overall).keys():
        temp_source_train_features_dummies_overall = source_train_features_dummies_overall.copy(deep=True)
        temp_source_train_labels_overall = source_train_labels_overall.copy(deep=True)
        for temp_index in range(num_source_train - 1, -1, -1):
            if temp_source_train_labels_overall[temp_index] != key:
                temp_source_train_labels_overall.drop(index=temp_index, inplace=True)
                temp_source_train_features_dummies_overall.drop(index=temp_index, inplace=True)
        temp_source_train_features_dummies, temp_source_train_labels, temp_source_val_features_dummies, temp_source_val_labels \
            = split(temp_source_train_features_dummies_overall, temp_source_train_labels_overall, validation_split)
        source_train_features_dummies = source_train_features_dummies.append(temp_source_train_features_dummies)
        source_train_labels = source_train_labels.append(temp_source_train_labels)
        source_val_features_dummies = source_val_features_dummies.append(temp_source_val_features_dummies)
        source_val_labels = source_val_labels.append(temp_source_val_labels)


    # wrap features and labels in Dataset objects
    source_train_dataset = Dataset(source_train_features_dummies, source_train_labels)
    source_val_dataset = Dataset(source_val_features_dummies, source_val_labels)
    target_test_dataset = Dataset(target_test_features_dummies, target_test_labels)

    print("source_train_feature_dummy:" + str(source_train_features_dummies.shape))
    print("source_val_feature_dummy:"+ str(source_val_features_dummies.shape))
    print("target_test_feature_dummy:" + str(target_test_features_dummies.shape))

    return source_train_dataset, source_val_dataset, target_test_dataset

def prepare_datasets_stratify_returnSourceVal(source_train_path, target_train_path, target_test_path, label_key, validation_split):
    """
    Prepares source train, target train, target validation, and target test
    data. The target validation data is taken from the target training data.
    When preparing the validation data, using stratify approach for each class to do the split.
    args:
        label_key (string): column name giving the data label
        validation_split (float): percentage of target train to use as validation data
    return:
        source_train (Dataset), target_train (Dataset), target_val (Dataset),
        target_test (Dataset)
    """
    # parse .csv's, split into features and labels, and convert categorical
    # features into dummy variables
    source_train_features_overall, source_train_labels_overall = parse_csv(source_train_path, label_key)
    source_train_features_dummies_overall = pd.get_dummies(source_train_features_overall)  # TODO: assumes all features are categorical?
    target_train_features_overall, target_train_labels_overall = parse_csv(target_train_path, label_key)
    target_train_features_dummies_overall = pd.get_dummies(target_train_features_overall)
    target_test_features, target_test_labels = parse_csv(target_test_path, label_key)
    target_test_features_dummies = pd.get_dummies(target_test_features)
    # ensure that all feature data have the same set of columns
    all_feature_cols = source_train_features_dummies_overall.keys().union(
        target_train_features_dummies_overall.keys().union(target_test_features_dummies.keys()))
    source_train_diff_cols = all_feature_cols.difference(source_train_features_dummies_overall.keys())
    for col in source_train_diff_cols:
        source_train_features_dummies_overall[col] = [0 for _ in range(source_train_features_dummies_overall.shape[0])]
    target_train_diff_cols = all_feature_cols.difference(target_train_features_dummies_overall.keys())
    for col in target_train_diff_cols:
        target_train_features_dummies_overall[col] = [0 for _ in range(target_train_features_dummies_overall.shape[0])]
    target_test_diff_cols = all_feature_cols.difference(target_test_features_dummies.keys())
    for col in target_test_diff_cols:
        target_test_features_dummies[col] = [0 for _ in range(target_test_features_dummies.shape[0])]

    # now, we change to stratified split
    num_source_train = len(source_train_features_dummies_overall.index)
    source_train_features_dummies = pd.DataFrame(columns=source_train_features_dummies_overall.columns)
    source_train_labels = pd.Series()
    source_val_features_dummies = pd.DataFrame(columns=source_train_features_dummies_overall.columns)
    source_val_labels = pd.Series()
    for key in Counter(source_train_labels_overall).keys():
        temp_source_train_features_dummies_overall = source_train_features_dummies_overall.copy(deep=True)
        temp_source_train_labels_overall = source_train_labels_overall.copy(deep=True)
        for temp_index in range(num_source_train - 1, -1, -1):
            if temp_source_train_labels_overall[temp_index] != key:
                temp_source_train_labels_overall.drop(index=temp_index, inplace=True)
                temp_source_train_features_dummies_overall.drop(index=temp_index, inplace=True)
        temp_source_train_features_dummies, temp_source_train_labels, temp_source_val_features_dummies, temp_source_val_labels \
            = split(temp_source_train_features_dummies_overall, temp_source_train_labels_overall, validation_split)
        source_train_features_dummies = source_train_features_dummies.append(temp_source_train_features_dummies)
        source_train_labels = source_train_labels.append(temp_source_train_labels)
        source_val_features_dummies = source_val_features_dummies.append(temp_source_val_features_dummies)
        source_val_labels = source_val_labels.append(temp_source_val_labels)

    # wrap features and labels in Dataset objects
    source_train_dataset = Dataset(source_train_features_dummies, source_train_labels)
    source_val_dataset = Dataset(source_val_features_dummies, source_val_labels)
    #target_train_dataset = Dataset(target_train_features_dummies_overall, target_train_labels_overall)
    target_test_dataset = Dataset(target_test_features_dummies, target_test_labels)

    return source_train_dataset, source_val_dataset, target_test_dataset

def split(t_target_train_feature_dummies_overall,t_target_train_labels_overall,validation_split):

    t_target_train_feature_dummies_overall = t_target_train_feature_dummies_overall.reset_index(drop=True)
    t_target_train_labels_overall = t_target_train_labels_overall.reset_index(drop=True)
    t_target_val_size = math.floor(len(t_target_train_feature_dummies_overall.index) * validation_split)
    t_target_train_size = len(t_target_train_feature_dummies_overall.index) - t_target_val_size
    t_target_train_features_dummies = t_target_train_feature_dummies_overall.loc[0:t_target_train_size-1, :]
    t_target_train_labels = t_target_train_labels_overall.loc[0:t_target_train_size-1]

    t_target_val_features_dummies = t_target_train_feature_dummies_overall.loc[t_target_train_size:, :]
    t_target_val_labels = t_target_train_labels_overall.loc[t_target_train_size:]
    return t_target_train_features_dummies, t_target_train_labels, t_target_val_features_dummies, t_target_val_labels


def prepare_datasets_stratify_combineSourceTarget(source_train_path, target_train_path, target_test_path, label_key, validation_split):
    """
    Prepares source train, target train, target validation, and target test
    data. The target validation data is taken from the target training data.
    When preparing the validation data, using stratify approach for each class to do the split.
    args:
        label_key (string): column name giving the data label
        validation_split (float): percentage of target train to use as validation data
    return:
        source_train (Dataset), target_train (Dataset), target_val (Dataset),
        target_test (Dataset)
    """
    # parse .csv's, split into features and labels, and convert categorical
    # features into dummy variables
    source_train_features, source_train_labels = parse_csv(source_train_path, label_key)
    source_train_features_dummies = pd.get_dummies(source_train_features)  # TODO: assumes all features are categorical?
    target_train_features_overall, target_train_labels_overall = parse_csv(target_train_path, label_key)
    target_train_features_dummies_overall = pd.get_dummies(target_train_features_overall)
    target_test_features, target_test_labels = parse_csv(target_test_path, label_key)
    target_test_features_dummies = pd.get_dummies(target_test_features)
    # ensure that all feature data have the same set of columns
    all_feature_cols = source_train_features_dummies.keys().union(
        target_train_features_dummies_overall.keys().union(target_test_features_dummies.keys()))
    source_train_diff_cols = all_feature_cols.difference(source_train_features_dummies.keys())
    for col in source_train_diff_cols:
        source_train_features_dummies[col] = [0 for _ in range(source_train_features_dummies.shape[0])]
    target_train_diff_cols = all_feature_cols.difference(target_train_features_dummies_overall.keys())
    for col in target_train_diff_cols:
        target_train_features_dummies_overall[col] = [0 for _ in range(target_train_features_dummies_overall.shape[0])]
    target_test_diff_cols = all_feature_cols.difference(target_test_features_dummies.keys())
    for col in target_test_diff_cols:
        target_test_features_dummies[col] = [0 for _ in range(target_test_features_dummies.shape[0])]

    # now, we change to stratified split
    num_target_train = len(target_train_features_dummies_overall.index)
    target_train_features_dummies = pd.DataFrame(columns=target_train_features_dummies_overall.columns)
    target_train_labels = pd.Series()
    target_val_features_dummies = pd.DataFrame(columns=target_train_features_dummies_overall.columns)
    target_val_labels = pd.Series()
    for key in Counter(target_train_labels_overall).keys():
        temp_target_train_features_dummies_overall = target_train_features_dummies_overall.copy(deep=True)
        temp_target_train_labels_overall = target_train_labels_overall.copy(deep=True)
        for temp_index in range(num_target_train-1,-1,-1):
            if temp_target_train_labels_overall[temp_index]!=key:
                temp_target_train_labels_overall.drop(index=temp_index,inplace=True)
                temp_target_train_features_dummies_overall.drop(index=temp_index,inplace=True)
        temp_target_train_features_dummies, temp_target_train_labels, temp_target_val_features_dummies, temp_target_val_labels \
            = split(temp_target_train_features_dummies_overall,temp_target_train_labels_overall,validation_split)
        target_train_features_dummies = target_train_features_dummies.append(temp_target_train_features_dummies)
        target_train_labels = target_train_labels.append(temp_target_train_labels)
        target_val_features_dummies = target_val_features_dummies.append(temp_target_val_features_dummies)
        target_val_labels = target_val_labels.append(temp_target_val_labels)

    # wrap features and labels in Dataset objects
    combine_train_features_dummies = source_train_features_dummies.copy(deep=True)
    combine_train_features_dummies.append(target_train_features_dummies)
    combine_train_labels = source_train_labels.copy(deep=True)
    combine_train_labels.append(target_train_labels)

    combine_train_dataset = Dataset(combine_train_features_dummies, combine_train_labels)
    target_val_dataset = Dataset(target_val_features_dummies, target_val_labels)
    target_test_dataset = Dataset(target_test_features_dummies, target_test_labels)

    #print("target_train_labels",target_train_labels.shape)
    #print("target_train_dataset",Counter(target_train_labels))
    return combine_train_dataset, target_val_dataset, target_test_dataset


def prepare_datasets_combineSourceTarget(source_train_path, target_train_path, target_test_path, label_key, validation_split):
    """
    Prepares source train, target train, target validation, and target test
    data. The target validation data is taken from the target training data.
    args:
        label_key (string): column name giving the data label
        validation_split (float): percentage of target train to use as validation data
    return:
        source_train (Dataset), target_train (Dataset), target_val (Dataset),
        target_test (Dataset)
    """
    # parse .csv's, split into features and labels, and convert categorical
    # features into dummy variables
    source_train_features, source_train_labels = parse_csv(source_train_path, label_key)
    source_train_features_dummies = pd.get_dummies(source_train_features)  # TODO: assumes all features are categorical?

    target_train_features_overall, target_train_labels_overall = parse_csv(target_train_path, label_key)
    target_train_features_dummies_overall = pd.get_dummies(target_train_features_overall)

    target_test_features, target_test_labels = parse_csv(target_test_path, label_key)
    target_test_features_dummies = pd.get_dummies(target_test_features)


    # ensure that all feature data have the same set of columns
    all_feature_cols = source_train_features_dummies.keys().union(
        target_train_features_dummies_overall.keys().union(target_test_features_dummies.keys()))

    source_train_diff_cols = all_feature_cols.difference(source_train_features_dummies.keys())
    for col in source_train_diff_cols:
        source_train_features_dummies[col] = [0 for _ in range(source_train_features_dummies.shape[0])]

    target_train_diff_cols = all_feature_cols.difference(target_train_features_dummies_overall.keys())
    for col in target_train_diff_cols:
        target_train_features_dummies_overall[col] = [0 for _ in range(target_train_features_dummies_overall.shape[0])]

    target_test_diff_cols = all_feature_cols.difference(target_test_features_dummies.keys())
    for col in target_test_diff_cols:
        target_test_features_dummies[col] = [0 for _ in range(target_test_features_dummies.shape[0])]

    # now, we change to last few as validation set

    target_val_size = math.floor(len(target_train_features_dummies_overall.index) * validation_split)
    target_train_size = len(target_train_features_dummies_overall.index) - target_val_size
    target_train_features_dummies = target_train_features_dummies_overall.loc[0:target_train_size, :]
    target_train_labels = target_train_labels_overall.loc[0:target_train_size]
    target_val_features_dummies = target_train_features_dummies_overall.loc[(target_train_size + 1):, :]
    target_val_labels = target_train_labels_overall.loc[(target_train_size + 1):]

    combine_train_features_dummies = source_train_features_dummies.copy(deep=True)
    combine_train_features_dummies.append(target_train_features_dummies)
    combine_train_labels = source_train_labels.copy(deep=True)
    combine_train_labels.append(target_train_labels)

    # wrap features and labels in Dataset objects
    combine_train_dataset = Dataset(combine_train_features_dummies, combine_train_labels)
    target_val_dataset = Dataset(target_val_features_dummies, target_val_labels)
    target_test_dataset = Dataset(target_test_features_dummies, target_test_labels)


    return combine_train_dataset, target_val_dataset, target_test_dataset
