from data_processing import prepare_datasets
from sklearn.linear_model import LogisticRegression
import numpy as np


def main():
    key_label = "diagnosis"
    validation_split = 0.125
    source_train_dataset, target_train_dataset, target_val_dataset, target_test_dataset = prepare_datasets(key_label, validation_split)

    target_train_features = target_train_dataset.features
    target_train_labels = target_train_dataset.labels
    target_val_features = target_val_dataset.features
    target_val_labels = target_val_dataset.labels

    lr = LogisticRegression(random_state=0)
    lr.fit(target_train_features, target_train_labels)

    train_acc = lr.score(target_train_features, target_train_labels)
    val_acc = lr.score(target_val_features, target_val_labels)
    print("train acc:", train_acc)
    print("val acc:", val_acc)

    
    print(target_val_labels.value_counts())
    val_pred = lr.predict(target_val_features)
    print(np.asarray(np.unique(val_pred, return_counts=True)).T)

if __name__ == "__main__":
    main()