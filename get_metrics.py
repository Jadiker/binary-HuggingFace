# compare the predictions to the true labels
import pandas as pd
import numpy as np

TRUE_POSITIVES = "true_positives"
FALSE_POSITIVES = "false_positives"
TRUE_NEGATIVES = "true_negatives"
FALSE_NEGATIVES = "false_negatives"
ACCURACY = "accuracy"
PRECISION = "precision"
RECALL = "recall"
F1_SCORE = "f1_score"

# compute the F1 score metrics
def f1_score_metrics(y_true, y_pred):
    true_positives = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    false_positives = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    true_negatives = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    false_negatives = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * precision * recall / (precision + recall)
    return {TRUE_POSITIVES: true_positives, FALSE_POSITIVES: false_positives, TRUE_NEGATIVES: true_negatives, FALSE_NEGATIVES: false_negatives, ACCURACY: accuracy, PRECISION: precision, RECALL: recall, F1_SCORE: f1_score}

def get_metrics(dataset_file, predictions_file, output_file, verbose=False):
    # the test dataset is a csv file with a label column
    # the predictions file is a txt file with a prediction column (columns are separated by tabs)

    # load the datasets
    test_dataset = pd.read_csv(dataset_file)
    predictions = pd.read_csv(predictions_file, sep="\t")

    if verbose:
        # display the first few lines of each
        print("Here are the first few lines of the files:")
        print(test_dataset.head())
        print(predictions.head())

    # check to make sure the datasets have the same number of rows
    if test_dataset.shape[0] != predictions.shape[0]:
        raise ValueError(f"The test dataset has {test_dataset.shape[0]} rows, but the predictions file has {predictions.shape[0]} rows.")

    # get the y_true and y_pred
    y_true = test_dataset["label"]
    y_pred = predictions["prediction"]
    metrics = f1_score_metrics(y_true, y_pred)

    if verbose:
        # print the F1 score metrics
        print(f"True Positives: {metrics[TRUE_POSITIVES]}")
        print(f"False Positives: {metrics[FALSE_POSITIVES]}")
        print(f"True Negatives: {metrics[TRUE_NEGATIVES]}")
        print(f"False Negatives: {metrics[FALSE_NEGATIVES]}")
        print(f"Accuracy: {metrics[ACCURACY]}")
        print(f"Precision: {metrics[PRECISION]}")
        print(f"Recall: {metrics[RECALL]}")
        print(f"F1 Score: {metrics[F1_SCORE]}")

    # save the metrics to the file
    with open(output_file, "w") as f:
        f.write(f"True Positives: {metrics[TRUE_POSITIVES]}\n")
        f.write(f"False Positives: {metrics[FALSE_POSITIVES]}\n")
        f.write(f"True Negatives: {metrics[TRUE_NEGATIVES]}\n")
        f.write(f"False Negatives: {metrics[FALSE_NEGATIVES]}\n")
        f.write(f"Accuracy: {metrics[ACCURACY]}\n")
        f.write(f"Precision: {metrics[PRECISION]}\n")
        f.write(f"Recall: {metrics[RECALL]}\n")
        f.write(f"F1 Score: {metrics[F1_SCORE]}\n")

    return metrics

if __name__ == "__main__":
    TEST_DATASET = "created_datasets/source_datasets/unbalanced_microtext_test_dataset.csv"
    PREDICTIONS_FILE = "shared_results/prediction1.txt"
    OUTPUT_FILE = "shared_results/metrics1.txt"

    print(get_metrics(TEST_DATASET, PREDICTIONS_FILE, OUTPUT_FILE, verbose=True))