import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4
model = KNeighborsClassifier(n_neighbors=1)


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter= ',')
        lines = []
        for row in reader:
            lines.append(row)
        column_titles = lines[0]
        lines = lines[1:] # skip title line
        evidence = []
        labels = []
        for line in lines:
            evidence_line = []
            try:
                evidence_line.append(int(line[0]))
                evidence_line.append(float(line[1]))
                evidence_line.append(int(line[2]))
                evidence_line.append(float(line[3]))
                evidence_line.append(int(line[4]))
                evidence_line.append(float(line[5]))
                evidence_line.append(float(line[6]))
                evidence_line.append(float(line[7]))
                evidence_line.append(float(line[8]))
                evidence_line.append(float(line[9]))
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                evidence_line.append(months.index(line[10][0:3]))
                evidence_line.append(int(line[11]))
                evidence_line.append(int(line[12]))
                evidence_line.append(int(line[13]))
                evidence_line.append(int(line[14]))
                evidence_line.append(1 if line[15] == 'Returning_Visitor' else 0)
                evidence_line.append(1 if line[16] == 'TRUE' else 0)
            except:
                print('skipping one line')
                continue
            evidence.append(evidence_line)
            labels.append(1 if line[17] == 'TRUE' else 0)
    return (evidence, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    return(model.fit(evidence, labels))

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    total_tests = 0
    true_positives = 0
    true_negatives = 0
    correct_positives = 0
    correct_negatives = 0
    for i in range(len(labels)):
        total_tests += 1
        if labels[i] == 1:
            true_positives += 1
            if labels[i] == predictions[i]:
                correct_positives += 1
        else:
            true_negatives += 1
            if labels[i] == predictions[i]:
                correct_negatives += 1
    sensitivity = correct_positives / true_positives
    specificity = correct_negatives / true_negatives
    return(sensitivity, specificity)

if __name__ == "__main__":
    main()
