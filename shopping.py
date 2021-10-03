import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Comentado para depurar
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    # modificacion para depurar
    # evidence, labels = load_data("shopping.csv")    
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

# Function to convert month to number
def mes_a_numero(argument):
    switcher = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "Jun": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11
    }
    return switcher.get(argument, -1)

# Function to convert boolean to number
def bool_a_numero(argument):
    switcher = {
        "FALSE": 0,
        "TRUE": 1
    }
    return switcher.get(argument, 0)

# Function to convert visitor to number
def visitor_a_numero(argument):
    switcher = {
        "New_Visitor": 0,
        "Returning_Visitor": 1
    }
    return switcher.get(argument, 0)

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
    file = open(filename)
    csvreader = csv.reader(file)
    
    evidence = []
    labels = []
    i = 0
    for row in csvreader:
        if i > 0:
            fila = []           
            for j in range(0, 17):
                if j == 0 or j == 2 or j == 4 or j == 11 or j == 12 or j == 13 or j == 14:
                    fila.append(int(row[j]))
                elif j == 1 or j == 3 or j == 5 or j == 6 or j == 7 or j == 8 or j == 9: 
                    fila.append(float(row[j]))
                elif j == 10:
                    fila.append(mes_a_numero(row[10]))
                elif j == 15:
                    fila.append(visitor_a_numero(row[15]))
                elif j == 16:
                    fila.append(bool_a_numero(row[16]))            
                        
            evidence.append(fila)
            labels.append(bool_a_numero(row[17]))
            
        i+=1
        # print(f"{i} row: {row}      row[5]: {row[5]}")
    file.close()
    # evidence.pop(0)
    # labels.pop(0)

    # for i in range(185,195):        
    #     print(f"evidence:   {evidence[i]}")
    #     print(f"label:      {labels[i]}")
    
    return evidence, labels
    # raise NotImplementedError


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    return model.fit(evidence, labels)
    #raise NotImplementedError


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
    correctPositive = 0
    correctNegative = 0
    conteoPositivo = 0
    conteoNegativo = 0
    for observacion, prediccion in zip(labels, predictions):
        if observacion == 1:
            conteoPositivo += 1
            if observacion == prediccion:
                correctPositive += 1
        else:
            conteoNegativo += 1
            if observacion == prediccion:
                correctNegative += 1
        
    sensibilidad  = correctPositive / conteoPositivo
    especificidad= correctNegative / conteoNegativo
    return (sensibilidad, especificidad)
    #raise NotImplementedError


if __name__ == "__main__":
    main()
