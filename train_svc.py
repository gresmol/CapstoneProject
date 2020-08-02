from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd


# Loads the dataset and splits it into training and testing sets
df = pd.read_csv('train.csv', header = 0, encoding='ISO-8859-1')
X = df.drop('Difficulty', 1)
Y = df['Difficulty']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# Initialises Random Forest classifier object
clf = svm.SVC(kernel='linear', C = 1.0)


def train_baseline_model():
    '''Trains and tests the benchmark model trained on 3 features'''
    X_train_baseline = X_train.filter(['FreqStandard', 'LenChar','PoS'], axis=1)
    X_test_baseline = X_test.filter(['FreqStandard', 'LenChar', 'PoS'], axis=1)
    clf.fit(X_train_baseline, y_train)
    y_pred = clf.predict(X_test_baseline)
    return y_pred


def train_model():
    '''Trains and tests the refined model trained on the full set of features'''
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


def model_classification_report(y_test, y_pred):
    '''Evaluated the model in terms of F1 score, accuracy, recall and precision'''
    return classification_report(y_test, y_pred)


def model_accuracy(y_test, y_pred):
    '''Returns accuracy score only'''
    result = 'Accuracy score: ' + str(accuracy_score(y_test, y_pred))
    return result


if __name__ == "__main__":
    print('\nBaseline model results (trained on 3 features):')
    y_pred = train_baseline_model()
    print(model_classification_report(y_test, y_pred))
    print(model_accuracy(y_test, y_pred))
    print('\nModel results (trained on all features):')
    y_pred = train_model()
    print(model_classification_report(y_test, y_pred))
    print(model_accuracy(y_test, y_pred))
