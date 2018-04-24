import pandas
import os

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# Build a confusion matrix and
# calculate evaluation metrics using it
def summarize_metrics(tp, tn, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_score = (2 * recall * precision) / (recall + precision)

    print("Precison:", precision)
    print("Recall:", recall)
    print("Accuracy:", accuracy)
    print("F1 score:", f1_score)


if __name__ == "__main__":
    status_data = pandas.read_csv("mypersonality_final.csv")

    NEG_INDEX = 2
    POS_INDEX = 3
    NEU_INDEX = 4
    COMP_INDEX = 5

    # Annotate the status with sentiment scores
    # From nltk.sentiment.vader corpus
    if not os.path.isfile("mypersonality_cleaned.csv"):
        status_data.insert(NEG_INDEX, "sentiNEG", 0)
        status_data.insert(POS_INDEX, "sentiPOS", 0)
        status_data.insert(NEU_INDEX, "sentiNEU", 0)
        status_data.insert(COMP_INDEX, "sentiCOMPOUND", 0)

        sid = SentimentIntensityAnalyzer()
        count = 0
        for row in status_data.itertuples():
            """
            pos: positive
            neg: negative
            neu: neutral
            compound: aggregated score for the sentence
            """
            ss = sid.polarity_scores(row.STATUS)
            status_data.iloc[count, NEG_INDEX] = ss["neg"]
            status_data.iloc[count, POS_INDEX] = ss["pos"]
            status_data.iloc[count, NEU_INDEX] = ss["neu"]
            status_data.iloc[count, COMP_INDEX] = ss["compound"]
            count += 1

        status_data.to_csv("mypersonality_cleaned.csv")
    else:
        status_data = pandas.read_csv("mypersonality_cleaned.csv")

    # Drop NAs
    status_data = status_data.dropna()

    # We drop columns which give us a score for personality type
    status_data = status_data.drop(['STATUS', '#AUTHID', 'sEXT', 'sNEU', 'sAGR',
                                    'sCON', 'sOPN', 'DATE'], axis=1)

    # Drop non-normalized scores of Brokerage and Betweenness
    status_data = status_data.drop(['BROKERAGE', 'BETWEENNESS', 'NBROKERAGE',
                                    'NBETWEENNESS', 'DENSITY', 'TRANSITIVITY', 'NETWORKSIZE'], axis=1)

    # Change the name of first row from "Unknown" to "rowID"
    new_columns = status_data.columns.values
    new_columns[0] = "rowID"
    status_data.columns = new_columns

    # Put the columns to be predicted, at the end
    cols = status_data.columns.tolist()
    cols = cols[:5] + cols[5:10]
    status_data = status_data[cols]

    # 'y' for 1 and 'n' for 0
    features = ['cEXT', 'cNEU', 'cOPN', 'cAGR', 'cCON']
    for feature in features:
        status_data[feature] = status_data[feature].map({'y': 1.0, 'n': 0.0}).astype(int)

    # Split into training and test data: 66% and 33%
    train_data, test_data = train_test_split(status_data, test_size=0.50)

    train = train_data.values
    test = test_data.values

    # Build a classifier
    # k is chosen to be square root of number of training example
    model = KNeighborsClassifier(n_neighbors=250)
    model = model.fit(train[0:, 1:5], train[0:, 7])

    # Predict
    output = model.predict(test[:, 1:5])
    rowID = [TEST.rowID for TEST in test_data.itertuples()]
    result_df = pandas.DataFrame({"rowID": rowID,
                                  "cOPN": list(output)})

    # Build the confusion matrix to assess the model
    tp_count = 0
    tn_count = 0
    fp_count = 0
    fn_count = 0
    for row in rowID:
        test_cEXT_val = int(test_data.loc[test_data['rowID'] == row].cOPN)
        result_cEXT_val = int(result_df.loc[result_df['rowID'] == row].cOPN)
        if test_cEXT_val == 1:
            if result_cEXT_val == 1:
                tp_count += 1
            else:
                fn_count += 1
        else:
            if result_cEXT_val == 1:
                fp_count += 1
            else:
                tn_count += 1

    print(tp_count, tn_count, fp_count, fn_count)
    summarize_metrics(tp_count, tn_count, fp_count, fn_count)
