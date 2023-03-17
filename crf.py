import sklearn
import csv
import sys
import sklearn_crfsuite
from sklearn_crfsuite import metrics


## based on https://github.com/cltl/ba-text-mining/blob/master/lab_sessions/lab4/Lab4a.4-NERC-CRF-Dutch.ipynb


def token2features(sentence, i):

    token = sentence[i][0]
    postag = sentence[i][1]
    
    features = {
        'bias': 1.0,
        'token': token.lower(),
        'postag': postag
    }
    if i == 0:
        features['BOS'] = True
    elif i == len(sentence) -1:
        features['EOS'] = True
        
    return features

def sent2features(sent):
    return [token2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    #if you added features to your input file, make sure to add them here as well.
    return [label for token, postag, chunklabel, label in sent]

def sent2tokens(sent):
    return [token for token, postag, chunklabel, label in sent]
    
    
def extract_sents_from_conll(inputfile):
    sents = []
    current_sent = []

    with open(inputfile, 'r') as my_conll:
        for line in my_conll:
            row = line.strip("\n").split('\t')
            if len(row) == 1:
                sents.append(current_sent)
                current_sent = []
            else:
                current_sent.append(tuple(row))
    return sents


def train_crf_model(X_train, y_train):

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    
    return crf

def create_crf_model(trainingfile):

    train_sents = extract_sents_from_conll(trainingfile)
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    crf = train_crf_model(X_train, y_train)
    
    return crf


def run_crf_model(crf, evaluationfile):

    test_sents = extract_sents_from_conll(evaluationfile)
    X_test = [sent2features(s) for s in test_sents]
    y_pred = crf.predict(X_test)
    
    return y_pred, X_test


def write_out_evaluation(eval_data, pred_labels, outputfile):

    outfile = open(outputfile, 'w')
    
    for evalsents, predsents in zip(eval_data, pred_labels):
        for data, pred in zip(evalsents, predsents):
            outfile.write(data.get('token') + "\t" + pred + "\n")

def train_and_run_crf_model(trainingfile, evaluationfile, outputfile):

    crf = create_crf_model(trainingfile)
    pred_labels, eval_data = run_crf_model(crf, evaluationfile)
    write_out_evaluation(eval_data, pred_labels, outputfile)

def main():

    args = sys.argv
    trainingfile = args[1] #Path to training ConLL file
    evaluationfile = args[2] #Path to dev/test ConLL file
    outputfile = args[3] #Path and name of output file
    
    train_and_run_crf_model(trainingfile, evaluationfile, outputfile)



if __name__ == '__main__':
    main()
