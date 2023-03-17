import sys
import sklearn
import csv
import gensim
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

#Inspired by sample_code_features_ablation_analysis.ipynb

feature_to_index = {'Token': 0, 'Next': 1, 'Capital': 2, 'Prev_cap': 3, 'Next_cap': 4, 'Start_sent': 5, 'P_Noun': 6, 'Cap_P_Noun': 7, 'P_Noun_Start_Sent': 8, 'All_Caps_P_Noun': 9}

all_features = ['Token', 'Next', 'Capital', 'Prev_cap', 'Next_cap', 'Start_sent', 'P_Noun', 'Cap_P_Noun', 'P_Noun_Start_Sent', 'All_Caps_P_Noun']

word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format(r'C:\Users\alyss\Desktop\CODE\ML4NLP\ma-ml4nlp-labs-main\models\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin', binary=True) #PLEASE ADD OWN PATH TO WE MODEL


def extract_features_and_gold_labels(trainfile, selected_features):
    
    '''
    Function that extracts features and gold labels 
    
    :param trainfile: path to TSV file
    :param selected_features: list of lists containing feature combinations
    
    :type trainfile: string
    :type selected_features: list
    
    :return features: list of of tokens
    :return labels: list of gold labels
    '''
  
    
    features = []
    labels = []
    train_input = open(trainfile, 'r')
    csvreader = csv.reader(train_input, delimiter='\t', quotechar='|')
    for row in csvreader:

        if len(row) > 0:
            feature_value = {}
            for feature_name in selected_features:
                row_index = feature_to_index.get(feature_name)
                feature_value[feature_name] = row[row_index]
            features.append(feature_value)
            labels.append(row[-1])
    return features, labels

def extract_word_embedding(token, word_embedding_model):
    '''
    Function that returns the word embedding for a given token out of a distributional semantic model and a 300-dimension vector of 0s otherwise
    
    :param token: the token
    :param word_embedding_model: the distributional semantic model
    
    :type token: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :returns a vector representation of the token
    '''
    if token in word_embedding_model:
        vector = word_embedding_model[token]
    else:
        vector = [0]*300
    return vector


def extract_feature_values(row, selected_features):
    '''
    Function that extracts feature value pairs from row
    
    :param row: row from conll file
    :param selected_features: list of selected features
    
    :type row: string
    :type selected_features: list of strings
    
    :returns: dictionary of feature value pairs
    '''
    feature_values = {}
    for feature_name in selected_features:
        r_index = feature_to_index.get(feature_name)
        feature_values[feature_name] = row[r_index]
        
    return feature_values

def create_classifier(features, labels):
    '''
    Function that creates classifier from features represented as vectors and gold labels for combining WE and one-hot encodings
    
    :param features: list of vector representations of tokens
    :param labels: list of gold labels
    :type features: list of vectors
    :type labels: list of strings
    
    :returns trained Logistic Regression classifier
    '''
    
    
    lr_classifier = LogisticRegression(solver='saga')
    lr_classifier.fit(features, labels)
    
    return lr_classifier


def create_vectorizer_and_classifier(features, labels, modelname):
    '''
    Function that takes feature-value pairs and gold labels as input and trains a classifier for one-hot encoded features
    
    :param features: feature-value pairs
    :param labels: gold labels
    
    :type features: a list of dictionaries
    :type labels: a list of strings
    
    :return model: a trained classifier
    :return vec: a DictVectorizer to which the feature values are fitted. 
    '''
    
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(features)
    
    if modelname == 'cnb':
        model = ComplementNB()
        
    elif modelname == 'logreg':
        model = LogisticRegression(solver='saga', max_iter=4000)
        
    model.fit(features_vectorized, labels)   
    return model, vec


def get_predicted_and_gold_labels(testfile, vec, model, selected_feature):
    '''
    Function that extracts features and runs classifier on a test file returning predicted and gold labels
    
    :param testfile: path to the TSV development data file
    :param vectorizer: vectorizer in which the mapping between feature values and dimensions is stored
    :param classifier: the trained classifier
    
    :type testfile: string
    :type vectorizer: DictVectorizer
    :type classifier: LogisticRegression()
    
    
    :return predictions: list of output labels provided by the classifier on the test file
    :return goldlabels: list of gold labels as included in the test file
    '''
    
    sparse_feature_reps, goldlabels = extract_features_and_gold_labels(testfile, selected_feature)
    test_features_vectorized = vec.transform(sparse_feature_reps)
    predictions = model.predict(test_features_vectorized)
    
    return predictions, goldlabels

def create_vectorizer_traditional_features(feature_values):
    '''
    Function that creates vectorizer for set of feature values
    
    :param feature_values: list of dictionaries containing feature-value pairs
    :type feature_values: list of dictionairies (key and values are strings)
    
    :returns: vectorizer with feature values fitted
    '''
    vectorizer = DictVectorizer()
    vectorizer.fit(feature_values)
    
    return vectorizer

        
    
def combine_sparse_and_dense_features(dense_vectors, sparse_features):
    '''
    Function that takes sparse and dense feature representations and appends their vector representation
    
    :param dense_vectors: list of dense vector representations
    :param sparse_features: list of sparse vector representations
    :type dense_vector: list of arrays
    :type sparse_features: list of lists
    
    :returns: list of arrays in which sparse and dense vectors are concatenated
    '''
    
    combined_vectors = []
    sparse_vectors = np.array(sparse_features.toarray())
    
    for index, vector in enumerate(sparse_vectors):
        combined_vector = np.concatenate((vector,dense_vectors[index]))
        combined_vectors.append(combined_vector)
    return combined_vectors
    

def extract_traditional_features_and_embeddings_plus_gold_labels(conllfile, word_embedding_model, vectorizer=None):
    '''
    Function that extracts traditional features as well as embeddings and gold labels using word embeddings for current and next token
    
    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    labels = []
    dense_vectors = []
    traditional_features = []
    
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    for row in csvreader:
        if len(row) > 0:
            token_vector = extract_word_embedding(row[0], word_embedding_model)
            next_vector = extract_word_embedding(row[1], word_embedding_model)
            dense_vectors.append(np.concatenate((token_vector,next_vector)))
            other_features = extract_feature_values(row, ['Capital', 'Prev_cap', 'Next_cap', 'Start_sent',
                             'P_Noun', 'Cap_P_Noun', 'P_Noun_Start_Sent', 'All_Caps_P_Noun'])
            traditional_features.append(other_features)
            #adding gold label to labels
            labels.append(row[-1])
            
    #create vector representation of traditional features
    if vectorizer is None:
        #creates vectorizer that provides mapping (only if not created earlier)
        vectorizer = create_vectorizer_traditional_features(traditional_features)
    sparse_features = vectorizer.transform(traditional_features)
    combined_vectors = combine_sparse_and_dense_features(dense_vectors, sparse_features)
    
    return combined_vectors, vectorizer, labels


def label_data_with_combined_features(testfile, lr_classifier, vectorizer, word_embedding_model):
    '''
    Function that labels data with model using both sparse and dense features
    
    :param testfile: path to the TSV file containing one-hot encoded features of development data
    :param lr_classifier: the trained Logistic Regression classifier
    :param vectorizer: vectorizer in which the mapping between feature values and dimensions is stored
    :param word_embedding_model: a pretrained word embedding model
    
    
    :type testfile: string
    :type vectorizer: DictVectorizer
    :type classifier: LogisticRegression()
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    '''

    feature_vectors, vectorizer, goldlabels = extract_traditional_features_and_embeddings_plus_gold_labels(testfile, word_embedding_model, vectorizer)
    predictions = lr_classifier.predict(feature_vectors)
    
    return predictions, goldlabels

def print_confusion_matrix(predictions, goldlabels):
    '''
    Function that prints out a confusion matrix
    
    :param predictions: predicted labels
    :param goldlabels: gold standard labels
    :type predictions, goldlabels: list of strings
    '''   
    
    #based on example from https://datatofish.com/confusion-matrix-python/ 
    data = {'Gold':    goldlabels, 'Predicted': predictions    }
    df = pd.DataFrame(data, columns=['Gold','Predicted'])

    confusion_matrix = pd.crosstab(df['Gold'], df['Predicted'], rownames=['Gold'], colnames=['Predicted'])
    print (confusion_matrix)


def print_precision_recall_fscore(predictions, goldlabels):
    '''
    Function that prints out precision, recall and f-score
    
    :param predictions: predicted output by classifier
    :param goldlabels: original gold labels
    :type predictions, goldlabels: list of strings
    '''
    
    precision = metrics.precision_score(y_true=goldlabels,
                        y_pred=predictions,
                        average='macro')

    recall = metrics.recall_score(y_true=goldlabels,
                     y_pred=predictions,
                     average='macro')


    fscore = metrics.f1_score(y_true=goldlabels,
                 y_pred=predictions,
                 average='macro')

    print('P:', precision, 'R:', recall, 'F1:', fscore)
    

# 

def main(argv=None):   
    
    if argv is None:
        argv = sys.argv
        
    trainfile = argv[1] #Path to TSV training file
    testfile = argv[2] #Path to TSV development file
    modelname = argv[3] #'logreg' or 'cnb'
    embeddings = argv[4] #True or False
            

    selected_features =  [['Token'], #Baseline                
    ['Token', 'Next', 'Capital', 'Prev_cap', 'Next_cap', 'Start_sent', 'P_Noun', 'Cap_P_Noun', 'P_Noun_Start_Sent', 'All_Caps_P_Noun'], #All at once
    ['Token', 'Next'], #One by one                
    ['Token', 'Next', 'Capital'], 
    ['Token', 'Next', 'Capital', 'Prev_cap'],
    ['Token', 'Next', 'Capital', 'Prev_cap', 'Next_cap'],
    ['Token', 'Next', 'Capital', 'Prev_cap', 'Next_cap', 'Start_sent'],
    ['Token', 'Next', 'Capital', 'Prev_cap', 'Next_cap', 'Start_sent', 'P_Noun'], 
    ['Token', 'Next', 'Capital', 'Prev_cap', 'Next_cap', 'Start_sent', 'P_Noun', 'Cap_P_Noun'],
    ['Token', 'Next', 'Capital', 'Prev_cap', 'Next_cap', 'Start_sent', 'P_Noun', 'Cap_P_Noun', 'P_Noun_Start_Sent'],
    ['Token', 'Next', 'Capital', 'Prev_cap', 'Next_cap', 'Start_sent', 'P_Noun', 'Cap_P_Noun', 'P_Noun_Start_Sent', 'All_Caps_P_Noun'],
    #All simple features
    ['Token', 'Next','Capital', 'Prev_cap', 'Next_cap', 'Start_sent', 'P_Noun'],
    #All complex features
    ['Cap_P_Noun', 'P_Noun_Start_Sent',  'All_Caps_P_Noun'],
    #Mix traditional and complex features
    ['Token', 'Capital', 'Start_sent', 'P_Noun', 'Cap_P_Noun', 'P_Noun_Start_Sent',  'All_Caps_P_Noun'],
    #Token and Next, separately and together                    
    ['Token', 'Cap_P_Noun', 'P_Noun_Start_Sent', 'All_Caps_P_Noun'],
    ['Next', 'Cap_P_Noun', 'P_Noun_Start_Sent', 'All_Caps_P_Noun'],
    ['Token', 'Next', 'Cap_P_Noun', 'P_Noun_Start_Sent',  'All_Caps_P_Noun']]   


    for selected_feature in selected_features:
        print('Feature(s) selected', 'for', modelname, selected_feature)

        feature_values, labels = extract_features_and_gold_labels(trainfile, selected_features=selected_feature)
        model, vectorizer = create_vectorizer_and_classifier(feature_values, labels, modelname)
        predictions, goldlabels = get_predicted_and_gold_labels(testfile, vectorizer, model, selected_feature)
        print_confusion_matrix(predictions, goldlabels)
        print_precision_recall_fscore(predictions, goldlabels)
        
    if embeddings:
        print('Loading WE model...')
        feature_vectors, vectorizer, gold_labels = extract_traditional_features_and_embeddings_plus_gold_labels(trainfile, word_embedding_model)
        print('Creating classifier...')
        model = create_classifier(feature_vectors, gold_labels)
        print('Running the evaluation...')
        predictions, goldlabels = label_data_with_combined_features(testfile, model, vectorizer, word_embedding_model)
        print_confusion_matrix(predictions, goldlabels)
        print_precision_recall_fscore(predictions, goldlabels)

    
if __name__ == '__main__':
    main()
    
