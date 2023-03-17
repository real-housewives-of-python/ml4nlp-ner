from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.feature_extraction import DictVectorizer
from gensim.models import KeyedVectors
from sklearn.metrics import make_scorer
import pandas as pd
import csv
import gensim
import sys

#Inspired by ner_machine_learning.py 

word_embedding_path = r'C:\Users\alyss\Desktop\CODE\ML4NLP\ma-ml4nlp-labs-main\models\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin' #PLEASE CHANGE PATH TO WORD EMBEDDING MODEL


def extract_features_and_labels(trainingfile):
    
    '''
    Function that extracts features and gold labels 
    
    :param trainingfile: path to TSV file
    
    :type trainingfile: string
    
    :return train_features: list containing dictionary mapping tokens to features
    :return train_targets: list of gold labels for each token
    '''
  
    
    train_features = []
    train_targets = []
    
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                next_t = components[1]
                capital = components[2]
                prev_cap = components[3]
                next_cap = components[4]
                start_sent = components[5]
                p_noun = components[6]
                cap_pnoun = components[7]
                pnoun_startsent = components[8]
                allcaps_pnoun = components[9]
                feature_dict = {'token':token, 'next': next_t, 'capital': capital, 'prev_cap': prev_cap, 'next_cap': next_cap, 'start_sent': start_sent,
                               'p_noun': p_noun, 'cap_pnoun': cap_pnoun, 'pnoun_startsent': pnoun_startsent,
                                'allcaps_pnoun': allcaps_pnoun}
                train_features.append(feature_dict)
                #gold is in the last column
                train_targets.append(components[-1])
    return train_features, train_targets

def extract_features(inputfile):
    
    '''
    Function that extracts features  
    
    :param inputfile: path to TSV file
    
    :type inputfile: string
    
    :return inputdata: list containing dictionary mapping tokens to features
    '''
   
    inputdata = []
    with open(inputfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                next_t = components[1]
                capital = components[2]
                prev_cap = components[3]
                next_cap = components[4]
                start_sent = components[5]
                p_noun = components[6]
                cap_pnoun = components[7]
                pnoun_startsent = components[8]
                allcaps_pnoun = components[9]
                feature_dict = {'token':token, 'next': next_t,  'capital': capital, 'prev_cap': prev_cap, 'next_cap': next_cap, 'start_sent': start_sent,
                               'p_noun': p_noun, 'cap_pnoun': cap_pnoun, 'pnoun_startsent': pnoun_startsent,
                                'allcaps_pnoun': allcaps_pnoun}
                inputdata.append(feature_dict)
    return inputdata

def create_classifier(train_features, train_targets, modelname, grid=False): 
    
    '''
    Function to select and train various classifiers  
    
    :param train_features: list containing dictionary mapping tokens to features
    :param train_targets: list of gold labels for each token
    :param modelname: name of classifier
    :param grid: Boolean value to indicate whether to do to grid search or not
  
    :type train_features: list
    :type train_targets: list
    :type modelname: string
    
    :return model: the trained classifier
    :return vec: the dict vectorizer used to transform the train_features
    
    '''
 
        
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(train_features)
        
    if modelname ==  'logreg':
        model = LogisticRegression(solver='lbfgs', max_iter=1000) 
        
    elif modelname == 'cnb':
        model = ComplementNB()
        
    elif modelname == 'svm' and grid==False: 
        model = SVC(max_iter=1000)
        
    elif modelname == 'svm' and grid==True:   
        model = SVC(max_iter=1000)
        grid = HalvingGridSearchCV(model, {'C': [0.1, 1, 10, 100, 1000],
        'kernel': ['rbf'],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001]})
        grid.fit(features_vectorized, train_targets)
        grid.best_params_  #Prints out the best parameters and the best score from the grid search
        grid.best_score_
                      
                
    model.fit(features_vectorized, train_targets)  
    return model, vec
        
            

def classify_data(model, vec, inputdata, outputfile):
    
    '''
    Function that performs classification of samples and outputs file mapping predicted labels to gold labels
    
    :param model: trained model
    :param vec: trained DictVectorizer
    :param inputdata: input file to be classified
    :param outputfile: new file containing gold and predicted labels
  
    :type inputdata: string
    :type outputfile: string
    
    :return ouputfile: ConLL file mapping predicted labels to gold labels
    
    '''
  
    features = extract_features(inputdata)
    features = vec.transform(features)
    predictions = model.predict(features)
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:        
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1                
    outfile.close()
    
def extract_embeddings_as_features_and_gold(trainingfile,word_embedding_model):
    '''
    Function that extracts features and gold labels using word embeddings
    
    :param trainingfile: path to ConLL file
    :param word_embedding_model: a pretrained word embedding model
    
    :type trainingfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :return we_features: list of vector representation of tokens
    :return we_labels: list of gold labels
    '''
    ### This code was partially inspired by code included in the HLT course, obtained from https://github.com/cltl/ma-hlt-labs/, accessed in May 2020.
    
    we_labels = []
    we_features = []
    
    conllinput = open(trainingfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t')
    for row in csvreader:
        #check for cases where empty lines mark sentence boundaries (which some conll files do).
        if len(row) > 3:
            if row[0] in word_embedding_model:
                vector = word_embedding_model[row[0]]
            else:
                vector = [0]*300
            we_features.append(vector)
            we_labels.append(row[-1])
    return we_features, we_labels

def extract_embeddings_as_features(inputfile,word_embedding_model):
    
    '''
    Function that extracts features using word embeddings
    
    :param inputfile: path to ConLL file
    :param word_embedding_model: a pretrained word embedding model
    
    :type inputfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :return we_features2: list of vector representation of tokens
    '''
    
    we_features2 = []
    
    conllinput = open(inputfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t')
    for row in csvreader:
        #check for cases where empty lines mark sentence boundaries (which some conll files do).
        if len(row) > 3:
            if row[0] in word_embedding_model:
                vector = word_embedding_model[row[0]]
            else:
                vector = [0]*300
            we_features2.append(vector)
    return we_features2

def create_we_classifier(we_features, we_labels, modelname):
    
    '''
    Function to train Logistic Regression classifier  
    
    :param we_features: list of vector representation of tokens
    :param we_targets: list of gold labels for each token
    :param modelname: name of classifier (Logistic Regression)
  
    :type we_features: list
    :type we_targetss: list
    :type modelname: string
    
    :return model: the trained classifier    
    '''
       
    if modelname ==  'logreg':
        we_model = LogisticRegression(solver='lbfgs', max_iter=1000)

    we_model.fit(we_features, we_labels)
    
    return we_model

def classify_we_data(we_model, inputdata, word_embedding_model, outputfile):
    
    '''
    Function that performs classification of samples and outputs file mapping predictions to gold labels
    
    :param we_model: trained model
    :param inputdata: input file to be classified
    :param word_embedding_model: a pretrained word embedding model
    :param outputfile: new file containing gold and predicted labels
  
    :type we_model: object
    :type inputdata: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    :type outputfile: string
    
    :return ouputfile: ConLL file mapping predicted labels to gold labels
    
    '''
  
    features = extract_embeddings_as_features(inputdata, word_embedding_model)
    predictions = we_model.predict(features)
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()
    
def main(argv=None):
    
    if argv is None:
        argv = sys.argv
                
    trainingfile = argv[1] #Training file
    inputfile = argv[2] #Development or Test file
    modelname = argv[3] #'logreg' or 'cnb' or 'svm'
    outputfile = argv[4] #e.g. 'data/logreg_dev' or 'data/logreg_test'
    embeddings = [5] #True or False
    
    if embeddings == True: #N.B. We only use this with the Logistic Regression model
        word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format(word_embedding_path, binary=True)  
        we_features, we_labels = extract_embeddings_as_features_and_gold(trainingfile, word_embedding_model)
        we_ml_model = create_we_classifier(we_features, we_labels, modelname)
        classify_we_data(we_ml_model, inputfile, word_embedding_model, outputfile)
    
    else:
        training_features, gold_labels = extract_features_and_labels(trainingfile)
        ml_model, vec = create_classifier(training_features, gold_labels, modelname)
        classify_data(ml_model, vec, inputfile, outputfile)
    
    
if __name__ == '__main__':
    main()
    