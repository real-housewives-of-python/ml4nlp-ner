import spacy
import sys
import pandas as pd

def analyze_informative_features(inputfile):
    
    '''
    Function that extracts potentially informative features from ConLL file for data exploration
    
    :param inputfile: path to ConLL file
    
    :type inputfile: string
    
    :return informative features: tsv file with informative features for each token in ConLL file
    '''

    
    conll_file = pd.read_csv(inputfile, delimiter='\t', header=None,  skipinitialspace = False)
    df = pd.DataFrame(conll_file) 
    token_list = df[0].tolist()
    gold_list = df[1].tolist()
    token_string = ' '.join(map(str, token_list))

    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 1093665
    doc = nlp(token_string)


    data = []

    for tok in doc:
            token = tok.text
            pos = tok.pos_
            lemma = tok.lemma_
            isalpha = tok.is_alpha
       

            feature_dict = {'Token': token, 'PoS': pos, 'Lemma': lemma, 'Is alpha': isalpha}
            data.append(feature_dict)

    
    df = pd.DataFrame(data=data)
    df['Gold'] = pd.Series(gold_list)
    df.to_csv('data/train_informative_features_analysis.tsv',sep='\t', index = False) #change name of output file for dev data
    
def main(argv=None):
    
    if argv is None:
        argv = sys.argv
                
    inputfile = argv[1] #Path to ConLL train or dev file
    
    analyze_informative_features(inputfile)

    
    
if __name__ == '__main__':
    main()