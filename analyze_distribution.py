import sys
import pandas as pd
from collections import Counter

def analyze_distribution(inputfile, delimiter = '\t', header=None):
    
    '''
    Function that outputs label distribution of ConLL data
    
    :param inputfile: path to ConLL file
    :param delimiter: tab delimiter
    :param header=None: skip header
    
    :type inputfile: string
    
    :return distribution: tsv file with distribution for each class
    '''
    
    
    conll_file = pd.read_csv(inputfile, delimiter='\t', header=None) #Partially inspired by basic_evaluation.ipynb 
    df = pd.DataFrame(conll_file)
    annotations = df[3].tolist()

    counts = Counter(annotations)

    BORG = counts["B-ORG"]
    IORG = counts["I-ORG"]
    BPER = counts["B-PER"]
    IPER = counts["I-PER"]
    BMISC = counts["B-MISC"]
    BLOC = counts["B-LOC"]
    ILOC = counts["I-LOC"]
    O = counts["O"]


    #The following is taken from https://www.geeksforgeeks.org/create-a-pandas-dataframe-from-lists/
    labels = ["B-ORG", "I-ORG", "B-PER", "I-PER", "B-MISC", "B-LOC", "I-LOC", "O"]
  
    total = [BORG, IORG, BPER, IPER, BMISC, BLOC, ILOC, O]
  
    df2 = pd.DataFrame(list(zip(labels, total)),
               columns =["Label", "Total"])
    df2.to_csv("data/train_distribution_analysis.tsv",sep="\t", index = False) #Please change output file name for dev data
                                                                          
    
def main(argv=None):
    
    if argv is None:
        argv = sys.argv
                
    inputfile = argv[1] #Path to training/development ConLL file
    
    analyze_distribution(inputfile)
     
    
if __name__ == '__main__':
    main()