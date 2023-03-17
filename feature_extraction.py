import csv
import sys
import pandas as pd


def feature_extraction(inputfile):
    
    '''
    Function that extracts one-hot encoded features from ConLL data which will be used to train the systems
    
    :param inputfile: path to ConLL file
    
    :type inputfile: string
    
    :return extracted features: tsv file with one-hot encoded features for each token in ConLL data
    '''
    
    infile = pd.read_csv(inputfile, delimiter='\t', header=None)  
    df = pd.DataFrame(infile)
    Token = df[0].tolist()
    POS = df[1].tolist()
    Gold = df[3].tolist()
            
    tokenlst = []
    nextlst = []
    capitallst = []
    prev_capitallst = []
    next_capitallst = []
    start_sentlst = []
    p_nounlst = []
    mix1lst = []
    mix2lst = []
    mix3lst = []
    goldlst = []
    
    for g in Gold:
        goldlst.append(g)

    for i, t in enumerate(Token):
        
           
            #PREVIOUS TOKEN:          
            prev_token = Token[i -1]
                          

            #NEXT TOKEN:
            try:
                next_token = Token[i + 1]
            except IndexError:
                break
            else:
                nextlst.append(next_token)
                
             #TOKEN
            tokenlst.append(t)
                
            
            #CAPITALIZATION
            if str(t)[0].isupper() == True:
                capitallst.append('1')
            else:
                capitallst.append('0')

            #CAPITALIZATION OF PREVIOUS TOKEN
            if str(prev_token)[0].isupper() == True:
                prev_capitallst.append('1')
            else:
                prev_capitallst.append('0')

            #CAPITALIZATION OF NEXT TOKEN
            if str(next_token)[0].isupper() == True:
                next_capitallst.append('1')
            else:
                next_capitallst.append('0')

            #IS START OF SENTENCE
            if prev_token == '.':
                start_sentlst.append('1')
            else:
                start_sentlst.append('0')

            #IS PROPER NOUN
            if POS[i] == 'NNP':
                p_nounlst.append('1')
            else:
                p_nounlst.append('0')
                
            #IS CAPITALIZED AND PROPER NOUN
            if str(t)[0].isupper() == True and POS[i] == 'NNP':
                mix1lst.append('1')
            else:
                mix1lst.append('0')
                
            #IS PROPER NOUN AND START OF SENTENCE
            if POS[i] == 'NNP' == True and prev_token == '.':
                mix2lst.append('1')
            elif t is Token[0]:
                mix2lst.append('1')
            else:
                mix2lst.append('0')
                
            #IS ALL CAPS AND PROPER NOUN:
            if str(t).isupper() == True and POS[i] == 'NNP':
                mix3lst.append('1')
            else:
                mix3lst.append('0')
            
                
 
            
                

            dict = {'Token': tokenlst, 'Next_token': nextlst,'Capital': capitallst, 'Prev_cap': prev_capitallst, 'Next_cap': next_capitallst,
                        'Start_sent': start_sentlst, 'P_noun': p_nounlst, 'Cap_P_Noun': mix1lst, 'P_Noun_Start_sent': mix2lst,
                     'AllCaps_Pnoun': mix3lst} 

    df = pd.DataFrame(dict)
    df['Gold'] = pd.Series(goldlst)
    df.to_csv('data/train_features.tsv',sep='\t', index = False) #Please change output file name for dev and test data
    

def main(argv=None):
    
    if argv is None:
        argv = sys.argv
                
    inputfile = argv[1] #Path to ConLL train, dev or test file
    
    feature_extraction(inputfile)
    
    
    
    
if __name__ == '__main__':
    main()