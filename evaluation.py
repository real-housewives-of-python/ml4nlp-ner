import pandas as pd
import sys
import csv



def extract_annotations(conll_file, delimiter='\t', engine='python'):
    '''
    Extract gold and system annotations and convert them to a list
    
    :param conll_file: ConLL file containing predicted and gold labels

    :type conll_file: string
    
    :returns annotations_gold: list of gold labels
    :returns annotations_system: list of predicted labels
    '''

    conllfile = pd.read_csv(conll_file, delimiter='\t', header=None) 
    df = pd.DataFrame(conllfile)
    Token = df[0].tolist()
    Gold = df[10].tolist()
    Predicted = df[11].tolist()

  
    dict = {'Token': Token, 'Gold': Gold, 'Predicted': Predicted} 
    
    df = pd.DataFrame(dict)
    
    annotations_gold = df['Gold'].tolist()
    annotations_system = df['Predicted'].tolist()
    
    return annotations_gold, annotations_system
    

def create_evaluations(annotations_gold, annotations_system):
    '''
    Compare the gold and predicted labels for each class and generate a confusion matrix and evaluation metrics
    
    :param annotations_gold: list of gold labels
    :param annotations_system: list of predicted labels

    :type annotations_gold: list 
    :type annotations_system: list 
    '''
    
    #1. B-ORG
    sys1 = 0
    sys2 = 0
    sys3 = 0
    sys4 = 0
    sys5 = 0
    sys6 = 0
    sys7 = 0
    sys8 = 0

    for index, ground_truth in enumerate(annotations_gold):
        if annotations_system[index] == ground_truth and ground_truth == 'B-ORG':
            sys1 += 1
        if annotations_system[index] == 'B-ORG' and ground_truth == 'I-ORG':
            sys2 += 1
        elif annotations_system[index] == 'B-ORG' and ground_truth == 'B-PER':
            sys3 += 1
        elif annotations_system[index] == 'B-ORG' and ground_truth == 'B-MISC':
            sys4 += 1
        elif annotations_system[index] == 'B-ORG' and ground_truth == 'I-PER':
            sys5 += 1
        elif annotations_system[index] == 'B-ORG' and ground_truth == 'O':
            sys6 += 1
        elif annotations_system[index] == 'B-ORG' and ground_truth == 'B-LOC':
            sys7 += 1
        elif annotations_system[index] == 'B-ORG' and ground_truth == 'I-LOC':
            sys8 += 1




    #2 I-ORG

    sys9 = 0
    sys10 = 0
    sys11 = 0
    sys12 = 0
    sys13 = 0
    sys14 = 0
    sys15 = 0
    sys16 = 0

    for index, ground_truth in enumerate(annotations_gold):
        if annotations_system[index] == 'I-ORG' and ground_truth == 'B-ORG':
            sys9 += 1
        elif annotations_system[index] == ground_truth and ground_truth == 'I-ORG':
            sys10 += 1
        elif annotations_system[index] == 'I-ORG' and ground_truth == 'B-PER':
            sys11 += 1
        elif annotations_system[index] == 'I-ORG' and ground_truth == 'B-MISC':
            sys12 += 1
        elif annotations_system[index] == 'I-ORG' and ground_truth == 'I-PER':
            sys13 += 1
        elif annotations_system[index] == 'I-ORG' and ground_truth == 'O':
            sys14 += 1
        elif annotations_system[index] == 'I-ORG' and ground_truth == 'B-LOC':
            sys15 += 1
        elif annotations_system[index] == 'I-ORG' and ground_truth == 'I-LOC':
            sys16 += 1

    #3 B-PER

    sys17 = 0
    sys18 = 0
    sys19 = 0
    sys20 = 0
    sys21 = 0
    sys22 = 0
    sys23 = 0
    sys24 = 0

    for index, ground_truth in enumerate(annotations_gold):
        if annotations_system[index] == 'B-PER' and ground_truth == 'B-ORG':
            sys17 += 1
        elif annotations_system[index] == 'B-PER' and ground_truth == 'I-ORG':
            sys18 += 1
        elif annotations_system[index] == ground_truth and ground_truth == 'B-PER':
            sys19 += 1
        elif annotations_system[index] == 'B-PER' and ground_truth == 'B-MISC':
            sys20 += 1
        elif annotations_system[index] == 'B-PER' and ground_truth == 'I-PER':
            sys21 += 1
        elif annotations_system[index] == 'B-PER' and ground_truth == 'O':
            sys22 += 1
        elif annotations_system[index] == 'B-PER' and ground_truth == 'B-LOC':
            sys23 += 1
        elif annotations_system[index] == 'B-PER' and ground_truth == 'I-LOC':
            sys24 += 1

    #4 B-MISC

    sys25 = 0
    sys26 = 0
    sys27 = 0
    sys28 = 0
    sys29 = 0
    sys30 = 0
    sys31 = 0
    sys32= 0

    for index, ground_truth in enumerate(annotations_gold):
        if annotations_system[index] == 'B-MISC' and ground_truth == 'B-ORG':
            sys25 += 1
        elif annotations_system[index] == 'B-MISC' and ground_truth == 'I-ORG':
            sys26 += 1
        elif annotations_system[index] == 'B-MISC' and ground_truth == 'B-PER':
            sys27 += 1
        elif annotations_system[index] == ground_truth and ground_truth == 'B-MISC':
            sys28 += 1
        elif annotations_system[index] == 'B-MISC' and ground_truth == 'I-PER':
            sys29 += 1
        elif annotations_system[index] == 'B-MISC' and ground_truth == 'O':
            sys30 += 1
        elif annotations_system[index] == 'B-MISC' and ground_truth == 'B-LOC':
            sys31 += 1
        elif annotations_system[index] == 'B-MISC' and ground_truth == 'I-LOC':
            sys32 += 1

    #5 I-PER

    sys33 = 0
    sys34 = 0
    sys35 = 0
    sys36 = 0
    sys37 = 0
    sys38 = 0
    sys39 = 0
    sys40 = 0
    

    for index, ground_truth in enumerate(annotations_gold):
        if annotations_system[index] == 'I-PER' and ground_truth == 'B-ORG':
            sys33 += 1
        elif annotations_system[index] == 'I-PER' and ground_truth == 'I-ORG':
            sys34 += 1
        elif annotations_system[index] == 'I-PER' and ground_truth == 'B-PER':
            sys35 += 1
        elif annotations_system[index] == 'I-PER' and ground_truth == 'B-MISC':
            sys36 += 1
        elif annotations_system[index] == ground_truth and ground_truth == 'I-PER':
            sys37 += 1
        elif annotations_system[index] == 'I-PER' and ground_truth == 'O':
            sys38 += 1
        elif annotations_system[index] == 'I-PER' and ground_truth == 'B-LOC':
            sys39 += 1
        elif annotations_system[index] == 'I-PER' and ground_truth == 'I-LOC':
            sys40 += 1

    #6 O

    sys41 = 0
    sys42 = 0
    sys43 = 0
    sys44 = 0
    sys45 = 0
    sys46 = 0
    sys47 = 0
    sys48 = 0

    for index, ground_truth in enumerate(annotations_gold):
        if annotations_system[index] == 'O' and ground_truth == 'B-ORG':
            sys41 += 1
        elif annotations_system[index] == 'O' and ground_truth == 'I-ORG':
            sys42 += 1
        elif annotations_system[index] == 'O' and ground_truth == 'B-PER':
            sys43 += 1
        elif annotations_system[index] == 'O' and ground_truth == 'B-MISC':
            sys44 += 1
        elif annotations_system[index] == 'O' and ground_truth == 'I-PER':
            sys45 += 1
        elif annotations_system[index] == ground_truth and ground_truth == 'O':
            sys46 += 1
        elif annotations_system[index] == 'O' and ground_truth == 'B-LOC':
            sys47 += 1
        elif annotations_system[index] == 'O' and ground_truth == 'I-LOC':
            sys48 += 1
            
    #7 B-LOC

    sys49 = 0
    sys50 = 0
    sys51 = 0
    sys52 = 0
    sys53 = 0
    sys54 = 0
    sys55 = 0
    sys56 = 0

    for index, ground_truth in enumerate(annotations_gold):
        if annotations_system[index] == 'B-LOC' and ground_truth == 'B-ORG':
            sys49 += 1
        elif annotations_system[index] == 'B-LOC' and ground_truth == 'I-ORG':
            sys50 += 1
        elif annotations_system[index] == 'B-LOC' and ground_truth == 'B-PER':
            sys51 += 1
        elif annotations_system[index] == 'B-LOC' and ground_truth == 'B-MISC':
            sys52 += 1
        elif annotations_system[index] == 'B-LOC' and ground_truth == 'I-PER':
            sys53 += 1
        elif annotations_system[index] == 'B-LOC' and ground_truth == 'O':
            sys54 += 1
        elif annotations_system[index] == ground_truth and ground_truth == 'B-LOC':
            sys55 += 1
        elif annotations_system[index] == 'B-LOC' and ground_truth == 'I-LOC':
            sys56 += 1
            
    #8 I-LOC

    sys57 = 0
    sys58 = 0
    sys59 = 0
    sys60 = 0
    sys61 = 0
    sys62 = 0
    sys63= 0
    sys64 = 0

    for index, ground_truth in enumerate(annotations_gold):
        if annotations_system[index] == 'I-LOC' and ground_truth == 'B-ORG':
            sys57 += 1
        elif annotations_system[index] == 'I-LOC' and ground_truth == 'I-ORG':
            sys58 += 1
        elif annotations_system[index] == 'I-LOC' and ground_truth == 'B-PER':
            sys59 += 1
        elif annotations_system[index] == 'I-LOC' and ground_truth == 'B-MISC':
            sys60 += 1
        elif annotations_system[index] == 'I-LOC' and ground_truth == 'I-PER':
            sys61 += 1
        elif annotations_system[index] == 'I-LOC' and ground_truth == 'O':
            sys62 += 1
        elif annotations_system[index] == 'I-LOC' and ground_truth == 'B-LOC':
            sys63 += 1
        elif annotations_system[index] == ground_truth and ground_truth == 'I-LOC':
            sys64 += 1
            
            

    data = {'B-ORG': [sys1, sys2, sys3, sys4, sys5, sys6, sys7, sys8],
    'I-ORG': [sys9, sys10, sys11, sys12, sys13, sys14, sys15, sys16],
    'B-PER': [sys17, sys18, sys19, sys20, sys21, sys22, sys23, sys24],
    'B-MISC': [sys25, sys26, sys27, sys28, sys29, sys30, sys31, sys32],
    'I-PER': [sys33, sys34, sys35, sys36, sys37, sys38, sys39, sys40],
    'O': [sys41, sys42, sys43, sys44, sys45, sys46, sys47, sys48],
    'B-LOC': [sys49, sys50, sys51, sys52, sys53, sys54, sys55, sys56],
    'I-LOC': [sys57, sys58, sys59, sys60, sys61, sys62, sys63, sys64]}
  
    df = pd.DataFrame(data, index=['B-ORG',
                               'I-ORG',
                               'B-PER',
                               'B-MISC',
                               'I-PER',
                                'O',
                                'B-LOC',
                                'I-LOC'])
                                                             

    df.loc['Total',:]= df.sum(axis=0)

    df.loc[:,'Total'] = df.sum(axis=1)
            
    print('---CONFUSION MATRIX---')
    print(df)
            
        
    #SYS B-ORG
    borg_pre = sys1 / df.iloc[8,0] if df.iloc[8, 0] != 0 else 0
    borg_rec = sys1 / df.iloc[0, 8]
    borg_f1 = 2 * ((borg_pre  * borg_rec) / (borg_pre  + borg_rec)) if borg_pre != 0 else 0


    #SYS I-ORG

    iorg_pre = sys10 / df.iloc[8,1]
    iorg_rec = sys8 / df.iloc[1, 8]
    iorg_f1 = 2 * ((iorg_pre  * iorg_rec) / (iorg_pre  + iorg_rec))

    #SYS B-PER

    bper_pre = sys19 / df.iloc[8,2] if df.iloc[8, 2] != 0 else 0
    bper_rec = sys15 / df.iloc[2, 8]
    bper_f1 = 2 * ((bper_pre  * bper_rec) / (bper_pre  + bper_rec)) if bper_pre != 0 else 0

    #SYS B-MISC

    bmisc_pre = sys28 / df.iloc[8,3] if df.iloc[8, 3] != 0 else 0
    bmisc_rec = sys28 / df.iloc[3, 8] if df.iloc[3, 8] != 0 else 0
    bmisc_f1 = 2 * ((bmisc_pre  * bmisc_rec) / (bmisc_pre  + bmisc_rec)) if bmisc_pre != 0 else 0

    #SYS I-PER

    iper_pre = sys37 / df.iloc[8,4]
    iper_rec = (sys37 / df.iloc[4, 8]) if df.iloc[4, 8] != 0 else 0
    iper_f1 = 2 * ((iper_pre  * iper_rec) / (iper_pre  + iper_rec)) if iper_pre != 0 else 0

    #SYS O

    o_pre = sys46 / df.iloc[8,5]
    o_rec = sys46 / df.iloc[5,8]
    o_f1 = 2 * ((o_pre  * o_rec) / (o_pre  + o_rec))
            
    #SYS B-LOC

    bloc_pre = sys55 / df.iloc[8,6]
    bloc_rec = sys55 / df.iloc[6, 8]
    bloc_f1 = 2 * ((o_pre  * o_rec) / (o_pre  + o_rec))
            
    #SYS I-LOC

    iloc_pre = sys64 / df.iloc[8,7]
    iloc_rec = sys64 / df.iloc[7,8]
    iloc_f1 = 2 * ((o_pre  * o_rec) / (o_pre  + o_rec))
    
    #CALCULATE MACRO AVERAGE
    
    list_precision = [borg_pre + iorg_pre + bper_pre + bmisc_pre + iper_pre + o_pre + bloc_pre + iloc_pre]
    list_recall = [borg_rec + iorg_rec + bper_rec + bmisc_rec + iper_rec + o_rec + bloc_rec + iloc_rec]
    list_f1 = [borg_f1 + iorg_f1 + bper_f1 + bmisc_f1 + iper_f1 + o_f1 + bloc_f1 + iloc_f1]
        
    macro_precision = sum(list_precision) / 8
    macro_recall =  sum(list_recall) / 8
    macro_f1 =  sum(list_f1) / 8
    
    print('---MACRO AVERAGE---')
    
    print('Macro Precision:', macro_precision, 'Macro Recall:', macro_recall, 'Macro F1:', macro_f1)

    evaluations = {}
    evaluations['B-ORG'] = {}
    evaluations['B-ORG']['Precision'] = borg_pre
    evaluations['B-ORG']['Recall'] = borg_rec
    evaluations['B-ORG']['F1'] = borg_f1
    evaluations['I-ORG'] = {}
    evaluations['I-ORG']['Precision'] = iorg_pre
    evaluations['I-ORG']['Recall'] = iorg_rec
    evaluations['I-ORG']['F1'] = iorg_f1
    evaluations['B-PER'] = {}
    evaluations['B-PER']['Precision'] = bper_pre 
    evaluations['B-PER']['Recall'] = bper_rec
    evaluations['B-PER']['F1'] = bper_f1
    evaluations['B-MISC'] = {}
    evaluations['B-MISC']['Precision'] = bmisc_pre
    evaluations['B-MISC']['Recall'] = bmisc_rec
    evaluations['B-MISC']['F1'] = bmisc_f1
    evaluations['I-PER'] = {}
    evaluations['I-PER']['Precision'] = iper_pre
    evaluations['I-PER']['Recall'] = iper_rec
    evaluations['I-PER']['F1'] = iper_f1
    evaluations['O'] = {}
    evaluations['O']['Precision'] = o_pre
    evaluations['O']['Recall'] = o_rec
    evaluations['O']['F1'] = o_f1
    evaluations['B-LOC'] = {}
    evaluations['B-LOC']['Precision'] = bloc_pre
    evaluations['B-LOC']['Recall'] = bloc_rec
    evaluations['B-LOC']['F1'] = bloc_f1
    evaluations['I-LOC'] = {}
    evaluations['I-LOC']['Precision'] = iloc_pre
    evaluations['I-LOC']['Recall'] = iloc_rec
    evaluations['I-LOC']['F1'] = iloc_f1
            
    return evaluations
            

def provide_output_tables(evaluations):
    '''
    Create tables based on the evaluation of various systems
    
    :param evaluations: the outcome of evaluating one or more systems
    
    :type evaluations: nested dictionary
    '''
    #https:stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
    evaluations_pddf = pd.DataFrame.from_dict({(i,j): evaluations[i][j]
                                              for i in evaluations.keys()
                                              for j in evaluations[i].keys()},
                                             orient='index')
    
    print('---EVALUATION METRICS---')
    print(evaluations_pddf)
    


    
    
    
def main(argv=None):
    
    if argv is None:
        argv = sys.argv
        

    
    conll_file = argv[1] #Path to ConLL file containing predicted and gold labels
    
    annotations_gold, annotations_system = extract_annotations(conll_file)
    evaluations = create_evaluations(annotations_gold, annotations_system)
    provide_output_tables(evaluations)

    
if __name__ == '__main__':
    main()