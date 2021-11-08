# -*- coding: utf-8 -*-
#This file includes pipeline to prepare the document for analysis
#functions used here can be found in prep_utilities.py file

#Pipeline for data prep

import numpy as np
import pandas as pd
from prep_utilities import * 

def prep_docs(doc, fix_contract = True, del_stopwords = True, lemmatize = True):
    copy_doc=doc.copy()
    
    #prepare clean tokens
    copy_doc['tokens']=prep_tokens(doc['quotation'], fix_contract, del_stopwords, lemmatize)
    
    #filter out unnecessary rows
    copy_doc=filter_quotes(copy_doc)
    
    #get domain names 
    copy_doc['websites']=copy_doc['urls'].apply(lambda x: get_website(x))
    copy_doc.drop(columns='urls', inplace=True)
    
    #replace None speakers
    copy_doc=replace_none_speaker(copy_doc)
    
    #get qid for replaced speakers
    #This is getting out of hand HELPPPPPPPPPPP
    for i in range(len(copy_doc)):
        if copy_doc['qids'][i]==[]:
            copy_doc['qids'][i]=find_qids(copy_doc['speaker'][i])
        else:
            copy_doc['qids'][i]=copy_doc['qids'][i]
    
    #copy_doc['qids']=copy_doc['speaker'].apply(lambda x: find_qids(x) if x==[] else x)

    #get the gender of the speaker
    copy_doc['gender']=copy_doc['qids'].apply(lambda x: find_gender(x))
    
    
    return copy_doc