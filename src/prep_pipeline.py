# -*- coding: utf-8 -*-
#This file includes pipeline to prepare the document for analysis
#functions used here can be found in prep_utilities.py file

#Pipeline for data prep

import numpy as np
import pandas as pd
from prep_utilities import * 

#Pipeline for data prep
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
    missing_qids=copy_doc[copy_doc['qids'].apply(lambda x: x==[])]
    copy_doc=copy_doc[copy_doc['qids'].apply(lambda x: x!=[])]
    missing_qids['qids']=missing_qids['speaker'].apply(lambda x: find_qids(x, speaker_attributes))
    copy_doc=copy_doc.append(missing_qids, ignore_index=True)

    #get the gender of the speaker
    copy_doc['gender']=copy_doc['qids'].apply(lambda x: find_gender(x,speaker_attributes))
    
    
    return copy_doc