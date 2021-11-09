# -*- coding: utf-8 -*-
#This file includes pipeline to prepare the document for analysis
#functions used here can be found in prep_utilities.py file

#Pipeline for data prep

#import numpy as np
#import pandas as pd
from prep_utilities import *

#Pipeline for data prep
#Pipeline for data prep
def prep_docs(doc, doc2, fix_contract = True, lemmatize = True):
    print("Copying...")
    copy_doc=doc.copy()

    #get date in YYYY-MM format
    print("Simplifying date column...")
    copy_doc['date']= copy_doc['date'].apply(lambda x: get_yyyy_mm(x))

    #prepare clean tokens
    print("Tokenizing quotes...")
    copy_doc['tokens']=prep_tokens(doc['quotation'], fix_contract, lemmatize)

    #filter out unnecessary rows
    print("Filtering rows...")
    copy_doc=filter_quotes(copy_doc)

    #get domain names
    print("Getting url domains...")
    copy_doc['websites']=copy_doc['urls'].apply(lambda x: get_website(x))
    copy_doc.drop(columns='urls', inplace=True)

    #replace None speakers
    copy_doc=replace_none_speaker(copy_doc)

    #get qid for replaced speakers
    print("Getting QIDs for replaced speakers...")
    missing_qids=copy_doc[copy_doc['qids'].apply(lambda x: x==[])]
    copy_doc=copy_doc[copy_doc['qids'].apply(lambda x: x!=[])]
    missing_qids['qids']=missing_qids['speaker'].apply(lambda x: find_qids(x, doc2))
    copy_doc=copy_doc.append(missing_qids, ignore_index=True)

    #If the qid is still missing we drop that row
    copy_doc=copy_doc[copy_doc['qids'].apply(lambda x: x!=[])]

    #get the gender of the speaker
    print("Getting genders...")
    copy_doc['gender']=copy_doc['qids'].apply(lambda x: find_gender(x,doc2))

    return copy_doc
