import os
import re
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer,AutoTokenizer,AutoModelForSequenceClassification,AutoModelForQuestionAnswering
from transformers import BertForQuestionAnswering
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import csv
from sklearn import metrics
from transformers import pipelines


modelname = 'deepset/bert-base-cased-squad2'
modelname2 = 'deepset/electra-base-squad2'


model = BertForQuestionAnswering.from_pretrained(modelname)
tokenizer = AutoTokenizer.from_pretrained(modelname)
nlp = pipelines.pipeline('question-answering',model = model,tokenizer = tokenizer)
"""
model2 = AutoModelForQuestionAnswering.from_pretrained(modelname2)
tokenizer2 = AutoTokenizer.from_pretrained(modelname2)
nlp2 = pipelines.pipeline('question-answering',model = model2,tokenizer = tokenizer2)
"""
context = "The Intergovernmental Panel on Climate Change (IPCC) is a scientific intergovernmental body under the auspices of the United Nations, set up at the request of member governments. It was first established in 1988 by two United Nations organizations, the World Meteorological Organization (WMO) and the United Nations Environment Programme (UNEP), and later endorsed by the United Nations General Assembly through Resolution 43/53. Membership of the IPCC is open to all members of the WMO and UNEP. The IPCC produces reports that support the United Nations Framework Convention on Climate Change (UNFCCC), which is the main international treaty on climate change. The ultimate objective of the UNFCCC is to \"stabilize greenhouse gas concentrations in the atmosphere at a level that would prevent dangerous anthropogenic [i.e., human-induced] interference with the climate system\". IPCC reports cover \"the scientific, technical and socio-economic information relevant to understanding the scientific basis of risk of human-induced climate change, its potential impacts and options for adaptation and mitigation.\""

bert = nlp({
    'question': 'What organization is the IPCC a part of?',
    'context': context
})

"""
electra = nlp2({
    'question': 'What organization is the IPCC a part of?',
    'context': context
})
"""
print("bert: ",bert)
#print("electra: ",electra)




"""
df = pd.read_table('clean1.dat',header = None)
df = df.drop(df.columns[[1]], axis=1)
dfnew = df.set_axis(['ID', 'Answer', 'Grade'], axis=1, inplace=False)
print(dfnew.head())
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

tokens = tokenizer.encode(dfnew.Answer[4], return_tensors='pt')

result = model(tokens)
semantic = int(torch.argmax(result.logits))+1

print(tokens[0])
#print(tokenizer.decode(tokens[0]))
print(semantic)
"""