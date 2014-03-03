__author__ = 'bryan'

import cPickle

with open('chiralityData.pkl', 'rb') as c_data_file:
    chiralityData = cPickle.load(c_data_file)

with open('growthData.pkl', 'rb') as g_data_file:
    growthData = cPickle.load(g_data_file)

# Now test korelov's model

import chirality_data_analysis.korelov_model as kor

kor.chiralityData = chiralityData
kor.growthData = growthData

model0 =  kor.chirality_pymc_model('chlor', 0.0)