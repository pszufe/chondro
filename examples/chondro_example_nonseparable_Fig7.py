# -*- coding: utf-8 -*-
'''
"chondro_example_nonseparable_Fig7.py"

Example Chondro usage for sensitivity analysis of non-separable decision trees

Developed by:
  Przemyslaw Szufel
  Michal Jakubczyk
  Bogumil Kaminski

Copyright 2015 Przemyslaw Szufel & Michal Jakubczyk & Bogumil Kaminski
  {pszufe, mjakubc, bkamins}@sgh.waw.pl

This software is licensed under the terms of
open source GNU Affero General Public License, version 3
'''


#import the module
from chondro import *



#Open the tree at figure 7
file_name = "exaple_non_separable_fig7.json"
#load from a file
tree = load_tree(file_name)

def tree_derived_probs_lambda(probs): 
    p=dict()
    sensitivity = Fraction("90/100")
    specifity = Fraction("70/100")
    p["gas"] = probs[0][0]
    p["no_gas"] = 1-p["gas"]
    p["pos._test"]=sensitivity*p["gas"]+(1-specifity)*p["no_gas"]
    p["neg._test"]=1-p["pos._test"]
    p["gas|pos._test"]=sensitivity*p["gas"]/p["pos._test"]
    p["no_gas|pos._test"]=1-p["gas|pos._test"]
    p["gas|neg._test"]=(1-sensitivity)*p["gas"]/p["neg._test"]
    p["no_gas|neg._test"]=1-p["gas|neg._test"]
    return p




#fundamental probabilities
fund_probs = [[Fraction("7/10")]]
#solve the decision tree by passing probability dictionary
ev,dec=solve_tree(tree,tree_derived_probs_lambda(fund_probs))
print ("Expected value: "+str(ev)+" reachable decisions: "+str(get_reachable(dec)))
#print a solved tree
print_tree(tree)

stabi = find_stability(tree,tree_derived_probs_lambda,fund_probs,precision=Fraction("1/1000"),max_epsilon=1,use_labels=False )
print ("stability", stabi)

ress = find_perturbation_mode(tree,tree_derived_probs_lambda,fund_probs,precision=Fraction("1/1000"),max_epsilon=1 )
print("mode perturbation",ress)

ress = find_perturbation_pessopty(tree,tree_derived_probs_lambda,fund_probs,precision=Fraction("1/100"),max_epsilon=1)
for key in ress.keys():
    print ("P "+key, ress[key])

    
    
# <codecell>
print("Now considering uncertainty for sensitivity and specifity values")
tree = load_tree(file_name)

def tree_derived_probs_lambda(probs): 
    p=dict()
    p["gas"] = probs[0][0]
    sensitivity = probs[1][0]
    specifity = probs[2][0]
    p["no_gas"] = 1-p["gas"]
    p["pos._test"]=sensitivity*p["gas"]+(1-specifity)*p["no_gas"]
    p["neg._test"]=1-p["pos._test"]
    p["gas|pos._test"]=p["pos._test"] if p["pos._test"]==0 else sensitivity*p["gas"]/p["pos._test"]
    p["no_gas|pos._test"]=1-p["gas|pos._test"]
    p["gas|neg._test"]=0 if p["neg._test"] == 0 else (1-sensitivity)*p["gas"]/p["neg._test"]
    p["no_gas|neg._test"]=1-p["gas|neg._test"]
    return p
fund_probs = [[Fraction("7/10")],[Fraction("9/10")],[Fraction("7/10")]]

#solve the decision tree by passing probability dictionary
solve_tree(tree,tree_derived_probs_lambda(fund_probs))
#print a solved tree
print_tree(tree)

s = [1,0.1,0.1]

stabi = find_stability(tree,tree_derived_probs_lambda,fund_probs,precision=Fraction("1/100"),s=s,max_epsilon=1)
print ("stability", stabi)

ress = find_perturbation_mode(tree,tree_derived_probs_lambda,fund_probs,precision=Fraction("1/100"),s=s,max_epsilon=1)
print("mode perturbation",ress)

ress = find_perturbation_pessopty(tree,tree_derived_probs_lambda,fund_probs,precision=Fraction("1/100"),s=s,max_epsilon=1)
for key in ress.keys():
    print ("P "+key, ress[key])
