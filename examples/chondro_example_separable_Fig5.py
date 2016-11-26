# -*- coding: utf-8 -*-
'''
"chondro_example_separable_Fig5.py"

Example Chondro usage for sensitivity analysis of separable decision trees

Developed by:
  Przemyslaw Szufel
  Michal Jakubczyk
  Bogumil Kaminski

Copyright 2015 Przemyslaw Szufel & Michal Jakubczyk & Bogumil Kaminski
  {pszufe, mjakubc, bkamins}@sgh.waw.pl

This software is licensed under the terms of
open source GNU Affero General Public License, version 3
'''

from chondro import *


file_name = "example_separable_Fig5.json"
##Separable decision tree from the file:" + file_name
#load from a file
tree = load_tree(file_name)


#solve the decision tree
ev,dec=solve_tree(tree)
#visualise the solved tree
print_tree(tree)
#print the expected value and reachable decisions
print ("Expected value: "+str(ev)+" reachable decisions: "+str(get_reachable(dec)))


stabi = find_stability(tree,precision=Fraction("1/10000") )
print ("stability", stabi)

ress = find_perturbation_mode(tree,precision=Fraction("1/1000"))
print("mode",ress)

ress = find_perturbation_pessopty(tree,precision=Fraction("1/1000"))
for key in ress.keys():
    print (key, ress[key])

# <codecell>
print("Now considering a lesser uncertainty about probability in nodes c2 and c3")
tree['nodes'][0]['s']=1
tree['nodes'][1]['s']=0.5
tree['nodes'][2]['s']=0.5
stabi = find_stability(tree,precision=Fraction("1/10000") )
print ("stability", stabi)

ress = find_perturbation_mode(tree,precision=Fraction("1/1000"))
print("mode",ress)

ress = find_perturbation_pessopty(tree,precision=Fraction("1/1000"))
for key in ress.keys():
    print (key, ress[key])