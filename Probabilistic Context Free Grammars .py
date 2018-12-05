
# coding: utf-8

# # Probabilistic Context Free Grammars

# ## What are PCFGs ? A brief introduction

# #### The simplest augmentation of the context-free grammar is the Probabilistic Context Free Grammar (PCFG), <br/><br/>It is also known as the Stochastic Context Free Grammar (SCFG).

# #### Recall that a context-free grammar G is defined by four parameters ($N ,\sum , R , S$) <br/><br/>A probabilistic context-free grammar is also defined by four parameters, with a slight augmentation to each of the rules in R.

# ####  1. $N$ - the set of non terminals <br/><br/>2. $\sum$ - The set of terminal symbols <br/><br/>3. $R$ - A set of rules of production. Each rule is associated with Some probability<br/><br/>4. $S$ - Start Symbol

# #### Each rule has a probabilty assigned to it. <br/><br/> For example $ A$  $- > B$ $[p]$<br/><br/>Here $p$ is the probability that $A$ will be expanded to $B$. <br/><br/> Thus $P( A$ $- > B | A ) = p$<br/><br/> These associated probabilities are learned from a treebank i.e. a corpus of already parsed sentences.

# ## Sentence Disambiguation using PCFGs

# #### One sentence consists of many words where each word can have multiple senses.<br/><br/> Sentence disambiguation helps us too decide which is the correct sense.

# #### PCFGs can be used to perform sentence disambiguation. <br/><br/>The idea is for a given sentence , we will use the PCFG and the CKY algorithm to generate all parse trees for that sentence. <br/><br/>Calculate the probability of each parse tree and then select that one which has the highest probability.

# #### Thus, out of all parse trees with a yield of S, the disambiguation algorithm picks the parse tree $\hat{T}$ that is most probable given S : <br/><br/> $ \hat{T}(S) =$ $argmax$ $P(T|S)$<br/><br/> However by definition probability $P(T|S)$ can be rewritten as $P(T,S)$ $/$ $P(S)$, thus leading to<br/><br/> $ \hat{T}(S) =$ $argmax$ $\frac{P(T,S)}{P(S)}$<br/><br/> However $P(S)$ is a constant , thus we get <br/><br/>  $ \hat{T}(S) =$ $argmax$ $P(T,S)$

# #### In the sections below , we will see an a Python implementation of PCFGs and the CKY algorithm.

# #### Some utilities

# In[1]:


import copy
import sys
import random
import pandas
import numpy as np
from tabulate import tabulate
from turtle import *

class DefaultDict (dict):
    def __init__(self, default):
        self.default = default
    def __getitem__(self, key):
        if key in self: return self.get(key)
        return self.setdefault(key, copy.deepcopy(self.default))
    def sorted(self, rev=True):
        counts = [ (c,w) for w,c in self.items() ]
        counts.sort(reverse=rev)
        return counts

class CountingDict (DefaultDict):
    def __init__(self):
        DefaultDict.__init__(self, 0)
        
def conv_2_dict(**args): 
    # This function returns a dictionary with argument names as the keys, 
    # and the argument values as the key values.
    return args
        
def mappend(fn, list):
    # Append the results of calling fn on each element of list.
    return reduce(lambda x,y: x+y, map(fn, list))


# ### Now let us take a Probabilistic Context Free Grammar in Chomsky Normal Form  (CNF)
# #### We will encode the grammar into a Python dictionary as follows :-  <br/><br/> NP -> DetN [ p1 ]<br/><br/> NP -> N  [ p2 ]<br/><br/>  NP -> N PP [ p3 ]<br/><br/> will become NP = { ( 'Det' , 'N' ) : p1 , ( 'N' ) : p2 , ( 'N', 'PP' ) : p3 }

# In[2]:



grammar = conv_2_dict(
        #start symbol
        S = {('NPPL','NVPL'):.25, ('NPS','NVS'):.3, ('NPPL','VPPPL'):.2, ('NPS','VPPS'):.25},
    
        #noun phrase 
        NPPL = {('DetP', 'ADJNPL'):.4, ('DetP', 'NPL'):.6},
        NPS = {('DetS', 'ADJNS'):.3, ('DetS', 'NS'):.70},
    
        #adj plural and singular
        ADJNS = {('J', 'NS'):1},
        ADJNPL = {('J', 'NPL'):1},
    
        #verb singular/plural with noun phrase singular/plural
        NVS = {('VS','NPS'):1},
        NVPL = {('VPL','NPPL'):1},
    
        #singular/purple verb and noun with singular/plural prep phrase
        VPPPL = {('NVPL', 'PPPL'):.8, ('NVPL', 'PPS'):.2},
        VPPS = {('NVS', 'PPS'):.8, ('NVS', 'PPPL'):.1},
    
        #prep phrase
        PPS = {('P', 'NPS'):1},
        PPPL = {('P', 'NPPL'):1},
    
        #verbs that should be followed with "with"
        EDP = {('ED','P'):1},
    
        #training data
        DetS = {'the':.36, 'a':.65},
        DetP = {'the':1},
        P = {'with':1},
        J = {'red':.5, 'big':.5},
        NS = {'dog':float(1)/3, 'ball':float(1)/3, 'light':float(1/3)},
        NPL = {'dogs':.5, 'pickles':.5},
        VS = {'pickles':.25, 'sees':.25, 'liked':.25,'EDP':.25},
        VPL = {'see':float(1)/3, 'liked':float(1)/3, 'light':float(1)/3, 'EDP':1},
        ED = {'slept':1}
        )

#print(grammar)


# #### Now using the below function we can print the grammar in the normal human readable format

# In[3]:


# Prints the grammar in human readable form
def print_grammar(grammar):
    for i in grammar.items():
        left = i[0]
        prods = i[1]
        for j in prods.keys():
            print(left," -> ",j,"[",prods[j],"]")
        
print_grammar(grammar)


# ### The Probabilistic CKY parsing algorithm
# #### The parsing problem for PCFGs is to produce the most-likely parse $\hat{T}$ for a given sentence $S$, that is,<br/><br/>$\hat{T}(S) =$ $argmax$ $P(T)$<br/><br/>Most modern probabilistic parsers are based on the Probabilistic probabilistic CKY algorithm.<br/><br/>As with the CKY algorithm, we assume for the probabilistic CKY algorithm that the PCFG is in Chomsky normal form.
# #### Some helper functions for writing the CKY algorithm.

# In[4]:


#Argument is a list containing the rhs of some rule; return all possible lhs's"
def list_of_producers(output):
    results = []
    for (lhs,rhss) in grammar.items():
        for rhs in rhss:
            if rhs == output:
                results.append(lhs)
                
    return results

# Print the CKY table 
def print_CKY_table(table,wordlist):
    tab=[]
    for i in range(len(table)):
        tab.append(table[i][1:])
    data = np.array(tab)
    df = pandas.DataFrame(data,[0]*len(table),wordlist)
    print(tabulate(df, tablefmt="markdown", headers="keys"))


# Creates in the parse tree in the form of a nested tuple
def make_tree(x, trace, i, j, X):
    n = j - i
    if n == 1:
        return (X, x[i])
    else:
        Y, Z, s = trace[i, j, X]
        return (X, make_tree(x, trace, i, s, Y),
                   make_tree(x, trace, s, j, Z))
        


# #### The CKY parse function , returns True if the argument sentence is in the grammar; returns False otherwise <br/><br/>It also outputs the table used in the Probabilistic CKY algorithm and also displays the tree. 

# In[5]:


def CKY_parse(sentence):
   
    global grammar
    
    # Create the table; index j for rows, i for columns
    length = len(sentence)
    table = [None] * (length)
    table2 = DefaultDict(float)
    trace = {}

    for j in range(length):
        table[j] = [None] * (length+1)
        for i in range(length+1):
            table[j][i] = []
    
    
    # Fill the diagonal of the table with the POS tag of the words
    for k in range(1,length+1):
        results = list_of_producers(sentence[k-1])
        for item in results:
            list = (item, sentence[k-1])
            prob = grammar[item][sentence[k-1]]
            #print grammar[item][sentence[k-1]]
            table2[k-1,k, item] = prob
        table[k-1][k].extend(results)

    #core CKY part
    for width in range(2,length+1): 
        for start in range(0,length+1-width): 
            end = start + width 
            for mid in range (start, end): 
                max_score = 0
                args = None
                for x in table[start][mid]: 
                    for y in table[mid][end]:
                        #print x,y
                        results = list_of_producers((x,y))
                        for item in results:
                            prob1 = grammar[item][(x,y)]
                            prob2 = prob1 * table2[start, mid, x] * table2[mid, end, y]
                            checkme = start, end, item
                            if checkme in table2:
                                if prob2 > table2[start, end, item]:
                                    table2[start, end, item] = prob2
                            else:
                                table2[start, end, item] = prob2
                            args2 = x, y, mid
                            if args2 in trace:
                                if prob2 > table2[start, end, item]:
                                    args = x, y, mid
                                    trace[start, end, item] = args
                            else:
                                args = x, y, mid
                                trace[start, end, item] = args
                            trace[start, end, item] = args
                            if item not in table[start][end]:
                                table[start][end].append(item)


    # Print the table
    print ("The CKY Algorithm Table\n")
    print_CKY_table(table, sentence)


    print ("\nThe Parse Tree\n")
    if table2[0, length-1, 'S']:
        mytree = make_tree(sentence, trace, 0, length, 'S')
        
    print(mytree)
      
CKY_parse('the pickles light the dogs'.split())


# ### Problems with PCFGs

# #### While probabilistic context-free grammars are a natural extension to context-free grammars, they have two main problems as probability estimators :- <br/><br/> 1. CFG rules impose an independence assumption on probabilities, resulting in poor modeling of structural dependencies across the parse tree. <br/><br/>  2. CFG rules don’t model syntactic facts about specific words, leading to problems with subcategorization ambiguities, preposition attachment, and coordinate structure ambiguities.<br/><br/>  Because of these problems, most current probabilistic parsing models use some augmented version of PCFGs,
