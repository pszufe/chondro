
Decision tree sensitivity analysis with Chondro
===============================================

Library overview
----------------

*Chondro* – is an analytical engine that implements the decision tree
(DT) sensitivity analysis (SA) algorithms described in the above
article. All the methods support both for separable and non-separable
decision trees. Chondro has been developed with the Python3 and has been
tested with Anaconda 2.3.0 running Python3 version 3.4.4.

The software can load files stored by the (available at
<http://www.silverdecisions.pl>) or uses internal JSON format. A DT is
presented as a Python dictionary structure with each node described with
a `type` (choice,decision,final), `id`, `value` (pay-off), and a list
(`nodes`) containing child nodes. The probability values `p` (for
separable DTs) or identifiers `pi` (for non-separable DTs) are stored in
children nodes of a chance node. Chondro
supports non separable DTs through injection of probability values as a
dictionary. In order to perform stability and perturbation analysis for
non-separable decision trees a function that generates probability
dictionary on the base of fundamental probabilities should be provided
to a respective algorithm. 

It should be noted that Chondro heavily relies on the Python `fractions`
package for numerical computing and hence enables calculation and
comparison of the exact values for P-optimal decisions. In this way we
managed to avoid numerical problems when expected values at different
nodes are equal.

Below you can find a sample separable decision tree and a corresponding JSON representation. The
probability and payoff values are given as string rather than number
values in order to enable a proper conversion with the `fractions`
module. 


    {
      "tree": {
        "type":"decision", "id":"d1",
        "nodes": [
          { 
            "type":"chance","id":"c1",
            "nodes": [
              {"p":"0.6","type":"final","id":"t1","value": "60" },
              {"p":"0.4","type":"final","id":"t2","value": "30" }        
            ]             
          },
          { 
            "type":"chance","id":"c2",
            "nodes": [
              {"p":"0.7","type":"final","id":"t3","value": "20" },
              {"p":"0.3","type":"final","id":"t4","value": "100" }        
            ]             
          },
          { 
            "type":"chance","id":"c3",
            "nodes": [
              {"p":"0.8","type":"final","id":"t5","value": "40" },
              {"p":"0.2","type":"final","id":"t6","value": "60" }             
            ]
          }     
        ]
      }
    }

A sample non-separable decision tree and a part of the corresponding JSON
representation.

    {
        "tree": {
            "type": "decision", "id": "d1",
            "nodes": [
                {
                    "type": "final","label":"sell","id": "t1","value": "800"
                }, 
                {
                    "type": "chance","label":"dig","id": "c1","value": "-300",
                    "nodes": [
                        {
                            "pi": "gas", "label":"gas",
                            "type": "final", "id": "t2", "value": "2500"
                        }, 
                        {
                            "pi": "no_gas", "label":"no_gas",
                            "type": "final", "id": "t3", "value": "0"
                        }
                    ] 
                }, 
                {
                    "type": "chance","label":"test","id": "c2","value": "-50",
                    "nodes": [
                        {
                            "pi": "neg._test", "label":"negative",
                            "type": "decision", "id": "d2",
                            "nodes": [
                                {
                                    "type": "final", 
                                    "label":"sell",
                                    "id": "t4", 
                                    "value": "600"
                                }, 
    (...)

Quick start - sensitivity analysis of separable trees
-----------------------------------------------------

A typical example session with Chondro might consist of the following
steps:

1.  create a JSON representation of a DT (either by saving a DT from
    SilverDecisions or manually creating a JSON file)

2.  Use the function to load a DT to memory

3.  Use the function to calculate optimal decision for the DT. The
    function supports non-separable trees by accepting a
    probability dictionary.

4.  perform the sensitivity analysis

    -   calculate the stability

    -   calculate $P_{mode,\varepsilon}$

    -   calculate $P_{\min,\varepsilon}$ and
        $P_{\max,\varepsilon}$

Listing \[lst:codesimple\] presents a sample code to solve the decision
tree. We first start by loading the module (line \[py:import\]). Next a
JSON file is loaded with the function. It should be noted that this
function supports JSON files in both internal dictionary format as well
as files that can be exported from SilverDecisions software (available
at <http://www.silverdecisions.pl>). The function (line \[py:solve\])
returns a tuple where the first element is the expected value of DT and
the second dictionary of optimal decisions.

It should be noted that in Chondro a decision tree is represented as a
python dictionary - a direct representation of the JSON file presented
in the Figure \[fig:json1\]. We assume that each node in a tree can be
identified by a path represented as a Python tuple of indices. The root
node is represented by `()` (an empty tuple) the first node (`c1` in
Listing \[lst:simpleresults\]) is represented by `(0,)` (one element
tuple) while the node `t3` could be presented as `(1,0)`. Node paths are
used to for represent $P$-optimal decisions. Further in the text the
tuple of node indices representing a path in the graph will be called
`node_path_tuple`. It should be noted that all sensitivity analysis
methods (, and ) provide a `use_labels` parameter. If it is set to
`True` (the default value) Chondro uses `label` attribute (see section
\[sec:jsonschema\]) of decision tree nodes rather than indices to
represent paths in the tree. For example consider a DT in Figure
\[fig:json2\]. The $P$-optimal decision can be presented a tuple of two
`node_path_tuple`s either as `((2, 0, 0), (2, 1, 1))` or
`((test, negative, sell), (test, positive, dig))`. The $P$-optimal
decision consists of two elements because the test performed at c2 node
can have two results - can be either positive or negative. Further in
the text a tuple of `node_path_tuple`s will be called
*`decision_path_tuple`*.

The function returns the optimal decision paths in the form of a
following dictionary:
$$\texttt{\{ node\_path\_tuple : [\textit{list of $P$-optimal node indices}] \}}$$
It should be noted that the function changes the state of the `tree`
object – now additionally it stores the data on optimal solution. The
function enables printing a human-readable textual representation of a
DT to the console (line \[py:print\]). A sample textual tree
representation can be found in Listing \[lst:simpleresults\].

Once a decision tree is loaded and solved a sensitivity analysis (SA)
can be performed. The first step is calculating the stability value and
$P_{mode,\varepsilon}$. The function (line \[py:fs\]) returns results in
the form of the following dictionary of $P$-optimal decisions:
$$\texttt{\{ decision\_path\_tuple : stability\_epsilon \} }$$

In the next step, the $P_{mode,\varepsilon}$ perturbations is calculated
(line \[py:pm\]). The output of the function is the following:
$$\texttt{\{ (e1,e2) : [\textit{list of} decision\_path\_tuple\textit{s}] \}}$$
where `e1` and `e2` represent some range of $\varepsilon$ values where
the set of $P$-optimal decisions does not change (please node that a
decision tree can have more than one $P$-optimal decision and hence a
list is here used).

Finally, $P_{\min,\varepsilon}$ and $P_{\max,\varepsilon}$ are
calculated with the function (line \[py:pp\]). The results will be
returned in the following format: $$\begin{split}   
        \texttt{\{ }
        \texttt{"max\_min" : \{ (e1,e2) : [\textit{list of} decision\_path\_tuple\textit{s}] \},}\\
        \texttt{"max\_max" : \{ (e1,e2) : [\textit{list of} decision\_path\_tuple\textit{s}] \}}
        \texttt{ \}}                  
    \end{split}$$ Both elements of the above dictionary (`max_min` and
`max_max`) are analogous to the output of the function. The value of
`max_min` represents $P_{\min,\varepsilon}$ perturbation sensitivity
while the value of `max_max` represents $P_{\max,\varepsilon}$
perturbation sensitivity.

It should be noted that for separable tree analysis Chondro supports the
$s\colon\mathcal{C}\rightarrow[0,1]$ function representing whether a
given chance node should be subject to sensitivity analysis. Simply add
the `s` property to any chance node in a separable tree. The default
value is $\texttt{s}=1$ i.e. all chance nodes in a separable tree will
be considered in sensitivity analysis of a DT.

    (*@\label{py:import}@*)from chondro import *

    file_name = "example_separable_Fig5.json"
    tree = load_tree(file_name)
    (*@\label{py:solve}@*)ev,dec=solve_tree(tree)
    print ("DT has been solved, the expected value is ev: "+str(ev)+ \
           " reachable decisions: "+str(get_reachable(dec)))

    (*@\label{py:print}@*)print_tree(tree)

    (*@\label{py:fs}@*)stabi = find_stability(tree,precision=Fraction("1/10000") )
    print ("DT stability", stabi)

    (*@\label{py:pm}@*)ress = find_perturbation_mode(tree,precision=Fraction("1/1000"))
    print("DT perturbation mode",ress)

    (*@\label{py:pp}@*)ress = find_perturbation_pessopty(tree,precision=Fraction("1/1000"))
    for key in ress.keys():
        print ("P_"+key, ress[key])     

    d1:decision
      *c1:chance (ev=48)
        t1:p=3/5 final [60]
        t2:p=2/5 final [30]
       c2:chance (ev=44)
        t3:p=7/10 final [20]
        t4:p=3/10 final [100]
       c3:chance (ev=44)
        t5:p=4/5 final [40]
        t6:p=1/5 final [60]

Sensitivity analysis of non-separable trees
-------------------------------------------

The Chondro library is capable of processing both separable and non
separable trees. Due to much larger computational complexity of
non-separable the library has different internal implementation of
stability and perturbation algorithms for both tree types. A sample
non-separable decision tree has been presented in Figure \[fig:json2\].

The support for non-separable trees is achieved by providing to the
function an additional parameter `derived_probs_dict` that contains a
dictionary of key-probability values that can be injected into `pi`
fields in a decision tree. Hence, there are two differences in
processing non-separable DTs compared to separable ones:

-   probabilities in the decision tree are represented as keys rather
    than values and use `pi` fields instead of `p` fields.

-   in order to perform sensitivity analysis a function needs to be
    provided that transforms fundamental probabilities into dictionary
    of key-value pairs that can be injected by Chondro into `pi` fields
    in a DT

Calculating stability and perturbation requires performing a sweep over
a set of fundamental probabilities and providing a function transferring
those probabilities into a key-probability dictionary. Hence, the
methods , and require providing two additional parameters:

-   *derived\_probs\_lambda* - a function that calculates derived
    probabilities on the base of fundamental ones. The function should
    return a dictionary where keys are corresponding to `pi` values in a
    decision tree.

-   *fundamental\_probs* - a list of initial vales of fundamental
    probabilities

An example function that calculates probabilities on the base of
fundamental ones has been presented in (e.g. see Listing
\[lst:codetrans\].

The initial values for fundamental probabilities is represented as a
list of events where each event is described by a list of outcome
probabilities. Moreover, if there are $n$ possible outcomes of an event
the probabilities of $n-1$ should be only passed - the last $n$-th
probability will be automatically calculated. For example suppose that
we consider two fundamental probabilities result of throwing a coin and
a result of throwing a four-sided dice. In that case the fundamental
probabilities in Chondro will be presented as:
`[[0.5],[0.25,0.25,0.25]]`.

In Listing \[lst:codetrans\] an example processing of a non-separable
decision tree has been presented. Firstly, fundamental probabilities
values need to be defined (line \[py:sep\_fund\_probs\]). Those values
can be used to calculate a $P$-optimal decision (line
\[py:sep\_solve\]). In the line \[py:sep\_s\] we defined $s$ values
representing whether a particular event (for which fundamental
probabilities have been given) should be a subject of sensitivity
analysis. Finally, we perform the stability analysis - in our
computations we limit the maximum considered value of $\varepsilon$ to
$1$.

    def tree_derived_probs_lambda(probs): 
        p=dict()
        p["gas"] = probs[0][0]
        sensitivity = probs[1][0]
        specifity = probs[2][0]
        p["no_gas"] = 1-p["gas"]
        p["pos._test"]=sensitivity*p["gas"]+  \
        (1-specifity)*p["no_gas"]
        p["neg._test"]=1-p["pos._test"]
        p["gas|pos._test"]=sensitivity*p["gas"]/p["pos._test"]
        p["no_gas|pos._test"]=1-p["gas|pos._test"]
        p["gas|neg._test"]=(1-sensitivity)*p["gas"]/  \
        p["neg._test"]
        p["no_gas|neg._test"]=1-p["gas|neg._test"]
        return p
        

    from chondro import *

    tree = load_tree("exaple_non_separable_fig7.json")
    (*@\label{py:sep_fund_probs}@*)fund_probs = [[Fraction("7/10")],[Fraction("9/10")],[Fraction("7/10")]]
    (*@\label{py:sep_solve}@*)solve_tree(tree,tree_derived_probs_lambda(fund_probs))
    print_tree(tree)
    (*@\label{py:sep_s}@*)s = [1,0.1,0.1]
    stabi = find_stability(tree,tree_derived_probs_lambda, \
              fund_probs,precision=Fraction("1/100"),s=s,max_epsilon=1) )
    print ("stability", stabi)
    ress = find_perturbation_mode(tree,tree_derived_probs_lambda,\
             fund_probs,precision=Fraction("1/100"),s=s,max_epsilon=1))
    print("mode perturbation",ress)
    ress = find_perturbation_pessopty(tree,tree_derived_probs_lambda,\
             fund_probs,precision=Fraction("1/100"),s=s,max_epsilon=1) )
    for key in ress.keys():
        print ("P "+key, ress[key])
