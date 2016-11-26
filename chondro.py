# -*- coding: utf-8 -*-
'''
"chondro.py"

Sensitivity analysis for separable and non-separable decision trees

Developed by:
  Przemyslaw Szufel
  Michal Jakubczyk
  Bogumil Kaminski

Copyright 2016 Przemyslaw Szufel & Michal Jakubczyk & Bogumil Kaminski
  {pszufe, mjakubc, bkamins}@sgh.waw.pl

This software is licensed under the terms of
open source GNU LESSER GENERAL PUBLIC LICENSE v. 3
'''
import json
from fractions import Fraction
from copy import deepcopy
from os.path import isfile


def go_fractions(node):
    '''
    Recursively enables Fractions for a dictionary decision tree.
    The parameter types of 'p' and 'value' fields are changed fractions

    *node* - tree or subtree
    '''
    if 'value' in node:
        node['value'] = Fraction(node['value']).limit_denominator()
    if 'p' in node:
        node['p'] = Fraction(node['p']).limit_denominator()
    if 'nodes' in node:
        for nnn in node['nodes']:
            go_fractions(nnn)


def perturb(probs, v, epsilon, maximize):
    '''
    Calculates a peturbation for a given parameter set

    *probs* - list initial probabilities

    *v* - payoff values

    *epsilon* - max perturbation

    *maximize* - maximize (True) or minimize payoff (False)
    '''
    assert len(probs) == len(v)
    assert sum(probs) == 1
    assert min(probs) >= 0
    # get indexes of values in a sorted list
    order = [y for x, y in sorted([(y, x) for x, y in enumerate(v)],
                                  reverse=maximize)]
    # need a copy of probability list
    newp = deepcopy(probs)
    # the order list will be analyzed from both sides (i.e. we first perturb
    # probabilities of nodes with the highest and lowes values)
    lo = 0
    hi = len(probs) - 1
    while lo < hi:
        ix_lo = order[lo]
        ix_hi = order[hi]
        # check how much the probability of low node can be increased
        inc = min(epsilon + probs[ix_lo] - newp[ix_lo],
                  1 - newp[ix_lo])
        # check how much the probability of high node can be decreased
        dec = min(epsilon - probs[ix_hi] + newp[ix_hi],
                  newp[ix_hi])
        if inc == 0:
            lo += 1
            continue
        if dec == 0:
            hi -= 1
            continue
        # the increase value must match the decrease value
        move = min(inc, dec)
        # update the probabilities of the high and low node
        newp[ix_lo] += move
        newp[ix_hi] -= move
    return newp


def load_tree(file_name, use_fractions=True):
    '''
    Loads a decision tree defintion from a give file.
    The file can be either a plain dict saved to JSON or a SilverDecisions JSON
    file or a file from previous SilverDecisions version. 
    If a SilverDecisions file has several trees only the first is read.

    *file_name* - a name of the file

    *use_fractions* - should "fractions" module be used for computational \
    accurancy
    '''
    f = open(file_name, "r")
    d = json.load(f)
    f.close()
    tree = dict()
    if isinstance(d, list) and '$type' in d[0]:
        # This is a old version Silverlight SilverDecisions file
        def create_node_from_sd(node_sd):
            '''
            parses a SilverDecisions *.dft node
            '''
            node_type = None
            if "DecisionNode" in node_sd['$type']:
                node_type = "decision"
            if "EndNode" in node_sd['$type']:
                node_type = "final"
            if "ChanceNode" in node_sd['$type']:
                node_type = "chance"
            node = {}
            node['type'] = node_type
            node['id'] = node_sd['Values']['Label']['Value']
            if not node_type == 'final':
                node['nodes'] = []
            return node
        nodes = {}
        root = d[0]
        tree = create_node_from_sd(root['Node'])
        nodes[root['Id']] = tree
        for i in range(1, len(d)):
            if 'Node' in d[i] and ("DecisionNode" in d[i]['Node']['$type'] or
                                   "EndNode" in d[i]['Node']['$type'] or
                                   "ChanceNode" in d[i]['Node']['$type']):
                nodes[d[i]['Id']] = create_node_from_sd(d[i]['Node'])
            elif "Connection" in d[i]['$type']:
                child_node = nodes[d[i]['ChildId']]
                parent_node = nodes[d[i]['ParentId']]
                if parent_node['type'] == "chance":
                    child_node['p'] = d[i]['Connection']['Probability']
                    if use_fractions:
                        child_node['p'] = \
                            Fraction(child_node['p']).limit_denominator()
                    child_node['p'] /= 100  # SilverDecisions uses %
                if child_node['type'] == \
                        "final" or d[i]['Connection']["Payoff"] != 0.0:
                    child_node['value'] = d[i]['Connection']["Payoff"]
                    if use_fractions:
                        child_node['value'] = Fraction(child_node['value']).\
                            limit_denominator()
                parent_node['nodes'] += [child_node]
    elif "trees" in d: 
        #this is a new JavaScript Silverdecisions file
        tree = None
        if len(d["trees"]) > 0:
            node = d["trees"][0]
            tree = dict()
            tree = build(node)
        if use_fractions:
            go_fractions(tree)
    else:
        tree = d['tree']
        if use_fractions:
            go_fractions(tree)
    return tree


def build(node, prob=None, value=None):
    if node is None or "type" not in node:
        return {}
    res = {}
    res["type"] = node["type"] if node["type"] != "terminal" else "final"
    res["id"] = node["name"] if "name" in node else ""
    if prob is not None and prob != "":
        res["p"] = prob
    if value is not None and value != "":
        res["value"] = value
    if 'childEdges' in node and len(node['childEdges'])>0:
        res["nodes"] = [build(n['childNode'], n["probability"] if "probability" in n else None,n["payoff"] if "payoff" in n else None ) for n in node['childEdges']]
    return res
    

def save_tree(tree, file_name, overwrite=False):
    '''
    Saves the tree definition to a file.

    *tree* - a dictionary representation of a decision tree

    *file_name* - a name of the file

    *overwrite* - should the file be overwritten if it exists
    '''
    if isfile(file_name) and not overwrite:
        raise Exception("The file " + str(file_name) +
                        " already exists and overwrite is set to False")
    f = open(file_name, "w")
    tree2 = deepcopy(tree)

    def tidy_keys(node):
        '''
        leave only a limited set of keys
        '''
        keys = list(node.keys())
        for key in keys:
            if key in [
                    "p",
                    "pi",
                    "type",
                    "nodes",
                    "id",
                    "value",
                    "label",
                    "s"]:
                if isinstance(node[key], Fraction):
                    node[key] = str(node[key])
            else:
                del node[key]
        if "p" in keys and "pi" in keys:
            del node["p"]
        if "nodes" in node:
            for nn in node["nodes"]:
                tidy_keys(nn)
    tidy_keys(tree2)
    json.dump({'tree': tree2}, f, indent=4)
    f.close()


def print_tree(node, __level=0):
    '''
    Recursively prints a decision tree subtree.

    *node* - tree root or sub-node

    *__level* - internal parameter of the function - level of the tree
    '''
    print(("  " * __level) +
          (("*" if node["best"] else " ") if "best" in node else "") +
          ((node["label"] + "%") if "label" in node else "") +
          (node["id"] if "id" in node else "") +
          (":" if "label" in node or "id" in node else "") +
          (("p=" + str(node["p"]) + " ") if "p" in node else "") +
          node['type'] +
          ((" [" + str(node["value"]) + "]") if "value" in node else "") +
          ((" (ev=" + str(node["ev"]) + ")") if node['type'] != 'final' and
           "ev" in node else ""))
    if "nodes" in node:
        for nn in node['nodes']:
            print_tree(nn, __level=__level + 1)


def solve_tree(node, derived_probs_dict=None, node_stability_type=None,
               decision=None, __branching=(), __best_path=None):
    '''
    Recursive method to find an optimal value for a decision tree.

    *node* - tree or sub-tree to be calculated

    *derived_probs_dict* - dictionary of probabilities for non-separable trees\
    (if present a node will use those probability values instead of 'p')

    *node_stability_type* - a lambda (probs,evs,best_path,s_i) to recalculate \
    probabilities at chance nodes

    *decision* - decision dictionary to calculate ev of a tree for \
    a particular decision

    *__branching* - internal method parameter - current position in the tree

    *__best_path* - internal method parameter - best paths in the tree

    The function returns a tuple of expected value and optimal decisions paths.
    The optimal decisions paths are represented as the following dictionary:
    { node_path_tuple : list of P-optimal node indices }

    WARNING! The function has the following side effects on the tree parameter:

    - expected values of chance nodes will be saved in the `ev' field

    - children of a decision node will store a `best' boolean field with True
    value indicating P-optimal decisions

    '''
    if node['type'] == 'final':
        # just return value and an empty optimal decision dictionary
        return (node['value'], dict())
    elif node['type'] == 'decision':
        best_nodes = []  # nodes with the highest expected value
        best_indexes = []  # indexes of the above nodes
        ix = 0  # index of the current node
        dec_struct = dict()
        for nn in node['nodes']:
            # recursively solve subtree calculating expected values
            # side effect - the expected value is stored in the node
            nn['ev'], dec_struct_t = solve_tree(
                nn, derived_probs_dict, node_stability_type, decision,
                __branching + (ix,), nn['best'] if 'best' in nn else None)
            # merge the structure of optimal decisions
            dec_struct.update(dec_struct_t)
            nn['best'] = False
            if decision is None:
                if len(best_nodes) == 0 or nn['ev'] > best_nodes[0]['ev']:
                    # setup a new list of best nodes
                    # (overwriting the existing if any)
                    best_nodes = [nn]
                    best_indexes = [ix]
                elif nn['ev'] == best_nodes[0]['ev']:
                    # add a best node to the list
                    best_nodes += [nn]
                    best_indexes += [ix]
            ix += 1  # update node index in the loop
        if decision is not None:
            if __branching in decision:
                best_indexes = decision[__branching]
                # print("best_indexes[0]",best_indexes[0])
                best_nodes = [node['nodes'][best_indexes[0]]]
            else:
                best_indexes = [0]
                best_nodes = [node['nodes'][0]]
        for best_node in best_nodes:
            best_node['best'] = True
        # update decision structure - store indexes of optimal decisions
        dec_struct[__branching] = best_indexes
        return ((node['value'] if 'value' in node else 0) +
                best_nodes[0]['ev'], dec_struct)
    elif node['type'] == 'chance':
        ev = 0  # expected value at the node
        ix = 0  # index of the current node
        dec_struct = dict()
        for nn in node['nodes']:
            # recursively solve subtree calculating expected values
            # side effect - the expected value is stored in the node
            nn['ev'], dec_struct_t = solve_tree(
                nn, derived_probs_dict, node_stability_type, decision,
                __branching + (ix,), __best_path)
            # merge the structure of optimal decisions
            dec_struct.update(dec_struct_t)
            # derived_probs_dict represents a dictionary of probabilities for\
            # the non-separable trees
            if derived_probs_dict is not None and 'pi' in nn:
                nn['p'] = derived_probs_dict[nn['pi']]
            ev += nn['ev'] * nn['p']
            ix += 1  # update node index in the loop
        if node_stability_type is not None:
            # the expected value at the node will be recalculated
            ev = 0
            # probabilities of children of this chance node
            probs = [nn['p'] for nn in node['nodes']]
            # expected values of children of this chance node
            evs = [nn['ev'] for nn in node['nodes']]
            # call the stability(probs, evs,best_path,s) method
            # in order to calculate new probabilities for this node
            newprobs = node_stability_type(probs, evs, __best_path, node['s']
                                           if 's' in node else 1)
            for i in range(len(node['nodes'])):
                # side effect - updating probabilities in the node
                node['nodes'][i]['p'] = newprobs[i]
                ev += node['nodes'][i]['ev'] * newprobs[i]
        return ((node['value'] if 'value' in node else 0) + ev, dec_struct)
    else:
        raise Exception("uknown node type for node ", str(node))


def bisect(fun, a, b, precision=0.000001, divide=2.0):
    '''
    Implements bisection with a given precision.

    *fun* - a boolean function

    *a* - start value

    *b* - end value

    *precision* - precision parameter

    *divide* - division parameter

    examples:

    bisect(lambda x: x*x-5 > 0, 0.0,5.0)

    bisect(lambda x: x*x-5 > 0, Fraction(0),Fraction(5),divide=Fraction(2))
    '''
    start = a
    end = b
    while end - start > precision:
        point = (start + end) / divide
        if fun(point):
            end = point
        else:
            start = point
    return (start, end)


def bisect_change(fun, start_value, a, b, precision=0.000001):
    '''
    Implements bisection with a given precision - searching for a change
    in function value.

    *fun* - a boolean function

    *start_value* - a starting value of the function

    *a* - start value

    *b* - end value

    *precision* - precision parameter

    returns x where the change was observed and the new function value
    '''
    start = a
    end = b
    point_change = start
    value_change = start_value
    while end - start > precision:
        point = (start + end) / 2.0
        value = fun(point)
        if value != start_value:
            end = point
            point_change = point
            value_change = value
        else:
            start = point
    # handling a border case where no change has been found
    if end == b:
        return(b, fun(b))
    else:
        return (point_change - precision, value_change)


def find_gamma_probs(probs, e):
    '''
    Finds a gamma value for the most probable values.

    *probs* - a list of probabilities

    *e* - epsilon

    examples:

    find_gamma_probs([Fraction("1/4"),Fraction("3/8"),Fraction("3/8")],\
    Fraction(0.249999))

    find_gamma_probs([Fraction("1/4"),Fraction("3/8"),Fraction("3/8")],\
    Fraction(0.25))

    find_gamma_probs([Fraction("1/4"),Fraction("3/8"),Fraction("3/8")],0.1)
    '''

    def distance(probs1, probs2):
        '''
        calculate maximum distance between two lists or dictionary
        probs1 - the first list/dictionary of probababilities
        probs2 - the second list/dictionary of probababilities
        '''
        return max([abs(probs1[i] - probs2[i]) for i in range(len(probs1))])

    # STEP 1. Check if calculations are allowed
    # if epsilon=0 no calculations are performed
    if e == 0:
        return probs
    # STEP 2. Check if gamma == infinity is within the epsilon range
    # indexes of maximum values
    mms = [i for i, j in enumerate(probs) if j == max(probs)]
    maxed = [0] * len(probs)  # zero-filled list
    for mm in mms:
        maxed[mm] = 1.0 / len(mms)
    if distance(probs, maxed) <= e:
        return maxed
    # STEP 3. gamma < infinity - calculate the value
    # create the gamma function for the given probabilities

    def gamma_f(gamma):
        '''
        implementation of the gamma function
        '''
        return [p**gamma / sum([p2**gamma for p2 in probs]) for p in probs]

    # first we find the gamma in the integer domain.
    # We do so in order to use full precision arithmetic - hence we avoid
    # number overflow problems
    g_value_a = bisect(
        lambda g: distance(
            probs,
            gamma_f(
                g + 1)) >= e,
        Fraction(0),
        Fraction(1024),
        precision=Fraction(1),
        divide=Fraction(2))
    # check for numerical problems
    num_problems = max([probs[i] > 0 and
                        probs[i]**(int(g_value_a[1]) + 2.0) == 0.0
                        for i in range(len(probs))])
    # Now a more exact value can be calculated with the given precision
    g_value = g_value_a if num_problems else \
        bisect(lambda g: distance(probs, gamma_f(g)) >= e, 1,
               int(g_value_a[1] + 1), precision=0.000000001)
    return gamma_f(g_value[0])


def get_reachable(dec):
    '''
    For a given decision strategy dictionary creates a copy
    with removed optimal decisions that are not reachable.

    *dec* - decision dictionary

    example:

    get_reachable({():[0,1],(1,):[2,3],(2,):[1,2],(2,2):[0]})  \
    will return  {(): [0, 1], (1,): [2, 3]} because the nodes (2,) and (2,2) \
    cannot be reached - at the root node the only optimal decisions are 0 and 1
    '''
    res = {}
    for key in dec.keys():
        key_ok = True
        for k in range(len(key)):
            # takes every subpath of the key
            sub = key[:k]
            # if subpath is not in dec dictionary it means is valid
            # (i.e. choice are related to chance nodes rather than decisons)
            if sub in dec:
                key_ok = key_ok and max(
                    [key[:k + 1] == sub + (f,) for f in dec[sub]])
        if key_ok:
            res[key] = dec[key]
    return res


def find_stability(tree, derived_probs_lambda=None,
                   grouped_fundamental_probs=None, precision=Fraction("1/10"),
                   max_epsilon=None, s=None, use_labels=True):
    '''
    Finds the stability coefficient for every optimal decision
    in a given separable or nonseparable tree.

    *tree* - a decision tree represented as a dictionary

    *derived_probs_lambda* - a function that calculates derived probabilities \
    on the  base of fundamental ones

    *fundamental_probs* - a list of groups of fundamental probabilities

    *precision* - precision (for separable) or sweep step \
    (for non separable trees)

    *max_epsilon* - maximum value of an epsilon in the sweep

    *s* - sensitivity list for fundamental probabilities

    *use_labels* - decisions will be presented as labels rather than indices

       Returns results in the form of the following dictionary of P-optimal
    decisions:

    { (decision_path_tuple) : stability_epsilon }
    '''
    assert s is None or max(s) > 0
    assert \
        (derived_probs_lambda is None and
         grouped_fundamental_probs is None and
         s is None) or (derived_probs_lambda is not None and
                        grouped_fundamental_probs is not None)
    res = None
    if max_epsilon is None:
        if s is None:
            max_epsilon = 1
        else:
            max_epsilon = 1 / min([ss for ss in s if ss > 0])

    if derived_probs_lambda is None:
        # find stability separable
        def epsilon(e):
            '''
            calculate perturbations
            '''
            return lambda probs, evs, best_path, s_i:\
                probs if best_path is None \
                else perturb(probs, evs, e * s_i, not best_path)
        dec_struct_curr = solve_tree(tree)[1]
        dec_struct_end = solve_tree(
            deepcopy(tree), node_stability_type=epsilon(max_epsilon))[1]

        dec_struct_curr = get_reachable(dec_struct_curr)
        dec_struct_end = get_reachable(dec_struct_end)

        def fun(e):
            '''
            returns reachable decision with epsilon perturbation
            '''
            return get_reachable(
                solve_tree(
                    deepcopy(tree),
                    node_stability_type=epsilon(e))[1])
        if dec_struct_curr == dec_struct_end:
            return {tuple(d): 1 for d in tree_sweep(dec_struct_curr)}
        e = bisect_change(fun, dec_struct_curr, 0, 1, precision)[0]
        res = {tree_decision_paths_to_tuple(
            dec): e for dec in tree_sweep(dec_struct_curr)}
    else:
        # find stability non separable
        base_probs = derived_probs_lambda(grouped_fundamental_probs)
        base_dec = solve_tree(tree, base_probs)[1]
        base_dec = get_reachable(base_dec)
        # all optimal decisioe presented as full paths
        opti_decisions = tree_sweep(base_dec)
        sweep = full_sweep(grouped_fundamental_probs, step=precision,
                           max_epsilon=max_epsilon, s=s)
        res = {}
        for point_ix in range(len(sweep)):
            point = sweep[point_ix]
            if point[0] == 0:
                continue

            dec = solve_tree(tree, derived_probs_lambda(point[1]))[1]
            dec = get_reachable(dec)
            dec_sweep = tree_sweep(dec)
            remove = []
            for odec in opti_decisions:
                if odec not in dec_sweep:
                    remove += [odec]
                    res[tree_decision_paths_to_tuple(odec)] = \
                        find_previous_epsilon(sweep, point_ix)
                    # print('###',tree_decision_paths_to_tuple(odec),"current
                    # eps",point[0],"POINT:",point[1])
            for r in remove:
                opti_decisions.remove(r)
            if len(opti_decisions) == 0:
                break
        for dec in opti_decisions:
            if tree_decision_paths_to_tuple(dec) not in res:
                res[tree_decision_paths_to_tuple(dec)] = 1
    if use_labels:
        for key in list(res.keys()):
            res[get_decision_name(tree, key)] = res.pop(key)
    return res


def find_previous_epsilon(sweep, point_ix):
    '''
    finds in a sorted sweep a point with lower epsilon value
    '''
    previous_epsilon = 0
    for ix in range(1, point_ix):
        if sweep[point_ix - ix][0] < sweep[point_ix][0]:
            previous_epsilon = sweep[point_ix - ix][0]
            break
    return previous_epsilon


def full_sweep(grouped_probs, step=Fraction("1/5"), max_epsilon=None, s=None):
    '''
    Generates a parameter sweep for a given set of grouped probabilities
    The corner probabilities 0 and 1 are included in the sweep if they are
    in the max_epsilon range.

    The sum of probabilities in any group does not exceed 1.

    *grouped_probs* - a list of probabilities groups [[p1, p2],[p3],[p4,p5]]

    *step* - step value used to generate the parameter sweep

    *max_epsilon* - maximum value of an epsilon in the sweep
    '''

    assert max_epsilon is None or max_epsilon >= 0
    assert s is None or max(s) > 0
    for group in grouped_probs:
        assert sum(group) <= 1
    if max_epsilon is None:
        if s is None:
            max_epsilon = 1
        else:
            max_epsilon = 1 / min([ss for ss in s if ss > 0])
    # sweep valuea from every variable
    # the values are grouped the same way grouped_probs are grouped
    sweep_vals = []
    for group_ix in range(len(grouped_probs)):

        group_sweep = []
        s_val = 1 if s is None else s[group_ix]
        for prob_ix in range(len(grouped_probs[group_ix])):
            prob = grouped_probs[group_ix][prob_ix]
            val = prob
            if s_val <= 0.00000000000001:
                group_sweep += [[(0, val / 1)]]
                continue
            # list of tuples (distance_from_epsilon, value)
            vals = []
            while val > 0 and prob - val <= max_epsilon * s_val:
                vals += [(abs(prob - val) / s_val, val)]
                val -= step
                # print("vals1",vals)
            if prob - 0 <= max_epsilon * s_val:
                vals += [(prob / s_val, 0)]
                # print("vals2",vals)
            if prob < 1:
                val = prob + step
                while val < 1 and val - prob <= max_epsilon * s_val:
                    vals += [(abs(prob - val) / s_val, val)]
                    val += step
                    # print("vals3",vals)
                if 1 - prob <= max_epsilon * s_val:
                    vals += [((1 - prob) / s_val, 1)]
                    # print("vals4",vals)
            vals.sort()

            group_sweep += [vals]
        # print([group_sweep])
        sweep_vals += [group_sweep]

    # sweep_vals now holds all possible values for each variable
    # now sweep values will be generated for every variable group
    sweeped_groups = []
    for group_ix in range(len(grouped_probs)):
        sweep_group = sweep_vals[group_ix]
        other_probability = 1 - sum(grouped_probs[group_ix])
        res = []
        sweep_len = 1
        sizes = [len(sg) for sg in sweep_group]
        for size in sizes:
            sweep_len *= size
        for i in range(sweep_len):
            vals = ()
            vv = i
            max_dist = 0
            for j in range(len(sizes)):
                ix = vv % sizes[j]
                vals += (sweep_group[j][ix][1],)
                max_dist = max(max_dist, sweep_group[j][ix][0])
                vv = (vv - ix) // sizes[j]
            if (sum(vals)) < 1.00000000000001:
                # check if the 'other' probability is within the given epsilon
                max_dist = max(
                    max_dist, abs(
                        (1 - sum(vals)) - other_probability))
                if max_dist <= max_epsilon:
                    # add the value to the parameter sweep
                    res += [(max_dist, vals)]
        res.sort()
        sweeped_groups += [res]

    # now we in sweeped_groups have sweeped parameter groups
    # res will hold a cartesian product of those variable groups
    res = []
    sweep_len = 1
    sizes = [len(sg) for sg in sweeped_groups]
    for size in sizes:
        sweep_len *= size
    for i in range(sweep_len):
        vals = ()
        vv = i
        max_dist = 0
        for j in range(len(sizes)):
            ix = vv % sizes[j]
            vals += ((sweeped_groups[j][ix][1]),)
            max_dist = max(max_dist, sweeped_groups[j][ix][0])
            vv = (vv - ix) // sizes[j]
        res += [(max_dist, vals)]
    res.sort()
    return res


def find_perturbation_pessopty(
        tree,
        derived_probs_lambda=None,
        grouped_fundamental_probs=None,
        precision=Fraction("1/10"),
        max_epsilon=None,
        s=None,
        use_labels=True):
    '''
    Performs a full grid sweep search in order to find optimistic and
    perturbation decision stabilit values.
    The grid contains the original probabilities as well as the corner cases
    (probabilities equal to zero and one).
    If only epsilon stability is selected the search stops after finding the
    first set of fundamental probabilities that changes the decision.

    *tree* - a decision tree represented as a dictionary

    *derived_probs_lambda* - a function that calculates derived probabilities \
    on the base of fundamental ones

    *fundamental_probs* - a list of groups of fundamental probabilities

    *step* - step value for the parameter sweep

    *max_epsilon* - maksimum epsilon value for parameter sweep generation

    *s* - the s parameter for fundamental probabilities

    *use_labels* - decisions will be presented as labels rather than indices

    Output format:

    {
        "max_min" : { (e1,e2) : [list of decision_path_tuples },
        "max_max" : { (e1,e2) : [list of decision_path_tuples }
    }
    The value of max_min represents P_min perturbation sensitivity
    while the value of max_max represents P_max perturbation sensitivity.
    e1 and e2 represent some range of epsilon values where the set of
    P-optimal decisions does not change.
    Please node that a decision tree can have more than one P-optimal decision
    and hence a list is here used.
    '''
    assert \
        (derived_probs_lambda is None and grouped_fundamental_probs is None and
         s is None) or (derived_probs_lambda is not None and
                        grouped_fundamental_probs is not None)
    assert s is None or max(s) > 0
    if max_epsilon is None:
        if s is None:
            max_epsilon = 1
        else:
            max_epsilon = 1 / min([ss for ss in s if ss > 0])
    stab_table = {}
    if derived_probs_lambda is None:
        # separable tree
        def optimistic(e):
            '''
            optimistic perturbation for a given epsilon
            '''
            return lambda probs, evs, best_path, s_i:\
                probs if best_path is None \
                else perturb(probs, evs, e * s_i, True)

        def pessimistic(e):
            '''
            pessimistic perturbation for a given epsilon
            '''
            return lambda probs, evs, best_path, s_i:\
                probs if best_path is None \
                else perturb(probs, evs, e * s_i, False)
        stab_types = {"max_min": pessimistic, "max_max": optimistic}
        for stab_name in stab_types.keys():
            node_stability_type = stab_types[stab_name]
            ranges = {}
            count = int(max_epsilon / precision)
            epsilon = 0
            last_dec = None
            last_epsilon = None
            range_start = 0
            for step in range(count + 1):
                ev, dec = solve_tree(
                    deepcopy(tree),
                    node_stability_type=node_stability_type(epsilon))
                dec = tree_sweep(get_reachable(dec))
                last_epsilon = epsilon
                epsilon = min((step + 1) * precision, max_epsilon)
                if last_dec is not None and last_dec != dec:
                    ranges[(range_start, last_epsilon)] = last_dec
                    range_start = epsilon
                last_dec = dec
            if range_start <= max_epsilon:
                ranges[(range_start, max_epsilon)] = last_dec
            stab_table[stab_name] = ranges
    else:
        # non separable tree
        sweep = full_sweep(grouped_fundamental_probs, step=precision,
                           max_epsilon=max_epsilon, s=s)
        # convert list to tuple - only tuples are hashable
        tree_branching = node_branching(tree)
        tree_decisions = tree_sweep(tree_branching)

        epsilons = []
        for s in sweep:
            if len(epsilons) == 0 or epsilons[-1] < s[0]:
                epsilons.append(s[0])
        max_min = {}
        max_max = {}
        for decision_ix in range(len(tree_decisions)):
            min_point_epsilon = {}
            max_point_epsilon = {}
            decision = tree_decisions[decision_ix]
            last_epsilon_min = None
            last_epsilon_max = None
            for point in sweep:
                ev, dec = solve_tree(tree, derived_probs_lambda(point[1]),
                                     decision=decision)
                if len(min_point_epsilon) == 0:  # epsilon = 0
                    min_point_epsilon[point[0]] = ev
                    max_point_epsilon[point[0]] = ev
                    last_epsilon_min = point[0]
                    last_epsilon_max = point[0]
                else:
                    if ev <= min_point_epsilon[last_epsilon_min]:
                        min_point_epsilon[point[0]] = ev
                        last_epsilon_min = point[0]
                    else:
                        min_point_epsilon[point[0]] = \
                            min_point_epsilon[last_epsilon_min]

                    if ev >= max_point_epsilon[last_epsilon_max]:
                        max_point_epsilon[point[0]] = ev
                        last_epsilon_max = point[0]
                    else:
                        max_point_epsilon[point[0]] = \
                            max_point_epsilon[last_epsilon_max]
            for epsilon in epsilons:
                max_min[(decision_ix, epsilon)] = min_point_epsilon[epsilon]
                max_max[(decision_ix, epsilon)] = max_point_epsilon[epsilon]

        var_mms = {"max_min": max_min, "max_max": max_max}
        for var_mm_name in var_mms.keys():
            var_mm = var_mms[var_mm_name]
            range_start = 0
            last_best = None
            last_epsilon = None
            ranges = {}
            for epsilon in epsilons:
                best_decisions = []
                best_pay = -1e99
                for decision_ix in range(len(tree_decisions)):
                    decision = tree_decisions[decision_ix]
                    if var_mm[(decision_ix, epsilon)] > best_pay:
                        best_pay = var_mm[(decision_ix, epsilon)]
                        best_decisions = [decision]
                    elif var_mm[(decision_ix, epsilon)] == best_pay:
                        best_decisions.append(decision)
                if last_best is not None and last_best != best_decisions:
                    ranges[(range_start, last_epsilon)] = last_best
                    range_start = epsilon
                if epsilon == epsilons[-1]:
                    ranges[(range_start, max_epsilon)] = best_decisions
                last_best = best_decisions
                last_epsilon = epsilon
            stab_table[var_mm_name] = ranges
    for k in stab_table.keys():
        ranges = stab_table[k]
        for key in ranges:
            ranges[key] = [
                get_decision_name(tree, tree_decision_paths_to_tuple(dec1))
                if use_labels else
                get_decision_name(tree, tree_decision_paths_to_tuple(dec1))
                for dec1 in ranges[key]]
    return stab_table


def find_perturbation_mode(
        tree,
        derived_probs_lambda=None,
        grouped_fundamental_probs=None,
        precision=Fraction("1/100"),
        max_epsilon=None,
        s=None,
        use_labels=True):
    '''
    Finds peturbation stability for mode-perturbation type
    The grid contains the original probabilities as well as the corner cases
    (probabilities equal to zero and one).

    *tree* - a decision tree represented as a dictionary

    *derived_probs_lambda* - a function that calculates derived probabilities\
    on the base of fundamental ones

    *grouped_fundamental_probs* - a list of groups of fundamental probabilities

    *precision* - step value for the gamma parameter sweep

    *max_epsilon* - maksimum epsilon value considered in the computation

    *s* - sensitivity list for fundamental probabilities

    *use_labels* - decisions will be presented as labels rather than indices

    Output format:

    { (e1,e2) : [list of decision_path_tuples }
    The values e1 and e2 represent some range of epsilon values where the set
    of  P-optimal decisions does not change.
    Please node that a decision tree can have more than one P-optimal decision
    and hence a list is here used.
    '''
    assert max_epsilon is None or max_epsilon >= 0
    assert \
        (derived_probs_lambda is None and grouped_fundamental_probs is None and
         s is None) or (derived_probs_lambda is not None and
                        grouped_fundamental_probs is not None)
    assert s is None or max(s) > 0
    if max_epsilon is None:
        if s is None:
            max_epsilon = 1
        else:
            max_epsilon = 1 / min([ss for ss in s if ss > 0])

    ranges = {}
    if derived_probs_lambda is None:
        # Handle as a separable tree
        def gamma(e):
            '''
            returns mode perturbation lambda for a given epsilon
            '''
            return lambda probs, evs, best_path, s_i:\
                find_gamma_probs(probs, e * s_i)
        count = int(max_epsilon / precision)
        epsilon = 0
        last_dec = None
        last_epsilon = None
        range_start = 0

        for step in range(count + 1):
            dec = solve_tree(
                deepcopy(tree), node_stability_type=gamma(epsilon))[1]
            dec = get_reachable(dec)
            last_epsilon = epsilon
            epsilon = min((step + 1) * precision, max_epsilon)
            if last_dec is not None and last_dec != dec:
                ranges[(range_start, last_epsilon)] = last_dec
                range_start = epsilon
            last_dec = dec
        if range_start <= max_epsilon:
            ranges[(range_start, max_epsilon)] = last_dec
    else:
        # Handle as a non separable tree
        for group in grouped_fundamental_probs:
            assert sum(group) <= 1
        param_sweep = []
        epsilon = 0.0
        while True:
            values = []
            for fp_group_ix in range(len(grouped_fundamental_probs)):
                fp_group = grouped_fundamental_probs[fp_group_ix]
                probs_l = fp_group + [1 - sum(fp_group)]
                if min([p for p in probs_l if p > 0]) == \
                        Fraction(1, sum(p > 0 for p in probs_l)):
                    # handling corner case - for all gamma values the same
                    # probabilities will be returned
                    # the above single equation works for the following both
                    # cases:
                    # a) one prob is 1 and all other are 0
                    # b) all the non zero probas are equal
                    values += [probs_l]
                else:
                    values +=\
                        [tuple(
                            find_gamma_probs(
                                probs_l,
                                epsilon * (1 if s is None
                                           else s[fp_group_ix]))[:-1])]
            param_sweep += [(epsilon, values)]
            if epsilon >= max_epsilon:
                break
            epsilon = min(epsilon + precision, max_epsilon)

        range_start = 0
        last_dec = None
        last_epsilon = None

        for param_set in param_sweep:
            dec = solve_tree(tree, derived_probs_lambda(param_set[1]))[1]
            dec = get_reachable(dec)
            epsilon = param_set[0]
            if last_dec is not None and last_dec != dec:
                ranges[(range_start, last_epsilon)] = last_dec
                range_start = epsilon
            if param_set == param_sweep[-1]:
                ranges[(range_start, max_epsilon)] = dec
            last_dec = dec
            last_epsilon = epsilon

    for key in ranges:
        ranges[key] = tree_branching_to_tuple(ranges[key], tree if use_labels
                                              else None)
    return ranges


def node_branching(node, __branching=()):
    '''
    Creates a set of all possible decisions for a given decision tree.

    *node* - root node of a decision tree
    '''
    dec_struct = dict()
    if node['type'] == 'final':
        pass
    elif node['type'] == 'decision':
        ix = 0  # index of the current node
        for nn in node['nodes']:
            dec_struct_t = node_branching(nn, __branching + (ix,))
            dec_struct.update(dec_struct_t)
            ix += 1
        dec_struct[__branching] = list(range(ix))
    elif node['type'] == 'chance':
        ix = 0  # index of the current node
        dec_struct = dict()
        for nn in node['nodes']:
            dec_struct_t = node_branching(nn, __branching + (ix,))
            dec_struct.update(dec_struct_t)
            ix += 1
    else:
        raise Exception("uknown node type for node ", str(node))
    return dec_struct


def tree_sweep(decision_dictionary):
    '''
    Transforms a *decision_dictionary* to list of dictionaries,
    with single decisions in each node.

    example:

    tree_sweep({(1,):[1,2]})

    returns [{(1,): [1]}, {(1,): [2]}]
    '''
    keys = sorted(decision_dictionary.keys())
    sweep_size = 1
    for key_ in keys:
        sweep_size *= len(decision_dictionary[key_])
    strategies = []
    for vv in range(sweep_size):
        strategy = dict()
        for key_ in keys:
            ix = vv % len(decision_dictionary[key_])
            strategy[key_] = [decision_dictionary[key_][ix]]
            vv = (vv - ix) // len(decision_dictionary[key_])
        strat_r = get_reachable(strategy)
        if strat_r not in strategies:
            strategies += [strat_r]
    return strategies


def tree_decision_paths_to_tuple(tree_sweep_decision):
    '''
    Converts a decision dictionary to a list decision.

    example:

    tree_decision_paths_to_tuple({():[1],(1,): [2],(1,2):[3]})

    returns ((1, 2, 3),)

    '''
    paths = [key + tuple(tree_sweep_decision[key]) for
             key in tree_sweep_decision.keys()]
    remove = []
    for e in paths:
        for e2 in paths:
            if len(e2) > len(e) and e2[0:len(e)] == e:
                remove += [e]
                break
    for r in remove:
        paths.remove(r)
    return tuple(paths)


def tree_branching_to_tuple(tree_branching, use_names_tree=None):
    '''
    Converts a full decision dictionary to a tuple of tuples.

    *tree_branching* - list of dictionaries generated with tree_sweep

    *use_names_tree* - use labels from decision tree \
    rather than indices to represent nodes

    example
    tree_branching_to_tuple({():[1],(1,): [2],(1,2):[3,4]})
    yields
    (((1, 2, 3),), ((1, 2, 4),))
    '''
    return tuple([tree_decision_paths_to_tuple(decision) if use_names_tree
                  is None else
                  get_decision_name(
                      use_names_tree,
                      tree_decision_paths_to_tuple(decision))
                  for decision in tree_sweep(tree_branching)])


def get_decision_name(tree, path_tuple):
    '''
    Transforms a given *path_tuple* to a tuple of node labels from
    a given decision *tree*
    '''
    assert isinstance(path_tuple, tuple)
    if len(path_tuple) == 0:
        return tuple([])
    if isinstance(path_tuple[0], tuple):
        return tuple([get_decision_name(tree, d) for d in path_tuple])
    dec_name = tuple([tree["nodes"][path_tuple[0]]["label"] if
                      "label" in tree["nodes"][path_tuple[0]]
                      else path_tuple[0]])
    if len(path_tuple) == 1:
        return dec_name
    else:
        return dec_name +\
            get_decision_name(tree["nodes"][path_tuple[0]], path_tuple[1:])


