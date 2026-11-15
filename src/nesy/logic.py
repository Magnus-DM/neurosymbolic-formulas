# Provided by the course Learning and Reasoning (KU Leuven, 2025-2026). Unchanged.

from nesy.term import Term, Clause
from nesy.parser import parse_term, parse_program
from abc import ABC, abstractmethod
from collections import namedtuple
from nesy.term import Variable, Fact
from typing import List
from collections import defaultdict
import itertools


class LogicEngine(ABC):

    @abstractmethod
    def reason(self, program: list[Clause], queries: list[Term]):
        pass


def unify_var(var, val, subst):
    if var in subst:
        return unify(subst[var], val, subst)
    elif val in subst:
        return unify(var, subst[val], subst)
    elif isinstance(val, Term) and val in val.arguments: # TODO this should be enough for datalog
        return False
    else:
        subst[var] = val
        return subst

def unify(sym1, sym2, subst=None):

    if subst is None:
        subst = {}
    if isinstance(sym1, Fact):
        sym1 = sym1.term
    if isinstance(sym2, Fact):
        sym2 = sym2.term


    if subst is False:
        return False
    elif sym1 == sym2:
        return subst
    #variable cases
    elif isinstance(sym1, Variable):
        return unify_var(sym1, sym2, subst)
    elif isinstance(sym2, Variable):
        return unify_var(sym2, sym1, subst)
    elif isinstance(sym1, Term) and isinstance(sym2, Term):
        return unify(sym1.arguments, sym2.arguments, unify(sym1.functor, sym2.functor, subst))
    elif (isinstance(sym1, tuple) or isinstance(sym1, list)) and (isinstance(sym2, tuple) or isinstance(sym2, list)):
        if len(sym1) != len(sym2):
            return False
        elif len(sym1) == 0 and len(sym2) == 0:
            return subst
        return unify(sym1[1:],sym2[1:], unify(sym1[0], sym2[0], subst))
    else:
        return False

def subst(s, x):
    """Substitute the substitution s into the term x.
    """

    if isinstance(x, Variable):
        return s.get(x, x)
    elif isinstance(x, Term):
        if len(x.arguments) == 0:
            return x
        else:
            return Term(functor=x.functor, arguments=tuple([subst(s, arg) for arg in x.arguments]))
    elif isinstance(x, Fact):
        return Fact(term=subst(s, x.term), weight=subst(s, x.weight))
    elif isinstance(x, list):
        return [subst(s, xi) for xi in x]
    elif isinstance(x, tuple):
        return tuple([subst(s, xi) for xi in x])
    else:
        raise Exception("Invalid input x for subst(s,x): {}".format(x))


def constants_in_clause(c: Clause|Fact):
    if isinstance(c, Fact):
        if c.weight is not None:
            terms = (c.weight, c.term)
        else:
            terms = (c.term,)
    else:
        terms = (c.head,) + c.body
    res = []
    for term in terms:
        if isinstance(term, int):
            res.append(int)
        else:
            for arg in term.arguments:
                if not isinstance(arg, Variable):
                    res.append(arg)
    return res



def forward_chaining(program, alpha=None):
    """Infer a conclusion from a knowledge base using forward chaining."""

    kb_consts = list({c for clause in program for c in constants_in_clause(clause)})
    known_terms = {f.term for f in program if isinstance(f, Fact)}
    answers = []
    ground_clauses = defaultdict(set)

    def enum_subst(p: List[Term]):
        query_vars = list({v for term in p for v in term.arguments if isinstance(v, Variable)})
        for assignment_list in itertools.product(kb_consts, repeat=len(query_vars)):
            theta = {x: y for x, y in zip(query_vars, assignment_list)}
            yield theta

    if alpha is not None:
        for c in program:
            phi = unify(c, alpha)
            if phi:
                answers.append(phi)

    while True:
        new = []
        for c in program:
            if isinstance(c, Fact):
                ground_clauses[c.term] = c.weight
            else: # isinstance(c, Clause):
                head, body = c
                for theta in enum_subst(body):
                    body_ = subst(theta, body)
                    if set(body_).issubset(known_terms):
                        head_ = subst(theta, head)
                        ground_clauses[head_].add(body_)
                        if head_ not in known_terms.union(new):
                            new.append(head_)
                            if alpha is not None:
                                phi = unify(head_, alpha)
                                if phi:
                                    answers.append(phi)
        if not new:
            break
        known_terms.update(new)
    if alpha is not None:
        return answers, ground_clauses
    else:
        return ground_clauses


class ForwardChaining(LogicEngine):

    def reason(self, program: list[Clause], queries: list[Term]):
        del queries # Forward Chaining is not query oriented
        ground_clauses = forward_chaining(program)
        return ground_clauses



if __name__ == '__main__':

    # program = """
    # ancestor(terach, abraham)."""

    # program = """
    # ancestor(X, Y) :- parent(X, Y).
    # parent(terach, abraham).
    # """


    program = """
        ancestor(X, Y) :- parent(X, Y).
        ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
        parent(terach, abraham).
        parent(abraham, isaac).
        parent(isaac, jacob).
        parent(jacob, joseph).
        parent(jacob, esau).
        parent(joseph, ephraim).
        parent(lorath, olimeh)."""

    query = "ancestor(terach, X)."

    program = parse_program(program)
    query = [a for a in parse_program(query)][0]
    answers, ground_clauses = forward_chaining(program, query)
    print(answers)
    for k, v in ground_clauses.items():
        print(k, v)
