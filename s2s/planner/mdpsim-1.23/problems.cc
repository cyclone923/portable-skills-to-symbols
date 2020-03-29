/*
 * Copyright (C) 2003 Carnegie Mellon University and Rutgers University
 *
 * Permission is hereby granted to distribute this software for
 * non-commercial research purposes, provided that this copyright
 * notice is included with any such distribution.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE
 * SOFTWARE IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU
 * ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.
 *
 * $Id: problems.cc,v 1.3 2004/04/12 23:53:34 hakan Exp $
 */
#include "problems.h"
#include "domains.h"
#include "exceptions.h"
#include <typeinfo>


/* ====================================================================== */
/* Problem */

/* Table of defined problems. */
Problem::ProblemMap Problem::problems = Problem::ProblemMap();


/* Returns a const_iterator pointing to the first problem. */
Problem::ProblemMap::const_iterator Problem::begin() {
  return problems.begin();
}


/* Returns a const_iterator pointing beyond the last problem. */
Problem::ProblemMap::const_iterator Problem::end() {
  return problems.end();
}


/* Returns the problem with the given name, or NULL if it is undefined. */
const Problem* Problem::find(const std::string& name) {
  ProblemMap::const_iterator pi = problems.find(name);
  return (pi != problems.end()) ? (*pi).second : NULL;
}


/* Removes all defined problems. */
void Problem::clear() {
  ProblemMap::const_iterator pi = begin();
  while (pi != end()) {
    delete (*pi).second;
    pi = begin();
  }
  problems.clear();
}


/* Constructs a problem. */
Problem::Problem(const std::string& name, const Domain& domain)
  : name_(name), domain_(&domain), terms_(TermTable(domain.terms())),
    goal_(&StateFormula::FALSE), goal_reward_(NULL), metric_(new Value(0)) {
  StateFormula::register_use(goal_);
  Expression::register_use(metric_);
  const Problem* p = find(name);
  if (p != NULL) {
    delete p;
  }
  problems[name] = this;
}


/* Deletes a problem. */
Problem::~Problem() {
  problems.erase(name());
  for (AtomSet::const_iterator ai = init_atoms_.begin();
       ai != init_atoms_.end(); ai++) {
    StateFormula::unregister_use(*ai);
  }
  for (ValueMap::const_iterator vi = init_values_.begin();
       vi != init_values_.end(); vi++) {
    Expression::unregister_use((*vi).first);
  }
  for (EffectList::const_iterator ei = init_effects_.begin();
       ei != init_effects_.end(); ei++) {
    delete *ei;
  }
  StateFormula::unregister_use(goal_);
  if (goal_reward_ != NULL) {
    delete goal_reward_;
  }
  Expression::unregister_use(metric_);
}


/* Adds an atomic state formula to the initial conditions of this
   problem. */
void Problem::add_init_atom(const Atom& atom) {
  if (init_atoms_.find(&atom) == init_atoms_.end()) {
    init_atoms_.insert(&atom);
    StateFormula::register_use(&atom);
  }
}


/* Adds a function application value to the initial conditions of
   this problem. */
void Problem::add_init_value(const Application& application,
			     const Rational& value) {
  if (init_values_.find(&application) == init_values_.end()) {
    init_values_.insert(std::make_pair(&application, value));
    Expression::register_use(&application);
  } else {
    init_values_[&application] = value;
  }
}


/* Adds an initial effect for this problem. */
void Problem::add_init_effect(const Effect& effect) {
  init_effects_.push_back(&effect);
}


/* Sets the goal for this problem. */
void Problem::set_goal(const StateFormula& goal) {
  if (&goal != goal_) {
    const StateFormula* tmp = goal_;
    goal_ = &goal;
    StateFormula::register_use(goal_);
    StateFormula::unregister_use(tmp);
  }
}


/* Sets the goal reward for this problem. */
void Problem::set_goal_reward(const Assignment& goal_reward) {
  if (&goal_reward != goal_reward_) {
    delete goal_reward_;
    goal_reward_ = &goal_reward;
  }
}


/* Sets the metric to maximize for this problem. */
void Problem::set_metric(const Expression& metric, bool negate) {
  const Expression* real_metric;
  if (negate) {
    real_metric = new Subtraction(*new Value(0), metric);
  } else {
    real_metric = &metric;
  }
  if (real_metric != metric_) {
    const Expression* tmp = metric_;
    metric_ = real_metric;
    Expression::register_use(metric_);
    Expression::unregister_use(tmp);
  }
}


/* Instantiates all actions. */
void Problem::instantiate_actions() {
  set_goal(goal().instantiation(SubstitutionMap(), *this));
  if (goal_reward() != NULL) {
    set_goal_reward(goal_reward()->instantiation(SubstitutionMap(), *this));
  }
  set_metric(metric().instantiation(SubstitutionMap(), *this));
  domain().instantiated_actions(actions_, *this);
}


/* Tests if the metric is constant. */
bool Problem::constant_metric() const {
  return typeid(*metric_) == typeid(Value);
}


/* Fills the provided object list with objects (including constants
   declared in the domain) that are compatible with the given
   type. */
void Problem::compatible_objects(ObjectList& objects, Type type) const {
  domain().compatible_constants(objects, type);
  Object last = terms().last_object();
  for (Object i = terms().first_object(); i <= last; i++) {
    if (domain().types().subtype(terms().type(i), type)) {
      objects.push_back(i);
    }
  }
}


/* Fills the given list with actions enabled in the given state. */
void Problem::enabled_actions(ActionList& actions, const AtomSet& atoms,
			      const ValueMap& values) const {
  for (ActionList::const_iterator ai = actions_.begin();
       ai != actions_.end(); ai++) {
    if ((*ai)->enabled(atoms, values)) {
      actions.push_back(*ai);
    }
  }
}


/* Output operator for problems. */
std::ostream& operator<<(std::ostream& os, const Problem& p) {
  os << "name: " << p.name();
  os << std::endl << "domain: " << p.domain().name();
  os << std::endl << "objects:";
  for (Object i = p.terms().first_object();
       i <= p.terms().last_object(); i++) {
    os << std::endl << "  ";
    p.terms().print_term(os, i);
    os << " - ";
    p.domain().types().print_type(os, p.terms().type(i));
  }
  os << std::endl << "init:";
  for (AtomSet::const_iterator ai = p.init_atoms_.begin();
       ai != p.init_atoms_.end(); ai++) {
    os << std::endl << "  ";
    (*ai)->print(os, p.domain().predicates(), p.domain().functions(),
		 p.terms());
  }
  for (ValueMap::const_iterator vi = p.init_values_.begin();
       vi != p.init_values_.end(); vi++) {
    os << std::endl << "  (= ";
    (*vi).first->print(os, p.domain().functions(), p.terms());
    os << ' ' << (*vi).second << ")";
  }
  for (EffectList::const_iterator ei = p.init_effects_.begin();
       ei != p.init_effects_.end(); ei++) {
    os << std::endl << "  ";
    (*ei)->print(os, p.domain().predicates(), p.domain().functions(),
		 p.terms());
  }
  os << std::endl << "goal: ";
  p.goal().print(os, p.domain().predicates(), p.domain().functions(),
		 p.terms());
  if (p.goal_reward() != NULL) {
    os << std::endl << "goal reward: ";
    p.goal_reward()->expression().print(os, p.domain().functions(), p.terms());
  }
  os << std::endl << "metric: ";
  p.metric().print(os, p.domain().functions(), p.terms());
  os << std::endl << "actions:";
  for (ActionList::const_iterator ai = p.actions_.begin();
       ai != p.actions_.end(); ai++) {
    os << std::endl << "  ";
    (*ai)->print(os, p.terms());
  }
  return os;
}
