/* -*-C++-*- */
/*
 * Effects.
 *
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
 * $Id: effects.h,v 1.3 2004/04/12 23:53:32 hakan Exp $
 */
#ifndef EFFECTS_H
#define EFFECTS_H

#include <config.h>
#include "terms.h"
#include <iostream>
#include <vector>

struct Rational;
struct PredicateTable;
struct FunctionTable;
struct ValueMap;
struct Expression;
struct Application;
struct StateFormula;
struct Atom;
struct AtomSet;
struct AtomList;
struct Problem;


/* ====================================================================== */
/* Assignment */

/*
 * An assignment.
 */
struct Assignment {
  /* Assignment operators. */
  typedef enum { ASSIGN_OP, SCALE_UP_OP, SCALE_DOWN_OP,
		 INCREASE_OP, DECREASE_OP } AssignOp;

  /* Constructs an assignment. */
  Assignment(AssignOp oper, const Application& application,
	     const Expression& expr);

  /* Deletes this assignment. */
  ~Assignment();

  /* Returns the application of this assignment. */
  const Application& application() const { return *application_; }

  /* Returns the expression of this assignment. */
  const Expression& expression() const { return *expr_; }

  /* Changes the given state according to this assignment. */
  void affect(ValueMap& values) const;

  /* Returns an instantiaion of this assignment. */
  const Assignment& instantiation(const SubstitutionMap& subst,
				  const Problem& problem) const;

  /* Prints this object on the given stream. */
  void print(std::ostream& os, const FunctionTable& functions,
	     const TermTable& terms) const;

private:
  /* Assignment operator. */
  AssignOp operator_;
  /* Application affected by this assignment. */
  const Application* application_;
  /* Expression. */
  const Expression* expr_;
};


/* ====================================================================== */
/* AssignmentList */

/*
 * List of assignments.
 */
struct AssignmentList : public std::vector<const Assignment*> {
};


/* ====================================================================== */
/* Effect */

/*
 * An effect.
 */
struct Effect {
  /* Register use of the given effect. */
  static void register_use(const Effect* e) {
    if (e != NULL) {
      e->ref_count_++;
    }
  }

  /* Unregister use of the given effect. */
  static void unregister_use(const Effect* e) {
    if (e != NULL) {
      e->ref_count_--;
      if (e->ref_count_ == 0) {
	delete e;
      }
    }
  }

  /* Deletes this effect. */
  virtual ~Effect() {}

  /* Fills the provided lists with a sampled state change for this
     effect in the given state. */
  virtual void state_change(AtomList& adds, AtomList& deletes,
			    AssignmentList& assignments,
			    const AtomSet& atoms,
			    const ValueMap& values) const = 0;

  /* Returns an instantiation of this effect. */
  virtual const Effect& instantiation(const SubstitutionMap& subst,
				      const Problem& problem) const = 0;

  /* Prints this object on the given stream. */
  virtual void print(std::ostream& os, const PredicateTable& predicates,
		     const FunctionTable& functions,
		     const TermTable& terms) const = 0;

protected:
  /* Constructs an effect. */
  Effect() : ref_count_(0) {}

private:
  /* Reference counter. */
  mutable size_t ref_count_;
};


/* ====================================================================== */
/* EffectList */

/*
 * List of effects.
 */
struct EffectList : public std::vector<const Effect*> {
};


/* ====================================================================== */
/* SimpleEffect */

/*
 * A simple effect.
 */
struct SimpleEffect : public Effect {
  /* Deletes this simple effect. */
  virtual ~SimpleEffect();

  /* Returns the atom associated with this simple effect. */
  const Atom& atom() const { return *atom_; }

protected:
  /* Constructs a simple effect. */
  explicit SimpleEffect(const Atom& atom);

private:
  /* Atom added by this effect. */
  const Atom* atom_;
};


/* ====================================================================== */
/* AddEffect */

/*
 * An add effect.
 */
struct AddEffect : public SimpleEffect {
  /* Constructs an add effect. */
  explicit AddEffect(const Atom& atom) : SimpleEffect(atom) {}

  /* Fills the provided lists with a sampled state change for this
     effect in the given state. */
  virtual void state_change(AtomList& adds, AtomList& deletes,
			    AssignmentList& assignments,
			    const AtomSet& atoms,
			    const ValueMap& values) const;

  /* Returns an instantiation of this effect. */
  virtual const Effect& instantiation(const SubstitutionMap& subst,
				      const Problem& problem) const;

  /* Prints this object on the given stream. */
  virtual void print(std::ostream& os, const PredicateTable& predicates,
		     const FunctionTable& functions,
		     const TermTable& terms) const;
};


/* ====================================================================== */
/* DeleteEffect */

/*
 * A delete effect.
 */
struct DeleteEffect : public SimpleEffect {
  /* Constructs a delete effect. */
  explicit DeleteEffect(const Atom& atom) : SimpleEffect(atom) {}

  /* Fills the provided lists with a sampled state change for this
     effect in the given state. */
  virtual void state_change(AtomList& adds, AtomList& deletes,
			    AssignmentList& assignments,
			    const AtomSet& atoms,
			    const ValueMap& values) const;

  /* Returns an instantiation of this effect. */
  virtual const Effect& instantiation(const SubstitutionMap& subst,
				      const Problem& problem) const;

  /* Prints this object on the given stream. */
  virtual void print(std::ostream& os, const PredicateTable& predicates,
		     const FunctionTable& functions,
		     const TermTable& terms) const;
};


/* ====================================================================== */
/* AssignmentEffect */

/*
 * An assignment effect.
 */
struct AssignmentEffect : public Effect {
  /* Constructs an assignment effect. */
  AssignmentEffect(const Assignment& assignment);

  /* Deletes this assignment effect. */
  virtual ~AssignmentEffect();

  /* Returns the assignment performed by this effect. */
  const Assignment& assignment() const { return *assignment_; }

  /* Fills the provided lists with a sampled state change for this
     effect in the given state. */
  virtual void state_change(AtomList& adds, AtomList& deletes,
			    AssignmentList& assignments,
			    const AtomSet& atoms,
			    const ValueMap& values) const;

  /* Returns an instantiation of this effect. */
  virtual const Effect& instantiation(const SubstitutionMap& subst,
				      const Problem& problem) const;

  /* Prints this object on the given stream. */
  virtual void print(std::ostream& os, const PredicateTable& predicates,
		     const FunctionTable& functions,
		     const TermTable& terms) const;

private:
  /* Assignment performed by this effect. */
  const Assignment* assignment_;
};


/* ====================================================================== */
/* ConjunctiveEffect */

/*
 * A conjunctive effect.
 */
struct ConjunctiveEffect : public Effect {
  /* Deletes this conjunctive effect. */
  virtual ~ConjunctiveEffect();

  /* Adds a conjunct to this conjunctive effect. */
  void add_conjunct(const Effect& conjunct);

  /* Returns the number of conjuncts of this conjunctive effect. */
  size_t size() const { return conjuncts_.size(); }

  /* Returns the ith conjunct of this conjunctive effect. */
  const Effect& conjunct(size_t i) const { return *conjuncts_[i]; }

  /* Fills the provided lists with a sampled state change for this
     effect in the given state. */
  virtual void state_change(AtomList& adds, AtomList& deletes,
			    AssignmentList& assignments,
			    const AtomSet& atoms,
			    const ValueMap& values) const;

  /* Returns an instantiation of this effect. */
  virtual const Effect& instantiation(const SubstitutionMap& subst,
				      const Problem& problem) const;

  /* Prints this object on the given stream. */
  virtual void print(std::ostream& os, const PredicateTable& predicates,
		     const FunctionTable& functions,
		     const TermTable& terms) const;

private:
  /* The conjuncts. */
  EffectList conjuncts_;
};


/* ====================================================================== */
/* ConditionalEffect */

/*
 * A conditional effect.
 */
struct ConditionalEffect : public Effect {
  /* Returns a conditional effect. */
  static const Effect& make(const StateFormula& condition,
			    const Effect& effect);

  /* Deletes this conditional effect. */
  virtual ~ConditionalEffect();

  /* Returns the condition of this effect. */
  const StateFormula& condition() const { return *condition_; }

  /* Returns the conditional effect of this effect. */
  const Effect& effect() const { return *effect_; }

  /* Fills the provided lists with a sampled state change for this
     effect in the given state. */
  virtual void state_change(AtomList& adds, AtomList& deletes,
			    AssignmentList& assignments,
			    const AtomSet& atoms,
			    const ValueMap& values) const;

  /* Returns an instantiation of this effect. */
  virtual const Effect& instantiation(const SubstitutionMap& subst,
				      const Problem& problem) const;

  /* Prints this object on the given stream. */
  virtual void print(std::ostream& os, const PredicateTable& predicates,
		     const FunctionTable& functions,
		     const TermTable& terms) const;

private:
  /* Effect condition. */
  const StateFormula* condition_;
  /* Effect. */
  const Effect* effect_;

  /* Constructs a conditional effect. */
  ConditionalEffect(const StateFormula& condition, const Effect& effect);
};


/* ====================================================================== */
/* ProbabilisticEffect */

/*
 * A probabilistic effect.
 */
struct ProbabilisticEffect : public Effect {
  /* Constructs an empty probabilistic effect. */
  ProbabilisticEffect() : weight_sum_(0) {}

  /* Deletes this probabilistic effect. */
  virtual ~ProbabilisticEffect();

  /* Adds an outcome to this probabilistic effect. */
  bool add_outcome(const Rational& p, const Effect& effect);

  /* Returns the number of outcomes of this probabilistic effect. */
  size_t size() const { return weights_.size(); }

  /* Returns the ith outcome's probability. */
  Rational probability(size_t i) const;

  /* Returns the ith outcome's effect. */
  const Effect& effect(size_t i) const { return *effects_[i]; }

  /* Fills the provided lists with a sampled state change for this
     effect in the given state. */
  virtual void state_change(AtomList& adds, AtomList& deletes,
			    AssignmentList& assignments,
			    const AtomSet& atoms,
			    const ValueMap& values) const;

  /* Returns an instantiation of this effect. */
  virtual const Effect& instantiation(const SubstitutionMap& subst,
				      const Problem& problem) const;

  /* Prints this object on the given stream. */
  virtual void print(std::ostream& os, const PredicateTable& predicates,
		     const FunctionTable& functions,
		     const TermTable& terms) const;

private:
  /* Weights associated with outcomes. */
  std::vector<int> weights_;
  /* The sum of weights. */
  int weight_sum_;
  /* Outcome effects. */
  EffectList effects_;
};


/* ====================================================================== */
/* QuantifiedEffect */

/*
 * A universally quantified effect.
 */
struct QuantifiedEffect : public Effect {
  /* Constructs a universally quantified effect. */
  explicit QuantifiedEffect(const Effect& effect);

  /* Deletes this universally quantifed effect. */
  virtual ~QuantifiedEffect();

  /* Adds a quantified variable to this universally quantified effect. */
  void add_parameter(Variable parameter) { parameters_.push_back(parameter); }

  /* Returns the number quantified variables of this universally
     quantified effect. */
  size_t arity() const { return parameters_.size(); }

  /* Returns the ith parameter of this universally quantified effect. */
  Variable parameter(size_t i) const { return parameters_[i]; }

  /* Returns the quantified effect. */
  const Effect& effect() const { return *effect_; }

  /* Fills the provided lists with a sampled state change for this
     effect in the given state. */
  virtual void state_change(AtomList& adds, AtomList& deletes,
			    AssignmentList& assignments,
			    const AtomSet& atoms,
			    const ValueMap& values) const;

  /* Returns an instantiation of this effect. */
  virtual const Effect& instantiation(const SubstitutionMap& subst,
				      const Problem& problem) const;

  /* Prints this object on the given stream. */
  virtual void print(std::ostream& os, const PredicateTable& predicates,
		     const FunctionTable& functions,
		     const TermTable& terms) const;

private:
  /* Quantified variables. */
  VariableList parameters_;
  /* The quantified effect. */
  const Effect* effect_;
};


#endif /* EFFECTS_H */
