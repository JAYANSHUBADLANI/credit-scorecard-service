"""Tests for the startup check between the published contract and the fitted artifact.

These exist because the first version of the check was wrong in a way that only showed up
under a refit. It compared the declared levels against the fitted levels for exact equality
in both directions, which meant a level that merely fell below the categorical population
floor took the whole service down at boot. On this data that is not hypothetical: two of the
declared levels rest on two rows in 215,000, and under a random split the API failed to start
on roughly one refit in twelve.

The asymmetry is the point, so it is tested in both directions rather than described.

Stubs are used rather than the real artifact so these run without a fitted model. The check
only ever reads two things, the fitted level list and the bins each level resolves to, and a
stub states exactly which case is under test.
"""

from __future__ import annotations

import pytest

from src.api import verify_contract
from src.binning import OTHER_LABEL
from src.schemas import CATEGORICAL_LEVELS


class _Binning:
    def __init__(self, resolvable):
        # Every fit has a catch all, and it is never a level a caller can name.
        self.level_to_index = {level: i for i, level in enumerate(resolvable)}
        self.level_to_index[OTHER_LABEL] = len(resolvable)


class _Transformer:
    def __init__(self, bins):
        self.bins = bins


class _Artifact:
    def __init__(self, categorical_levels, bins):
        self.categorical_levels = categorical_levels
        self.transformer = _Transformer(bins)


class _Scoring:
    def __init__(self, artifact):
        self.artifact = artifact


def build_scoring(fitted=None, resolvable=None) -> _Scoring:
    """A stub artifact that agrees with the contract except where a test says otherwise.

    `fitted` overrides the levels the model was fitted on. `resolvable` overrides the levels
    that have a bin of their own, everything else falling to the catch all.
    """
    fitted = fitted or {}
    resolvable = resolvable or {}
    levels = {
        field: list(fitted.get(field, declared))
        for field, declared in CATEGORICAL_LEVELS.items()
    }
    bins = {
        field: _Binning(resolvable.get(field, levels[field]))
        for field in CATEGORICAL_LEVELS
    }
    return _Scoring(_Artifact(levels, bins))


# The contract and the fit agree ----------------------------------------------------------

def test_a_matching_contract_starts_cleanly():
    verify_contract(build_scoring())


# A level the model knows and the contract does not: refuse to start -----------------------

def test_a_fitted_level_the_contract_does_not_accept_is_fatal():
    """The card was built to score it and the API would answer 422. That is real business
    refused, and nothing in the rejection would explain why."""
    scoring = build_scoring(
        fitted={"NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans", "Overdraft"]}
    )
    with pytest.raises(RuntimeError) as error:
        verify_contract(scoring)
    assert "Overdraft" in str(error.value)
    assert "NAME_CONTRACT_TYPE" in str(error.value)


def test_a_characteristic_missing_from_the_fit_entirely_is_fatal():
    scoring = build_scoring()
    del scoring.artifact.categorical_levels["NAME_EDUCATION_TYPE"]
    with pytest.raises(RuntimeError, match="NAME_EDUCATION_TYPE"):
        verify_contract(scoring)


# A level the contract accepts and the fit cannot resolve: start anyway ---------------------

def test_a_declared_level_below_the_population_floor_does_not_block_startup():
    """The regression this file exists for.

    `fit_categorical` folds every level below `min_categorical_fraction` into the catch all,
    so a level with no bin of its own scores exactly as it did before. Refusing to boot over
    one is an outage in exchange for a distinction the card never drew.
    """
    scoring = build_scoring(
        resolvable={"NAME_INCOME_TYPE": ["Working", "Pensioner", "Commercial associate"]}
    )
    verify_contract(scoring)  # must not raise


def test_a_declared_level_absent_from_the_fit_does_not_block_startup():
    """A refit whose training slice happened to miss a rare level still serves.

    The level is still accepted, and the transformer routes it through the catch all the same
    way it routes any level it has not seen.
    """
    kept = [level for level in CATEGORICAL_LEVELS["NAME_INCOME_TYPE"] if level != "Student"]
    scoring = build_scoring(
        fitted={"NAME_INCOME_TYPE": kept}, resolvable={"NAME_INCOME_TYPE": kept}
    )
    verify_contract(scoring)  # must not raise


def test_unresolved_levels_are_reported_on_stderr_rather_than_silently(capsys):
    """Not blocking is not the same as not saying. A reviewer should be able to see which
    accepted levels the card cannot tell apart."""
    scoring = build_scoring(resolvable={"NAME_INCOME_TYPE": ["Working", "Pensioner"]})
    verify_contract(scoring)

    reported = capsys.readouterr().err
    assert "NAME_INCOME_TYPE" in reported
    assert "Businessman" in reported
    assert "catch all" in reported


# Both directions at once -------------------------------------------------------------------

def test_an_undeclared_level_is_still_fatal_alongside_an_unresolved_one():
    """The permissive direction must not make the strict one permissive too."""
    scoring = build_scoring(
        fitted={"NAME_EDUCATION_TYPE": CATEGORICAL_LEVELS["NAME_EDUCATION_TYPE"] + ["Doctorate"]},
        resolvable={"NAME_INCOME_TYPE": ["Working"]},
    )
    with pytest.raises(RuntimeError, match="Doctorate"):
        verify_contract(scoring)


# Gender is gone from the contract entirely --------------------------------------------------

def test_gender_is_not_part_of_the_request_contract():
    """A prohibited basis is excluded at the boundary, so there is no level list to check."""
    assert "CODE_GENDER" not in CATEGORICAL_LEVELS
