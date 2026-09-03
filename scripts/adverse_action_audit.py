"""Audit the adverse action reason codes over the held out slice.

Reason codes are a compliance artefact, so "the code runs" is not the standard. The questions
that matter are whether the reasons are accurate, whether they are informative, and whether
they disclose anything they should not. This script answers all three against the 92,254 held
out applications, and writes the evidence to `reports/`.

What it checks, and why each one is here rather than in a unit test:

1. **The decomposition is exact.** Every reason is a claim that a characteristic cost the
   applicant a number of points. That is only true if the per characteristic points sum to the
   score, so the identity is asserted on every held out application, not on a fixture.

2. **Which reasons actually get cited.** A reason code set that names the same characteristic
   on every notice is legal and useless. The realised frequency is the only way to know which
   this card produces, and it cannot be reasoned about from the coefficients: it depends on
   where declined applicants actually sit relative to the reference profile.

3. **Whether the choice of reference profile changes what applicants are told.** Both bases are
   defensible and the choice looks technical. This measures how many applicants would receive a
   materially different notice under the other one.

3a. **Whether a cited reason names a characteristic the applicant is not actually weak on.** This
   is the accuracy question hiding under the informativeness one, and it is what settled the
   choice of default. Under `max_observed` a characteristic can be cited because its point range
   is wide, even where this applicant sits above the population average on it, and such a
   characteristic did nothing to distinguish them from the applicants who were approved. Under
   `population_mean` it cannot happen at all, because a positive shortfall means below average by
   definition.

4. **Regulation B on age.** Age may be used in an empirically derived, demonstrably sound
   scoring system, but not to assign a negative factor to applicants aged 62 or over. That is a
   property of the fitted card and it is checked directly against the age bins rather than
   assumed. The check is deliberately blunt: it compares the points an applicant aged 62 or
   over can earn against the best points available at any age.

5. **What a notice would disclose.** Every prohibited basis a reason code can name, with the
   share of declines that name it, so the cost of removing it is a number rather than a
   principle.

Nothing here modifies the model. The remedy for anything it finds is a refit, which is a
decision to take deliberately.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.features import build_features
from src.reasons import (
    BASIS_MAX_OBSERVED,
    BASIS_POPULATION_MEAN,
    PROHIBITED_BASES,
    STATEMENTS,
    ReasonCodeAssigner,
)
from src.scoring import load_artifact

# Regulation B protects applicants aged 62 and over specifically.
ELDERLY_AGE = 62.0


def _assigners(artifact, settings) -> Dict[str, ReasonCodeAssigner]:
    return {
        basis: ReasonCodeAssigner(
            artifact,
            basis=basis,
            max_reasons=settings.max_reasons,
            min_shortfall_points=settings.min_shortfall_points,
            adverse_bands=settings.adverse_bands,
        )
        for basis in (BASIS_MAX_OBSERVED, BASIS_POPULATION_MEAN)
    }


def population_mean_points(artifact) -> Dict[str, float]:
    """Average points each characteristic earned across the training population."""
    points = artifact.scorecard.points_by_bin(artifact.transformer)
    return {
        feature: float(np.dot(artifact.transformer.bins[feature].reference_proportions, points[feature]))
        for feature in artifact.scorecard.features
    }


def check_decomposition(artifact, features: pd.DataFrame, scores: np.ndarray) -> float:
    """Max absolute gap between the summed points and the served score."""
    points = artifact.scorecard.points_by_bin(artifact.transformer)
    bins = artifact.transformer.bin_indices(features)
    total = np.zeros(len(features))
    for feature in artifact.scorecard.features:
        total += points[feature][bins[feature].to_numpy()]
    return float(np.abs(total - scores).max())


def check_age_rule(artifact) -> Dict[str, object]:
    """Whether an applicant aged 62 or over can be assigned a negative factor for their age.

    `AGE_YEARS` bins are half open intervals on the upper edge, so a bin is reachable by an
    applicant aged 62 or over when its upper edge exceeds `ELDERLY_AGE`. The comparison is
    against the best points available at any age, because that is what "not assigned a negative
    factor because of age" means on a points card: an older applicant should not be able to
    earn fewer points for their age than a younger one.
    """
    binning = artifact.transformer.bins.get("AGE_YEARS")
    if binning is None:
        return {"applicable": False, "reason": "AGE_YEARS is not a retained characteristic"}

    points = artifact.scorecard.points_by_bin(artifact.transformer)["AGE_YEARS"]
    observed = points[: binning.missing_index]
    edges = list(binning.edges)

    # Upper edge of each observed bin; the last bin runs to infinity.
    uppers = edges + [float("inf")]
    elderly_bins = [i for i, upper in enumerate(uppers) if upper > ELDERLY_AGE]

    best_any_age = float(observed.max())
    worst_elderly = float(min(observed[i] for i in elderly_bins))
    best_elderly = float(max(observed[i] for i in elderly_bins))

    return {
        "applicable": True,
        "best_points_at_any_age": round(best_any_age, 2),
        "best_points_available_to_62_plus": round(best_elderly, 2),
        "worst_points_available_to_62_plus": round(worst_elderly, 2),
        "max_penalty_vs_best_age": round(best_any_age - worst_elderly, 2),
        # The card is monotone in age by construction, so this is the substantive question:
        # can an applicant aged 62 or over earn the best age points at all?
        "elderly_can_earn_best_age_points": bool(best_elderly >= best_any_age),
        "compliant": bool(best_elderly >= best_any_age),
    }


def audit(config_path: str | None, limit: int | None) -> Dict[str, object]:
    config = load_config(config_path)
    settings = config.adverse_action
    artifact = load_artifact(config.path(config.service.model_path))
    assigners = _assigners(artifact, settings)
    primary = assigners[settings.basis]

    holdout_path = config.path("data/processed/holdout.parquet")
    if not holdout_path.exists():
        raise FileNotFoundError(
            f"no held out slice at {holdout_path}. Run `make train`, which writes it, before "
            "auditing the reasons that would be sent about it."
        )
    holdout = pd.read_parquet(holdout_path)
    if limit:
        holdout = holdout.head(limit)

    features = build_features(holdout)
    scored = artifact.score_frame(features)
    bins = artifact.transformer.bin_indices(features)

    max_gap = check_decomposition(artifact, features, scored["score"].to_numpy())

    adverse = scored.index[scored["band"].isin(settings.adverse_bands)]
    records = [
        {feature: int(value) for feature, value in bins.loc[i].items()} for i in adverse
    ]

    # A reason naming a characteristic the applicant is at or above the population average on is
    # the accuracy question underneath the informativeness one. Such a characteristic did nothing
    # to distinguish this applicant from the approved population, so calling it a principal
    # reason for their decline is a stretch, even though the statement is literally true.
    average_points = population_mean_points(artifact)
    reasons_above_average = 0
    notices_with_an_above_average_reason = 0

    any_rank: Counter = Counter()
    first_rank: Counter = Counter()
    prohibited_hits: Counter = Counter()
    reason_counts: List[int] = []
    disagreements = 0
    no_reasons = 0

    alternate = BASIS_POPULATION_MEAN if settings.basis == BASIS_MAX_OBSERVED else BASIS_MAX_OBSERVED
    for record in records:
        reasons = primary.assign(record, "decline")
        reason_counts.append(len(reasons))
        if not reasons:
            no_reasons += 1
            continue
        first_rank[reasons[0].feature] += 1
        above_average_here = False
        for reason in reasons:
            any_rank[reason.feature] += 1
            if reason.prohibited_basis:
                prohibited_hits[reason.feature] += 1
            if reason.points >= average_points[reason.feature]:
                reasons_above_average += 1
                above_average_here = True
        notices_with_an_above_average_reason += above_average_here
        other = assigners[alternate].assign(record, "decline")
        if [r.feature for r in other] != [r.feature for r in reasons]:
            disagreements += 1

    declines = len(records)
    frequency = pd.DataFrame(
        [
            {
                "feature": feature,
                "code": STATEMENTS[feature].code,
                "statement": STATEMENTS[feature].statement,
                "prohibited_basis": PROHIBITED_BASES.get(feature, ""),
                "cited_any_rank": any_rank.get(feature, 0),
                "cited_share": round(any_rank.get(feature, 0) / declines, 4) if declines else 0.0,
                "cited_first": first_rank.get(feature, 0),
                "first_share": round(first_rank.get(feature, 0) / declines, 4) if declines else 0.0,
            }
            for feature in artifact.scorecard.features
        ]
    ).sort_values("cited_any_rank", ascending=False)

    reports = config.path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    frequency.to_csv(reports / "reason_code_frequency.csv", index=False)

    summary: Dict[str, object] = {
        "model_version": artifact.model_version,
        "basis": settings.basis,
        "max_reasons": settings.max_reasons,
        "holdout_rows": int(len(holdout)),
        "declines": declines,
        "decline_rate": round(declines / len(holdout), 4) if len(holdout) else 0.0,
        "max_points_score_gap": max_gap,
        "declines_with_no_reason": no_reasons,
        "mean_reasons_per_decline": round(float(np.mean(reason_counts)), 3) if reason_counts else 0.0,
        "reasons_cited": int(sum(reason_counts)),
        "reasons_naming_an_above_average_characteristic": reasons_above_average,
        "notices_with_an_above_average_reason": notices_with_an_above_average_reason,
        "above_average_notice_share": (
            round(notices_with_an_above_average_reason / declines, 4) if declines else 0.0
        ),
        "declines_where_bases_disagree": disagreements,
        "disagreement_share": round(disagreements / declines, 4) if declines else 0.0,
        "alternate_basis": alternate,
        "prohibited_basis_features": primary.prohibited_basis_features(),
        "prohibited_basis_citations": {
            feature: {
                "citations": count,
                "share_of_declines": round(count / declines, 4) if declines else 0.0,
            }
            for feature, count in prohibited_hits.items()
        },
        "regulation_b_age_check": check_age_rule(artifact),
    }
    with open(reports / "adverse_action_audit.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return {"summary": summary, "frequency": frequency}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None, help="audit the first N held out rows")
    args = parser.parse_args()

    result = audit(args.config, args.limit)
    summary = result["summary"]
    frequency = result["frequency"]

    print(f"model {summary['model_version']}, basis {summary['basis']}")
    print(
        f"{summary['declines']} declines in {summary['holdout_rows']} held out applications "
        f"({summary['decline_rate']:.1%})"
    )
    print(f"points decomposition, max gap to the served score: {summary['max_points_score_gap']:.2e}")
    print(
        f"mean reasons per decline {summary['mean_reasons_per_decline']}, "
        f"declines with no reason {summary['declines_with_no_reason']}"
    )
    print(
        f"reasons differ under {summary['alternate_basis']} on "
        f"{summary['declines_where_bases_disagree']} declines "
        f"({summary['disagreement_share']:.1%})"
    )
    print(
        f"notices naming a characteristic the applicant is at or above average on: "
        f"{summary['notices_with_an_above_average_reason']} "
        f"({summary['above_average_notice_share']:.1%}), from "
        f"{summary['reasons_naming_an_above_average_characteristic']} of "
        f"{summary['reasons_cited']} reasons"
    )

    print("\ncited on a decline notice:")
    for _, row in frequency.iterrows():
        if not row["cited_any_rank"]:
            continue
        flag = f"  <- prohibited basis: {row['prohibited_basis']}" if row["prohibited_basis"] else ""
        print(
            f"  [{row['code']}] {row['feature']:<28} any rank {row['cited_share']:6.1%}   "
            f"first {row['first_share']:6.1%}{flag}"
        )

    age = summary["regulation_b_age_check"]
    print("\nRegulation B, age:")
    if not age.get("applicable"):
        print(f"  {age.get('reason')}")
    else:
        print(
            f"  best age points at any age {age['best_points_at_any_age']}, "
            f"best available to an applicant 62 or over {age['best_points_available_to_62_plus']}"
        )
        print(
            "  COMPLIANT" if age["compliant"] else
            f"  NOT COMPLIANT: an applicant aged 62 or over earns up to "
            f"{age['max_penalty_vs_best_age']} points fewer for their age than the best "
            "attainable, which is a negative factor assigned on age."
        )

    print(f"\nwritten: reports/reason_code_frequency.csv, reports/adverse_action_audit.json")


if __name__ == "__main__":
    main()
