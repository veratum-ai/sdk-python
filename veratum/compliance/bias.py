"""Built-in bias computation engine.

Computes adverse impact metrics required by:
- NYC Local Law 144 (bias audit for automated employment decision tools)
- EEOC Uniform Guidelines §60-3.4D (four-fifths / 80% rule)
- EU AI Act Art.9 (risk management — bias monitoring)
- Colorado SB24-205 s6 (impact assessment)
- NAIC Model Bulletin (unfair discrimination in insurance)

Produces litigation-ready audit artifacts with statistical backing.
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core metric computations
# ---------------------------------------------------------------------------

def selection_rate(
    decisions: List[Dict[str, Any]],
    group_field: str,
    outcome_field: str = "selected",
    positive_value: Any = True,
) -> Dict[str, float]:
    """
    Compute selection rate per group.

    Selection rate = (# positive outcomes in group) / (# total in group)

    Args:
        decisions: List of decision dicts, each containing at least
                   group_field and outcome_field.
        group_field: Key identifying the demographic group (e.g., "race", "gender").
        outcome_field: Key identifying the outcome (default: "selected").
        positive_value: Value that counts as a positive outcome (default: True).

    Returns:
        Dict mapping group name → selection rate (0.0–1.0).
    """
    counts: Dict[str, int] = defaultdict(int)
    positives: Dict[str, int] = defaultdict(int)

    for d in decisions:
        group = d.get(group_field)
        if group is None:
            continue
        group = str(group)
        counts[group] += 1
        if d.get(outcome_field) == positive_value:
            positives[group] += 1

    rates: Dict[str, float] = {}
    for group, total in counts.items():
        rates[group] = positives[group] / total if total > 0 else 0.0
    return rates


def impact_ratio(
    rates: Dict[str, float],
    reference_group: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute impact ratio for each group relative to a reference group.

    Impact ratio = group_rate / reference_rate
    If no reference group specified, uses group with highest selection rate.

    Per EEOC Uniform Guidelines §60-3.4D, a ratio below 0.8 (four-fifths)
    indicates potential adverse impact.

    Args:
        rates: Dict mapping group → selection rate.
        reference_group: Group to compare against (optional; defaults to highest).

    Returns:
        Dict mapping group → impact ratio.
    """
    if not rates:
        return {}

    if reference_group and reference_group in rates:
        ref_rate = rates[reference_group]
    else:
        ref_rate = max(rates.values())

    if ref_rate == 0:
        return {g: 0.0 for g in rates}

    return {g: rate / ref_rate for g, rate in rates.items()}


def four_fifths_rule(
    rates: Dict[str, float],
    reference_group: Optional[str] = None,
    threshold: float = 0.8,
) -> Dict[str, Dict[str, Any]]:
    """
    Apply the EEOC four-fifths (80%) rule.

    For each group, determines whether the selection rate is at least
    80% of the reference group's rate. Groups below this threshold
    show potential adverse impact.

    Legal basis:
    - EEOC Uniform Guidelines on Employee Selection Procedures (29 CFR 1607)
    - Griggs v. Duke Power Co., 401 U.S. 424 (1971)

    Args:
        rates: Dict mapping group → selection rate.
        reference_group: Reference group (defaults to highest rate).
        threshold: Four-fifths threshold (default 0.8 per EEOC).

    Returns:
        Dict mapping group → {
            "selection_rate": float,
            "impact_ratio": float,
            "passes_four_fifths": bool,
            "adverse_impact_detected": bool,
        }
    """
    ratios = impact_ratio(rates, reference_group)
    results: Dict[str, Dict[str, Any]] = {}

    for group in rates:
        ratio = ratios.get(group, 0.0)
        results[group] = {
            "selection_rate": rates[group],
            "impact_ratio": round(ratio, 6),
            "passes_four_fifths": ratio >= threshold,
            "adverse_impact_detected": ratio < threshold,
        }

    return results


# ---------------------------------------------------------------------------
# NYC LL144 bias audit report generator
# ---------------------------------------------------------------------------

def nyc_ll144_bias_audit(
    decisions: List[Dict[str, Any]],
    *,
    group_fields: Optional[List[str]] = None,
    outcome_field: str = "selected",
    positive_value: Any = True,
    reference_groups: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Generate NYC Local Law 144 compliant bias audit report.

    LL144 requires annual bias audits of automated employment decision
    tools (AEDTs) with selection rate and impact ratio analysis by:
    - Race/ethnicity
    - Sex/gender
    - Intersectional categories (race × gender)

    The audit must be conducted by an independent auditor but the
    underlying calculations are standardized.

    Args:
        decisions: List of decision records.
        group_fields: Demographic fields to analyze (default: ["race", "gender"]).
        outcome_field: Outcome field name.
        positive_value: Value indicating positive outcome.
        reference_groups: Dict mapping field → reference group name.

    Returns:
        LL144-compliant audit report dict.
    """
    if group_fields is None:
        group_fields = ["race", "gender"]
    if reference_groups is None:
        reference_groups = {}

    report: Dict[str, Any] = {
        "audit_type": "nyc_ll144_bias_audit",
        "regulation": "NYC Local Law 144 of 2021",
        "total_decisions": len(decisions),
        "analyses": {},
        "intersectional_analyses": {},
        "overall_adverse_impact": False,
        "flags": [],
    }

    # Per-field analysis
    for field in group_fields:
        rates = selection_rate(decisions, field, outcome_field, positive_value)
        if not rates:
            continue
        ref = reference_groups.get(field)
        results = four_fifths_rule(rates, reference_group=ref)
        report["analyses"][field] = {
            "groups_analyzed": len(rates),
            "results": results,
        }
        # Check for adverse impact
        for group, data in results.items():
            if data["adverse_impact_detected"]:
                report["overall_adverse_impact"] = True
                report["flags"].append({
                    "field": field,
                    "group": group,
                    "impact_ratio": data["impact_ratio"],
                    "severity": "high" if data["impact_ratio"] < 0.6 else "medium",
                    "legal_reference": "EEOC 29 CFR 1607.4D",
                })

    # Intersectional analysis (race × gender as required by LL144)
    if len(group_fields) >= 2:
        for i, f1 in enumerate(group_fields):
            for f2 in group_fields[i + 1:]:
                intersect_field = f"{f1}_x_{f2}"
                # Build composite group
                augmented = []
                for d in decisions:
                    v1 = d.get(f1)
                    v2 = d.get(f2)
                    if v1 is not None and v2 is not None:
                        augmented.append({
                            **d,
                            intersect_field: f"{v1}_{v2}",
                        })
                if augmented:
                    rates = selection_rate(augmented, intersect_field, outcome_field, positive_value)
                    results = four_fifths_rule(rates)
                    report["intersectional_analyses"][intersect_field] = {
                        "groups_analyzed": len(rates),
                        "results": results,
                    }
                    for group, data in results.items():
                        if data["adverse_impact_detected"]:
                            report["overall_adverse_impact"] = True
                            report["flags"].append({
                                "field": intersect_field,
                                "group": group,
                                "impact_ratio": data["impact_ratio"],
                                "severity": "high" if data["impact_ratio"] < 0.6 else "medium",
                                "legal_reference": "NYC LL144 intersectional requirement",
                            })

    return report


# ---------------------------------------------------------------------------
# Statistical significance (Fisher's exact test approximation)
# ---------------------------------------------------------------------------

def _chi_squared_test(
    group_a_selected: int,
    group_a_total: int,
    group_b_selected: int,
    group_b_total: int,
) -> Dict[str, Any]:
    """
    Chi-squared test for independence between two groups.

    This is the standard statistical test used in EEOC enforcement
    to determine whether selection rate differences are statistically
    significant beyond the four-fifths rule.

    Returns:
        Dict with chi2 statistic, p_value approximation, significant flag.
    """
    n = group_a_total + group_b_total
    if n == 0:
        return {"chi2": 0.0, "p_value": 1.0, "significant": False}

    # Observed
    o11 = group_a_selected
    o12 = group_a_total - group_a_selected
    o21 = group_b_selected
    o22 = group_b_total - group_b_selected

    # Expected
    row1 = o11 + o12
    row2 = o21 + o22
    col1 = o11 + o21
    col2 = o12 + o22

    if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
        return {"chi2": 0.0, "p_value": 1.0, "significant": False}

    e11 = (row1 * col1) / n
    e12 = (row1 * col2) / n
    e21 = (row2 * col1) / n
    e22 = (row2 * col2) / n

    # Chi-squared statistic with Yates' correction
    chi2 = 0.0
    for obs, exp in [(o11, e11), (o12, e12), (o21, e21), (o22, e22)]:
        if exp > 0:
            chi2 += (abs(obs - exp) - 0.5) ** 2 / exp

    # Approximate p-value using chi-squared CDF (1 degree of freedom)
    # Using the complementary error function approximation
    p_value = _chi2_survival(chi2, df=1)

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
    }


def _chi2_survival(x: float, df: int = 1) -> float:
    """
    Approximate survival function (1 - CDF) for chi-squared distribution.

    Uses the Wilson-Hilferty approximation for df=1.
    """
    if x <= 0:
        return 1.0
    if df == 1:
        # For df=1, P(X > x) ≈ 2 * (1 - Φ(√x)) where Φ is standard normal CDF
        z = math.sqrt(x)
        return 2.0 * (1.0 - _normal_cdf(z))
    return 0.5  # Fallback for other df


def _normal_cdf(z: float) -> float:
    """Approximate standard normal CDF using Abramowitz & Stegun formula 7.1.26."""
    if z < -8.0:
        return 0.0
    if z > 8.0:
        return 1.0

    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1.0 if z >= 0 else -1.0
    z_abs = abs(z)
    t = 1.0 / (1.0 + p * z_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z_abs * z_abs / 2.0)

    return 0.5 * (1.0 + sign * y)


# ---------------------------------------------------------------------------
# Full adverse impact analysis
# ---------------------------------------------------------------------------

def adverse_impact_analysis(
    decisions: List[Dict[str, Any]],
    group_field: str,
    outcome_field: str = "selected",
    positive_value: Any = True,
    reference_group: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive adverse impact analysis combining four-fifths rule
    and statistical significance testing.

    Suitable for:
    - EEOC Title VII compliance
    - NYC LL144 bias audits
    - Colorado SB24-205 impact assessments
    - EU AI Act Art.9 risk management

    Args:
        decisions: List of decision records.
        group_field: Demographic field to analyze.
        outcome_field: Outcome field.
        positive_value: Positive outcome value.
        reference_group: Reference group for comparison.

    Returns:
        Comprehensive analysis report.
    """
    rates = selection_rate(decisions, group_field, outcome_field, positive_value)
    if not rates:
        return {"error": "No data for the specified group field"}

    # Determine reference group
    if not reference_group or reference_group not in rates:
        reference_group = max(rates, key=rates.get)  # type: ignore[arg-type]

    ff_results = four_fifths_rule(rates, reference_group)

    # Count per group
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "positive": 0})
    for d in decisions:
        g = d.get(group_field)
        if g is None:
            continue
        g = str(g)
        counts[g]["total"] += 1
        if d.get(outcome_field) == positive_value:
            counts[g]["positive"] += 1

    # Statistical tests against reference group
    ref_counts = counts[reference_group]
    statistical_tests: Dict[str, Any] = {}
    for group in rates:
        if group == reference_group:
            continue
        g_counts = counts[group]
        test = _chi_squared_test(
            ref_counts["positive"], ref_counts["total"],
            g_counts["positive"], g_counts["total"],
        )
        statistical_tests[group] = test

    # Determine overall finding
    adverse_impact_found = any(
        r["adverse_impact_detected"] for r in ff_results.values()
    )
    statistically_significant = any(
        t.get("significant", False) for t in statistical_tests.values()
    )

    return {
        "reference_group": reference_group,
        "group_field": group_field,
        "total_decisions": len(decisions),
        "groups": {
            group: {
                **ff_results.get(group, {}),
                "count": counts[group]["total"],
                "positive_count": counts[group]["positive"],
                "statistical_test": statistical_tests.get(group),
            }
            for group in rates
        },
        "summary": {
            "adverse_impact_detected": adverse_impact_found,
            "statistically_significant": statistically_significant,
            "practical_significance": adverse_impact_found and statistically_significant,
            "recommendation": (
                "Adverse impact detected with statistical significance. "
                "Review selection criteria for potential disparate impact."
                if adverse_impact_found and statistically_significant
                else (
                    "Adverse impact detected by four-fifths rule but not "
                    "statistically significant. Monitor and re-evaluate."
                    if adverse_impact_found
                    else "No adverse impact detected."
                )
            ),
        },
        "legal_references": [
            "EEOC Uniform Guidelines 29 CFR 1607.4D",
            "Griggs v. Duke Power Co., 401 U.S. 424 (1971)",
            "Connecticut v. Teal, 457 U.S. 440 (1982)",
        ],
    }
