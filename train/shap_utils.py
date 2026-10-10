"""Positive-class SHAP validation, case selection and formatting."""

from typing import Any, List, Sequence
import math
import re

from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import numpy as np
import pandas as pd

from . import shap_publication_labels as shap_labels
from .optional_dependencies import shap


SHAP_GLOBAL_TOP_FEATURES = 10
# Matched to the global figures so a reader moves between them without
# re-learning how many features each one shows. A model with fewer features
# than this displays all of them; the direct late-fusion meta-model has three.
SHAP_LOCAL_TOP_FEATURES = 10
SHAP_PAIR_COUNT = 5
SHAP_HIGH_PROBABILITY = 0.85
SHAP_LOW_PROBABILITY = 0.15
SHAP_FIGURE_DPI = 600
# SHAP's own default for force_plot, which is the canvas these figures have
# always used. A narrower canvas prints the hard-coded 12 pt labels larger once
# the figure is scaled into a journal column, but changes the layout, so the
# default is kept deliberately.
SHAP_FORCE_FIGURE_SIZE = (20, 3)


class ShapInputMismatchError(ValueError):
    """Raised when SHAP cases diverge from audited evaluation predictions."""


def validate_shap_output_probability(
        explanation: Any,
        probability: Sequence[float],
        model_key: str,
        tolerance: float = 1e-5,
):
    """Require base value plus SHAP contributions to reproduce probability."""
    shap_output = (
        np.asarray(explanation.base_values, dtype=float).reshape(-1)
        + np.asarray(explanation.values, dtype=float).sum(axis=1)
    )
    probability = np.asarray(probability, dtype=float)
    if not np.allclose(
            shap_output, probability, rtol=tolerance, atol=tolerance,
    ):
        max_difference = float(np.max(np.abs(shap_output - probability)))
        raise ShapInputMismatchError(
            f"{model_key} SHAP output is not on the audited probability scale; "
            f"max difference={max_difference:.3e}."
        )


def shap_positive_class_explanation(
        explainer: Any,
        shap_values: Any,
        X_plot: pd.DataFrame,
):
    """Normalize old/new SHAP classifier outputs to class 1 only."""
    n_samples, n_features = X_plot.shape
    if isinstance(shap_values, list):
        if len(shap_values) < 2:
            raise ValueError("Binary classifier SHAP output has no positive class.")
        values = np.asarray(shap_values[1], dtype=float)
    else:
        values = np.asarray(shap_values, dtype=float)
        if values.ndim == 3:
            if values.shape[:2] == (n_samples, n_features):
                if values.shape[2] < 2:
                    raise ValueError("Binary classifier SHAP output has no class 1 axis.")
                values = values[:, :, 1]
            elif values.shape[1:] == (n_samples, n_features):
                if values.shape[0] < 2:
                    raise ValueError("Binary classifier SHAP output has no class 1 axis.")
                values = values[1]
            else:
                raise ValueError(
                    f"Unexpected three-dimensional SHAP shape: {values.shape}."
                )
    if values.ndim == 1 and n_samples == 1:
        values = values.reshape(1, -1)
    if values.shape != (n_samples, n_features):
        raise ValueError(
            f"SHAP shape {values.shape} does not match data shape "
            f"{(n_samples, n_features)}."
        )

    expected_value = np.asarray(explainer.expected_value, dtype=float)
    if expected_value.ndim == 0 or expected_value.size == 1:
        positive_base_value = float(expected_value.ravel()[0])
    else:
        positive_base_value = float(expected_value.ravel()[1])
    return shap.Explanation(
        values=values,
        base_values=np.repeat(positive_base_value, n_samples),
        data=X_plot.to_numpy(),
        feature_names=X_plot.columns.tolist(),
    )


def select_shap_explanation_rows(
        probability: Sequence[float],
        max_explain_cases: int,
        pair_count: int = SHAP_PAIR_COUNT,
        high_cutoff: float = SHAP_HIGH_PROBABILITY,
        low_cutoff: float = SHAP_LOW_PROBABILITY,
) -> tuple[np.ndarray, List[tuple[int, int, int]]]:
    """Keep up to five strict high/low-probability pairs in the SHAP sample."""
    probability = np.asarray(probability, dtype=float)
    high_positions = np.flatnonzero(probability > high_cutoff)
    low_positions = np.flatnonzero(probability < low_cutoff)
    high_positions = high_positions[np.argsort(probability[high_positions])[::-1]]
    low_positions = low_positions[np.argsort(probability[low_positions])]
    n_pairs = min(pair_count, len(high_positions), len(low_positions))
    pairs = [
        (pair_number, int(high_positions[pair_number - 1]),
         int(low_positions[pair_number - 1]))
        for pair_number in range(1, n_pairs + 1)
    ]

    required_positions = []
    for _, high_position, low_position in pairs:
        required_positions.extend([high_position, low_position])
    required_positions = list(dict.fromkeys(required_positions))
    remaining_positions = np.setdiff1d(
        np.arange(len(probability), dtype=int),
        np.asarray(required_positions, dtype=int),
        assume_unique=False,
    )
    remaining_count = max(0, min(
        max_explain_cases - len(required_positions), len(remaining_positions)
    ))
    if remaining_count:
        rng = np.random.default_rng(42)
        sampled_positions = rng.choice(
            remaining_positions, size=remaining_count, replace=False
        ).tolist()
    else:
        sampled_positions = []
    explain_positions = np.asarray(
        required_positions + sampled_positions, dtype=int
    )
    return explain_positions, pairs


def shap_local_top_feature_explanation(
        explanation: Any,
        row_index: int,
        max_features: int = SHAP_LOCAL_TOP_FEATURES,
):
    """Select one sign-balanced local feature set for Force and Waterfall.

    When meaningful positive and negative SHAP contributions both exist, retain
    the strongest feature from each direction before filling the remaining slots
    by absolute importance. Contributions are never fabricated merely to add a
    plot color. Omitted contributions are absorbed into the displayed baseline.
    """
    row = explanation[row_index]
    values = np.asarray(row.values, dtype=float)
    n_keep = min(max_features, len(values))
    ranked_positions = np.argsort(np.abs(values))[::-1]
    max_absolute_value = float(np.max(np.abs(values))) if len(values) else 0.0
    sign_tolerance = max(1e-12, max_absolute_value * 1e-6)
    positive_positions = [
        int(position) for position in ranked_positions
        if values[position] > sign_tolerance
    ]
    negative_positions = [
        int(position) for position in ranked_positions
        if values[position] < -sign_tolerance
    ]

    if n_keep >= 2 and positive_positions and negative_positions:
        selected_positions = [positive_positions[0], negative_positions[0]]
        for position in ranked_positions:
            position = int(position)
            if position not in selected_positions:
                selected_positions.append(position)
            if len(selected_positions) == n_keep:
                break
        top_positions = np.asarray(sorted(
            selected_positions,
            key=lambda position: abs(values[position]),
            reverse=True,
        ), dtype=int)
    else:
        top_positions = ranked_positions[:n_keep]
    omitted_contribution = float(values.sum() - values[top_positions].sum())
    adjusted_base_value = float(np.asarray(row.base_values).ravel()[0]) + omitted_contribution
    local_explanation = shap.Explanation(
        values=values[top_positions],
        base_values=adjusted_base_value,
        data=np.asarray(row.data)[top_positions],
        feature_names=np.asarray(row.feature_names, dtype=object)[top_positions].tolist(),
    )
    return local_explanation, top_positions, omitted_contribution


def shap_plot_explanation_with_two_decimals(explanation: Any):
    """Copy one local explanation ready to plot.

    Feature values are formatted to two decimals and the programming names are
    swapped for the publication labels used by the global beeswarm, so Waterfall
    and Force figures read in the same terms as the summary figure. The audit
    CSVs keep the original column names.
    """
    formatted_data = []
    for value in np.asarray(explanation.data, dtype=object):
        try:
            numeric_value = float(value)
            formatted_data.append(
                f"{numeric_value:.2f}" if math.isfinite(numeric_value) else str(value)
            )
        except (TypeError, ValueError):
            formatted_data.append(str(value))
    return shap.Explanation(
        values=np.asarray(explanation.values, dtype=float),
        base_values=float(np.asarray(explanation.base_values).ravel()[0]),
        data=np.asarray(formatted_data, dtype=object),
        feature_names=[
            shap_labels.label_for(name) for name in explanation.feature_names
        ],
    )


def format_force_plot_axis_two_decimals(figure: Any, base_value: float):
    """Use two decimals and hide the tick printed directly below base value."""
    figure.canvas.draw()
    for axis in figure.axes:
        ticks = np.asarray(axis.get_xticks(), dtype=float)
        if not len(ticks):
            continue
        base_tick = float(ticks[np.argmin(np.abs(ticks - base_value))])
        tolerance = max(1e-10, np.ptp(ticks) * 1e-7)
        axis.xaxis.set_major_formatter(FuncFormatter(
            lambda value, _position, hidden=base_tick, atol=tolerance: (
                "" if np.isclose(value, hidden, atol=atol) else f"{value:.2f}"
            )
        ))


def format_shap_x_axes_two_decimals(figure: Any):
    """Format numeric SHAP output axes without replacing feature-name labels."""
    if figure.axes:
        figure.axes[0].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def format_waterfall_bar_labels_two_decimals(figure: Any):
    """Print every Waterfall bar label with two decimals.

    SHAP formats a bar with "%+0.02f" and then strips trailing zeros, so a
    contribution of exactly 0.10 prints as "+0.1" beside "+0.29" and the figure
    mixes one and two decimals. The bar labels are the only text objects on the
    main axis, so they can be rewritten in place. SHAP's unicode minus is kept.
    """
    if not figure.axes:
        return
    for label in figure.axes[0].texts:
        text = label.get_text().replace("−", "-")
        try:
            value = float(text)
        except ValueError:
            continue
        label.set_text(f"{value:+.2f}".replace("-", "−"))


def format_waterfall_axes_two_decimals(
        figure: Any,
        explanation: Any,
):
    """Preserve SHAP labels and prevent base/output ticks from overlapping."""
    figure.canvas.draw()
    if not figure.axes:
        return
    base_value = float(np.asarray(explanation.base_values).ravel()[0])
    output_value = base_value + float(np.asarray(explanation.values).sum())
    main_axis = figure.axes[0]
    main_ticks = np.asarray(main_axis.get_xticks(), dtype=float)
    hidden_ticks = []
    if len(main_ticks):
        for marker in (base_value, output_value):
            hidden_ticks.append(
                float(main_ticks[np.argmin(np.abs(main_ticks - marker))])
            )
    tolerance = max(1e-10, np.ptp(main_ticks) * 1e-7) if len(main_ticks) else 1e-10
    main_axis.xaxis.set_major_formatter(FuncFormatter(
        lambda value, _position, hidden=tuple(hidden_ticks), atol=tolerance: (
            "" if any(np.isclose(value, item, atol=atol) for item in hidden)
            else f"{value:.2f}"
        )
    ))

    # SHAP adds one auxiliary axis for E[f(X)] and one for f(x). Retain those
    # labels while rounding only their numeric component.
    for auxiliary_axis in figure.axes[1:]:
        tick_positions = auxiliary_axis.get_xticks()
        tick_labels = []
        label_colors = []
        for label in auxiliary_axis.get_xticklabels():
            text = label.get_text()
            match = re.search(r"(-?\d+(?:\.\d+)?)", text)
            if match:
                rounded = f"{float(match.group(1)):.2f}"
                text = text[:match.start()] + rounded + text[match.end():]
            tick_labels.append(text)
            label_colors.append(label.get_color())
        auxiliary_axis.set_xticks(tick_positions, labels=tick_labels)
        for label, color in zip(
                auxiliary_axis.get_xticklabels(), label_colors
        ):
            label.set_color(color)
