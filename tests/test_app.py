"""The Streamlit dashboard.

``app.py`` computes nothing of its own, so there is no numerical result here
worth re-checking - every value it shows comes from functions the rest of this
suite already covers.  What these tests protect is the part that only fails
when the app is actually run: widget bounds, session state, and text that has
to survive a trip through a font.

Each of them corresponds to something that shipped broken and was caught by
running the thing rather than by reading it:

* the app imported a package whose ``__init__`` pulls in ``emcee``, so a
  machine without it saw an ImportError instead of a dashboard;
* ``number_input`` rejected the package's own default seed of 20260830 as
  above ``max_value`` - a crash on first load, not a warning;
* Microsoft JhengHei has no subscript glyphs, so ``I₁`` and ``I₂`` rendered as
  empty boxes, and switching the y-axis to log brought the same failure back
  through the tick formatter's mathtext minus sign;
* a machine with no CJK font at all - a bare Ubuntu runner, or the container
  Streamlit Cloud deploys into - drew every Chinese label on every figure as a
  box.  That one was found by this file, on CI, after the app had already been
  looked at on a Windows desktop where it could not happen.

Streamlit comes with both the ``dev`` and the ``app`` extra; with neither
installed, this module skips rather than fails.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from pathlib import Path

import pytest

from hiv_drc import DRC_2020, generate_observations, reproduction_number

testing = pytest.importorskip(
    "streamlit.testing.v1",
    reason='streamlit is in the optional "app" extra; pip install -e ".[app]"',
)

#: The dashboard, found relative to this file so any checkout works.
APP = Path(__file__).resolve().parents[1] / "app.py"

#: Generous, because a fit runs the forward model a few hundred times.  The
#: default of 3 seconds is a timeout, not a performance budget.
TIMEOUT = 120.0


def run_app(**state: object):
    """A freshly executed dashboard, with ``state`` written before the run.

    Every test gets its own instance.  ``AppTest`` carries session state
    across ``run()`` calls, so sharing one between tests would make them
    depend on the order they happen to execute in.
    """
    at = testing.AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    for key, value in state.items():
        # Written one key at a time: session_state is a proxy, not a dict.
        at.session_state[key] = value
    at.run()
    assert not at.exception, at.exception
    return at


def button(at, fragment: str):
    """The button whose label contains ``fragment``."""
    matches = [b for b in at.button if fragment in b.label]
    assert matches, f"no button matching {fragment!r} among {[b.label for b in at.button]}"
    return matches[0]


def checkbox(at, fragment: str):
    """The checkbox whose label contains ``fragment``."""
    matches = [c for c in at.checkbox if fragment in c.label]
    assert matches, f"no checkbox matching {fragment!r} among {[c.label for c in at.checkbox]}"
    return matches[0]


def missing_glyphs(caught, caplog) -> str:
    """Every complaint about a character the font could not draw, as one string.

    Missing glyphs arrive on two channels: ordinary text goes through
    ``warnings``, while mathtext - which the log-scale tick labels used to
    use - goes through matplotlib's logger.  Neither is an exception; the
    figure renders with blank boxes and nothing fails.
    """
    complaints = [str(w.message) for w in caught] + [r.getMessage() for r in caplog.records]
    return "\n".join(
        c for c in complaints if "missing from font" in c or "does not have a glyph" in c
    )


def fit_table(at) -> str:
    """The rendered Wald-interval table, or ``""`` when no fit is displayed."""
    tables = [block.value for block in at.markdown if "Wald" in block.value]
    return tables[0] if tables else ""


@pytest.fixture(scope="module")
def dashboard():
    """One baseline run, shared by the tests that only read from it."""
    return run_app()


# -- it starts at all -----------------------------------------------------


def test_the_dashboard_runs_at_the_baseline(dashboard):
    """The whole script executes: imports, integration, figures, tables."""
    r0 = reproduction_number(DRC_2020).R0
    assert dashboard.metric[0].value == f"{r0:.4f}"


def test_the_baseline_is_reported_as_a_self_sustaining_epidemic(dashboard):
    """R₀ = 1.21 ≥ 1, so the app must warn rather than reassure."""
    assert dashboard.warning, "R0 >= 1 should be flagged"
    assert not dashboard.success


def test_the_widget_bounds_admit_the_packages_own_default_seed(dashboard):
    """Regression: ``max_value`` once rejected 20260830 and crashed on load."""
    default = inspect.signature(generate_observations).parameters["seed"].default
    seeds = [w for w in dashboard.number_input if w.value == default]
    assert seeds, f"no seed input holding the package default {default}"


# -- the scenario sandbox -------------------------------------------------


def test_a_subcritical_contact_rate_is_reported_as_elimination():
    """β = 0.02 puts R₀ below 1, which has to read as success, not warning."""
    at = run_app(beta=0.02)
    assert float(at.metric[0].value) < 1.0
    assert at.success, "R0 < 1 should be flagged as the disease-free case"
    assert not at.warning


def test_the_extremes_of_every_slider_still_integrate():
    """No treatment and no behaviour change is a valid, if bleak, scenario."""
    at = run_app(beta=0.3, alpha=0.0, phi=0.0, years=100)
    assert float(at.metric[0].value) > 1.0


def test_every_glyph_the_figures_ask_for_exists_in_the_chosen_font(caplog):
    """Both panels, log axis included, render without falling back to boxes."""
    at = run_app()
    with caplog.at_level(logging.WARNING, logger="matplotlib"), \
            warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        checkbox(at, "S 與 R").check().run()
        checkbox(at, "對數").check().run()
    assert not at.exception, at.exception

    missing = missing_glyphs(caught, caplog)
    assert not missing, missing


def test_the_figures_drop_to_ascii_on_a_machine_with_no_chinese_font(monkeypatch, caplog):
    """The Linux case, reproduced here rather than waited for in CI.

    A bare Ubuntu runner - and the container Streamlit Cloud deploys into -
    ships DejaVu Sans and no CJK font at all, so every Chinese label on the
    figures becomes an empty box.  matplotlib does not raise for that; it
    substitutes and carries on.  Restricting the font list to DejaVu is what
    that machine looks like from inside the process.
    """
    from matplotlib import font_manager

    dejavu = [f for f in font_manager.fontManager.ttflist if f.name.startswith("DejaVu")]
    assert dejavu, "matplotlib ships DejaVu Sans; something is wrong with this environment"
    monkeypatch.setattr(font_manager.fontManager, "ttflist", dejavu)

    with caplog.at_level(logging.WARNING, logger="matplotlib"), \
            warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        at = run_app()
        # The widgets stay in Chinese - they are HTML, and the browser has
        # its own fonts.  Only what matplotlib draws has to change.
        checkbox(at, "對數").check().run()
    assert not at.exception, at.exception
    missing = missing_glyphs(caught, caplog)
    assert not missing, missing


# -- the inverse problem --------------------------------------------------


def test_a_fit_reports_intervals_that_contain_the_generating_values():
    """The default two-parameter fit, driven the way a user drives it."""
    at = run_app()
    button(at, "生成").click().run()
    button(at, "擬合").click().run()
    assert not at.exception, at.exception

    table = fit_table(at)
    assert "beta" in table and "alpha" in table
    assert "❌" not in table, f"an interval missed the truth:\n{table}"


def test_an_unidentifiable_fit_is_displayed_rather_than_hidden():
    """φ alongside β and α is not identified from A and T.

    The point is not that the estimates are bad - ``test_estimation`` owns
    that - but that the app renders the evidence instead of falling over on
    an interval that runs through zero.
    """
    at = run_app()
    at.multiselect[0].set_value(["beta", "alpha", "phi"]).run()
    button(at, "生成").click().run()
    button(at, "擬合").click().run()
    assert not at.exception, at.exception

    table = fit_table(at)
    assert table.count("|\n") >= 4, f"expected three parameter rows:\n{table}"
    assert "[-" in table, f"expected an interval reaching below zero:\n{table}"


def test_fitting_before_generating_is_refused_rather_than_attempted():
    at = run_app()
    button(at, "擬合").click().run()
    assert not at.exception, at.exception
    assert at.error, "clicking fit with no data should explain itself"
    assert not fit_table(at)


def test_fitting_nothing_is_refused():
    at = run_app()
    button(at, "生成").click().run()
    at.multiselect[0].set_value([]).run()
    button(at, "擬合").click().run()
    assert not at.exception, at.exception
    assert at.error
    assert not fit_table(at)


def test_changing_the_scenario_discards_data_generated_under_the_old_one():
    """Observations belong to the parameters that produced them.

    Leaving them on screen after a slider moves would show a scatter drawn
    from one parameter set beside a curve drawn from another.
    """
    at = run_app()
    button(at, "生成").click().run()
    assert any("觀測點" in block.value for block in at.caption)

    at.session_state["beta"] = 0.12
    at.run()
    assert not at.exception, at.exception
    assert not any("觀測點" in block.value for block in at.caption)
    assert any("尚未生成觀測資料" in block.value for block in at.info)


# -- the controls themselves ----------------------------------------------


def test_the_reset_button_restores_every_baseline():
    at = run_app(beta=0.3, alpha=0.1, phi=0.2, years=100)
    button(at, "基準值").click().run()
    assert not at.exception, at.exception
    assert at.session_state["beta"] == pytest.approx(DRC_2020.beta)
    assert at.session_state["alpha"] == pytest.approx(DRC_2020.alpha)
    assert at.session_state["phi"] == pytest.approx(DRC_2020.phi)
