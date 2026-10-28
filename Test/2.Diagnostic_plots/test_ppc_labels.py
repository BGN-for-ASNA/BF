"""Titles and axis labels of every PPC plot in BayesForge/Diagnostic/ppc.py.

Every public PPC function takes ``title``, ``name``, ``x_label`` and
``y_label``:

  * ``name`` names the outcome in the default title, the axis titles and the
    trace names (``m.diag`` passes the observed site name, e.g. "height"), so
    nothing on the plot should still read "y"/"yrep" (the bayesplot jargon the
    labels replaced -- see ``_obs``/``_preds``/``_title`` in ppc.py);
  * ``title`` replaces the figure title verbatim;
  * ``x_label``/``y_label`` override the axis titles -- a string sets every
    x (y) axis, a list sets them one per subplot in layout order and a None
    entry keeps that axis's own default (``_axis_labels`` in ppc.py).

Each test below is parametrized over ``(function name, call lambda)`` so every
function gets its own test id, and checks, per function:

  1. default call with name="height": the title contains "height" and no axis
     title / trace name / subplot annotation still uses "y" or "yrep";
  2. title="My title" -> fig.layout.title.text == "My title";
  3. x_label/y_label as strings -> every xaxis*/yaxis* title is that string;
  4. multi-subplot figures: x_label/y_label as a list with one entry per axis,
     the second entry None -> each axis got its entry, the None one kept its
     default text;
  5. labels do not change the data: same traces, same x/y arrays as the
     default call.

Sections 6 to 8 check the label details that are not per-function: ppc_hist's
shared bins, ppc_bars' whole-number guard, and the discrete-x ticks and
rounded group levels drawn by ``_discrete_ticks``/``_level_labels``.

Fixtures are the same arrays as test_ppc_vs_bayesplot.py reads from
fixtures/input.json (continuous y/yrep/log_lik, count y/yrep, x).
"""
import json
import os
import re
import warnings

import numpy as np
import pytest

from BayesForge.Diagnostic import ppc

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "fixtures", "input.json")) as _f:
    INP = json.load(_f)

Y = np.array(INP["continuous"]["y"], dtype=float)          # (30,)
YREP = np.array(INP["continuous"]["yrep"], dtype=float)    # (200, 30)
LL = np.array(INP["continuous"]["log_lik"], dtype=float)   # (200, 30)
X = np.array(INP["x"], dtype=float)                        # (30,) covariate
CY = np.array(INP["count"]["y"])                           # (30,) ints
CYREP = np.array(INP["count"]["yrep"])                     # (200, 30) ints
# X is loaded for parity with test_ppc_vs_bayesplot.py; the label checks below
# call each function with its own defaults, so x is never passed to it.

NAME = "height"
TITLE = "My title"
X_CUSTOM = "X custom"
Y_CUSTOM = "Y custom"


def _call_ppc_loo_intervals(**kwargs):
    """ppc_loo_intervals is a deprecated alias of ppc_intervals_sorted and
    warns on every call; the warning is not what these tests are about, so
    keep it out of the report (test_ppc_vs_bayesplot.py covers the alias)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ppc.ppc_loo_intervals(Y, YREP, **kwargs)


# (function name, call lambda) -- one entry per public PPC function; the
# lambdas take only the labelling kwargs (title/name/x_label/y_label).
PPC_PLOTS = [
    ("ppc_density", lambda **kw: ppc.ppc_density(Y, YREP, **kw)),
    ("ppc_hist", lambda **kw: ppc.ppc_hist(Y, YREP, **kw)),
    ("ppc_boxplot", lambda **kw: ppc.ppc_boxplot(Y, YREP, **kw)),
    ("ppc_stat", lambda **kw: ppc.ppc_stat(Y, YREP, **kw)),
    ("ppc_stat_2d", lambda **kw: ppc.ppc_stat_2d(Y, YREP, **kw)),
    ("ppc_intervals", lambda **kw: ppc.ppc_intervals(Y, YREP, **kw)),
    ("ppc_ribbon", lambda **kw: ppc.ppc_ribbon(Y, YREP, **kw)),
    ("ppc_error_scatter", lambda **kw: ppc.ppc_error_scatter(Y, YREP, **kw)),
    ("ppc_error_hist", lambda **kw: ppc.ppc_error_hist(Y, YREP, **kw)),
    ("ppc_scatter", lambda **kw: ppc.ppc_scatter(Y, YREP, **kw)),
    ("ppc_rootogram", lambda **kw: ppc.ppc_rootogram(CY, CYREP, **kw)),
    ("ppc_bars", lambda **kw: ppc.ppc_bars(CY, CYREP, **kw)),
    ("ppc_loo_pit", lambda **kw: ppc.ppc_loo_pit(Y, YREP, log_likelihood=LL, **kw)),
    ("ppc_intervals_sorted", lambda **kw: ppc.ppc_intervals_sorted(Y, YREP, **kw)),
    ("ppc_loo_intervals", _call_ppc_loo_intervals),
]
PPC_IDS = [name for name, _ in PPC_PLOTS]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _axis_titles(fig, kind):
    """{axis key: title text} for every xaxis*/yaxis* in fig.layout, in the
    numeric layout order _axis_labels() itself uses."""
    layout = fig.layout.to_plotly_json()
    keys = sorted((k for k in layout if re.fullmatch(kind + r"\d*", k)),
                  key=lambda k: int(k[len(kind):] or 1))
    return {k: ((layout[k] or {}).get("title") or {}).get("text") for k in keys}


def _labelled_texts(fig):
    """Every string the figure shows as a title or label: the figure title, all
    axis titles, all trace names and all subplot annotations."""
    layout = fig.layout.to_plotly_json()
    texts = [(layout.get("title") or {}).get("text")]
    texts += list(_axis_titles(fig, "xaxis").values())
    texts += list(_axis_titles(fig, "yaxis").values())
    texts += [tr.name for tr in fig.data]
    texts += [a.get("text") for a in layout.get("annotations", [])]
    return [t for t in texts if t]


def _as_comparable(value):
    """Trace coordinates as an array: segment traces (ppc_intervals) hold
    None-separated tuples, everything else a numeric array."""
    if value is None:
        return None
    seq = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
    if any(v is None for v in seq):
        return np.asarray(seq, dtype=object)
    return np.asarray(seq, dtype=float)


# ---------------------------------------------------------------------------
# 1. default labels name the outcome, no leftover "y"/"yrep"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", PPC_PLOTS, ids=PPC_IDS)
def test_default_labels_name_the_outcome(case):
    fname, call = case
    fig = call(name=NAME)

    title = fig.layout.title.text or ""
    assert NAME in title, f"{fname}: default title {title!r} does not name the outcome"

    texts = _labelled_texts(fig)
    assert any(NAME in t for t in texts), \
        f"{fname}: no axis title, trace name or subplot title mentions {NAME!r}"

    leftovers = []
    for text in texts:
        if re.search(r"\byrep\b", text, re.IGNORECASE):
            leftovers.append((text, "says 'yrep'"))
        elif re.search(r"\by\b", text):
            leftovers.append((text, "uses a bare 'y' for the outcome"))
    assert not leftovers, f"{fname}: labels still use the placeholder names: {leftovers}"


# ---------------------------------------------------------------------------
# 2. title= replaces the figure title verbatim
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", PPC_PLOTS, ids=PPC_IDS)
def test_title_overrides_default(case):
    fname, call = case
    fig = call(name=NAME, title=TITLE)
    assert fig.layout.title.text == TITLE, f"{fname}: title= was not used verbatim"


# ---------------------------------------------------------------------------
# 3. string x_label/y_label set every axis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", PPC_PLOTS, ids=PPC_IDS)
def test_string_labels_set_every_axis(case):
    fname, call = case
    fig = call(name=NAME, x_label=X_CUSTOM, y_label=Y_CUSTOM)

    x_axes, y_axes = _axis_titles(fig, "xaxis"), _axis_titles(fig, "yaxis")
    assert x_axes and y_axes, f"{fname}: no xaxis*/yaxis* key in fig.layout"
    wrong_x = {k: v for k, v in x_axes.items() if v != X_CUSTOM}
    wrong_y = {k: v for k, v in y_axes.items() if v != Y_CUSTOM}
    assert not wrong_x, f"{fname}: x axes without the custom title: {wrong_x}"
    assert not wrong_y, f"{fname}: y axes without the custom title: {wrong_y}"


# ---------------------------------------------------------------------------
# 4. list x_label/y_label: one per subplot, None keeps that axis's default
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", PPC_PLOTS, ids=PPC_IDS)
def test_list_labels_one_per_subplot(case):
    fname, call = case
    default = call(name=NAME)
    default_x, default_y = _axis_titles(default, "xaxis"), _axis_titles(default, "yaxis")
    if len(default_x) < 2:
        pytest.skip(f"{fname} draws a single subplot: nothing to set one-per-axis")

    x_keys, y_keys = list(default_x), list(default_y)
    x_labels = [f"x axis {k}" for k in x_keys]
    y_labels = [f"y axis {k}" for k in y_keys]
    x_labels[1] = y_labels[1] = None       # the second axis keeps its own title

    fig = call(name=NAME, x_label=x_labels, y_label=y_labels)
    got_x, got_y = _axis_titles(fig, "xaxis"), _axis_titles(fig, "yaxis")
    assert list(got_x) == x_keys, f"{fname}: the set of x axes changed"
    assert list(got_y) == y_keys, f"{fname}: the set of y axes changed"

    for kind, got, want, defaults in (("x", got_x, x_labels, default_x),
                                      ("y", got_y, y_labels, default_y)):
        for key, text, default_text in zip(got, want, defaults.values()):
            if text is None:
                assert got[key] == default_text, (
                    f"{fname}: {kind} axis {key!r} was passed None but lost its "
                    f"default title {default_text!r} (got {got[key]!r})")
            else:
                assert got[key] == text, (
                    f"{fname}: {kind} axis {key!r} should read {text!r}, "
                    f"got {got[key]!r}")


# ---------------------------------------------------------------------------
# 5. labelling does not touch the data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", PPC_PLOTS, ids=PPC_IDS)
def test_labels_do_not_change_the_data(case):
    fname, call = case
    plain = call(name=NAME)
    labelled = call(name=NAME, title=TITLE, x_label=X_CUSTOM, y_label=Y_CUSTOM)

    assert len(labelled.data) == len(plain.data), (
        f"{fname}: {len(plain.data)} traces by default, "
        f"{len(labelled.data)} with labels")
    for i, (before, after) in enumerate(zip(plain.data, labelled.data)):
        assert before.type == after.type, (
            f"{fname}: trace {i} is {before.type} by default, {after.type} with labels")
        for attr in ("x", "y"):
            a, b = _as_comparable(getattr(before, attr)), _as_comparable(getattr(after, attr))
            assert (a is None) == (b is None), (
                f"{fname}: trace {i} attribute {attr!r} present in only one figure")
            if a is None:
                continue
            np.testing.assert_array_equal(
                a, b, err_msg=f"{fname}: trace {i} attribute {attr!r} changed "
                              "when the labels were set")


# ---------------------------------------------------------------------------
# 6. ppc_hist: one shared x scale and identical bins in every panel
# ---------------------------------------------------------------------------

def test_ppc_hist_panels_share_x_scale_and_bins():
    bins = 30
    fig = ppc.ppc_hist(Y, YREP, bins=bins)
    # the same seed-0 draw sample ppc_hist takes (n=8 default)
    idx = np.random.default_rng(0).choice(len(YREP), size=8, replace=False)
    lo = min(Y.min(), YREP[idx].min())
    hi = max(Y.max(), YREP[idx].max())

    layout = fig.layout.to_plotly_json()
    x_keys = sorted((k for k in layout if re.fullmatch(r"xaxis\d*", k)),
                    key=lambda k: int(k[len("xaxis"):] or 1))
    assert len(x_keys) >= len(fig.data), "no x axis per histogram panel"
    for k in x_keys:
        rng = (layout[k] or {}).get("range")
        assert rng is not None, f"panel axis {k!r} has no explicit x range"
        np.testing.assert_allclose([float(rng[0]), float(rng[1])], [lo, hi],
                                   rtol=1e-12, err_msg=f"{k}: x range not [lo, hi]")

    starts = {t.xbins.start for t in fig.data}
    ends = {t.xbins.end for t in fig.data}
    sizes = {t.xbins.size for t in fig.data}
    assert len(starts) == len(ends) == len(sizes) == 1, \
        "panels do not share identical xbins"
    np.testing.assert_allclose(starts.pop(), lo, rtol=1e-12)
    np.testing.assert_allclose(ends.pop(), hi, rtol=1e-12)
    np.testing.assert_allclose(sizes.pop(), (hi - lo) / bins, rtol=1e-12)


# ---------------------------------------------------------------------------
# 7. ppc_bars: whole-number guard, no cap on the number of categories
# ---------------------------------------------------------------------------

def test_ppc_bars_many_categories_and_non_integer_guard():
    # Count data with > 50 distinct values across y and yrep is fine:
    # bayesplot's ppc_bars only requires whole numbers, not few categories.
    rng = np.random.default_rng(0)
    y = np.arange(80)
    yrep = rng.poisson(40, size=(20, 80))
    assert len(np.unique(np.concatenate([y, yrep.ravel()]))) > 50
    fig = ppc.ppc_bars(y, yrep)
    assert len(fig.data) == 2  # observed bars + median/interval trace

    # Continuous outcomes are still refused.
    with pytest.raises(ValueError, match="whole numbers"):
        ppc.ppc_bars(y + 0.5, yrep)


# ---------------------------------------------------------------------------
# 8. discrete x: one tick per level, and levels rounded in panel titles
#
# A normalized 1/2/3 design reaches the plot as -1.2018504251546631 / 0 /
# +1.2018504251546631. Left to plotly, the x axis is labelled -1.0, -0.5,
# 0.0, 0.5, 1.0: five ticks, none of them under two of the three bars. Same
# story for the panel titles of a grouped plot. See _discrete_ticks and
# _level_labels in ppc.py.
# ---------------------------------------------------------------------------

XD = np.tile([-1.2018504251546631, 0.0, 1.2018504251546631], 10)   # (30,)
# group must vary independently of x, else each panel holds a single x level
# and there is nothing to tick.
GD = np.repeat([-1.2018504251546631, 0.0, 1.2018504251546631], 10)  # (30,)
LEVEL_TEXTS = ["-1.2", "0", "1.2"]


def _ticks(fig, axis="xaxis"):
    spec = fig.layout.to_plotly_json()[axis] or {}
    return spec.get("tickmode"), spec.get("tickvals"), spec.get("ticktext")


def _x_axes(fig):
    return sorted((k for k in fig.layout.to_plotly_json()
                   if re.fullmatch(r"xaxis\d*", k)),
                  key=lambda k: int(k[len("xaxis"):] or 1))


@pytest.mark.parametrize("fname", ["ppc_intervals", "ppc_ribbon"])
def test_discrete_x_gets_one_tick_per_level(fname):
    call = getattr(ppc, fname)
    mode, vals, texts = _ticks(call(Y, YREP, x=XD))
    assert mode == "array", f"{fname}: discrete x kept plotly's default ticks"
    np.testing.assert_allclose([float(v) for v in vals], np.unique(XD),
                               err_msg=f"{fname}: ticks are not the x levels")
    assert [str(t) for t in texts] == LEVEL_TEXTS, (
        f"{fname}: tick labels are not the short form of the levels: {texts}")


def test_discrete_x_gets_one_tick_per_level_in_every_panel():
    fig = ppc.ppc_intervals(Y, YREP, x=XD, group=GD, group_name="shade")
    axes = _x_axes(fig)
    assert len(axes) == 3, f"expected one x axis per shade level, got {axes}"
    for axis in axes:
        mode, vals, texts = _ticks(fig, axis)
        assert mode == "array", f"panel {axis!r} kept plotly's default ticks"
        np.testing.assert_allclose([float(v) for v in vals], np.unique(XD))
        assert [str(t) for t in texts] == LEVEL_TEXTS


@pytest.mark.parametrize("fname", ["ppc_intervals", "ppc_ribbon"])
def test_continuous_x_keeps_default_ticks(fname):
    mode, _, _ = _ticks(getattr(ppc, fname)(Y, YREP, x=Y + 0.1 * np.arange(len(Y))))
    assert mode is None, f"{fname}: a continuous x got forced ticks ({mode})"


def test_index_x_keeps_default_ticks():
    # x=None is the observation index: bayesplot's default, untouched.
    mode, _, _ = _ticks(ppc.ppc_intervals(Y, YREP))
    assert mode is None, "the observation index got forced ticks"


def test_non_numeric_x_keeps_default_ticks():
    # plotly draws its own categorical axis; numeric tickvals would not match.
    labels = np.tile(np.array(["low", "mid", "high"]), 10)
    mode, _, _ = _ticks(ppc.ppc_intervals(Y, YREP, x=labels))
    assert mode is None, "a string x got numeric tickvals"


@pytest.mark.parametrize("fname", ["ppc_density", "ppc_stat", "ppc_intervals"])
def test_group_levels_are_rounded_in_panel_titles(fname):
    call = getattr(ppc, fname)
    fig = call(Y, YREP, group=GD, group_name="shade")
    titles = [a["text"] for a in fig.layout.to_plotly_json()["annotations"]]
    want = [f"shade = {lv}" for lv in LEVEL_TEXTS]
    # ppc_stat appends its per-panel p value to the same title
    assert [t.split(" (p = ")[0] for t in titles[:3]] == want, (
        f"{fname}: panel titles are {titles[:3]}, expected {want}")


@pytest.mark.parametrize("levels, want", [
    ([-1.2018504251546631, 0.0, 1.2018504251546631], ["-1.2", "0", "1.2"]),
    ([1.0, 2.0, 3.0], ["1", "2", "3"]),                    # whole: no decimals
    ([0.0, 0.01, 0.02], ["0", "0.01", "0.02"]),            # close: enough of them
    (["a", "b"], ["a", "b"]),                              # not numbers: as-is
])
def test_level_labels(levels, want):
    assert ppc._level_labels(levels) == want
