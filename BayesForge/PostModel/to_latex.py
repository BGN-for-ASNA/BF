import ast
import inspect
import re
import textwrap

import numpy as np

try:
    from IPython.display import display, Latex
except ImportError:
    display = print
    Latex = str

# Greek symbols mapping
greek_symbols = {
    'alpha': '\\alpha', 'beta': '\\beta', 'gamma': '\\gamma', 'delta': '\\delta',
    'epsilon': '\\epsilon', 'zeta': '\\zeta', 'eta': '\\eta', 'theta': '\\theta',
    'iota': '\\iota', 'kappa': '\\kappa', 'lambda_': '\\lambda', 'lambda': '\\lambda',
    'mu': '\\mu', 'nu': '\\nu', 'xi': '\\xi', 'omicron': 'o', 'pi': '\\pi',
    'rho': '\\rho', 'sigma': '\\sigma', 'tau': '\\tau', 'upsilon': '\\upsilon',
    'phi': '\\phi', 'chi': '\\chi', 'psi': '\\psi', 'omega': '\\omega'
}

# LaTeX accents mapping
latex_accents = {
    'bar': '\\bar', 'hat': '\\hat', 'tilde': '\\tilde', 'vec': '\\vec',
    'dot': '\\dot', 'ddot': '\\ddot'
}

# Structural keyword arguments that are implementation details, not math parameters
_STRUCTURAL_KWARGS = frozenset({
    'name', 'shape', 'sample', 'to_jax', 'create_obj', 'rng_key', 'event',
    'axis', 'keepdims', 'dtype', 'out'
})

# NumPyro primitives whose SITE NAME is the first positional argument.
_NUMPYRO_SITE_FNS = (
    "numpyro.sample", "sample",
    "numpyro.deterministic", "deterministic"
)


def extract_lines(code_str):
    # Drop blank lines AND comment-only lines: a `# note` line otherwise reached
    # the LaTeX output verbatim and broke the render.
    return [line.rstrip() for line in code_str.split("\n")
            if line.strip() != "" and not line.strip().startswith("#")]


def convert_to_greek(var_name):
    # Case-sensitive first, so Sigma (a covariance matrix) does not collapse
    # onto sigma (a scalar sd).
    if var_name in greek_symbols:
        return greek_symbols[var_name]
    lowered = var_name.lower()
    if lowered in greek_symbols and var_name[:1].isupper():
        sym = greek_symbols[lowered]
        # Only some Greek letters have a distinct uppercase command.
        if sym.lstrip("\\") in ("gamma", "delta", "theta", "lambda", "xi",
                                "pi", "sigma", "upsilon", "phi", "psi", "omega"):
            return "\\" + sym.lstrip("\\").capitalize()
        return sym
    return greek_symbols.get(lowered, var_name)


def format_latex_var(var_name):
    """
    Formats a Python variable name into a LaTeX string, handling Greek symbols,
    accents (e.g., 'bar_alpha' or 'alpha_bar'), and subscripts with escaped underscores.
    """
    if not isinstance(var_name, str) or not var_name:
        return str(var_name)
    if var_name in greek_symbols:
        return greek_symbols[var_name]

    if '_' in var_name:
        parts = var_name.split('_', 1)
        part1, part2 = parts[0], parts[1]

        if part1 in latex_accents:
            accent_cmd = latex_accents[part1]
            inner_var_latex = format_latex_var(part2)
            return f"{accent_cmd}{{{inner_var_latex}}}"
        if part2 in latex_accents:
            accent_cmd = latex_accents[part2]
            inner_var_latex = format_latex_var(part1)
            return f"{accent_cmd}{{{inner_var_latex}}}"
        base = convert_to_greek(part1)
        subscript = part2.replace('_', r'\_')
        return f"{base}_{{{subscript}}}"

    return convert_to_greek(var_name)


def _escape_text(s):
    """Escape underscores inside \\text{...} so KaTeX/MathJax never fail."""
    return str(s).replace("_", r"\_")


def _callee_str(func_node):
    try:
        return ast.unparse(func_node)
    except Exception:
        return ""


def _is_dist_callee(func_node):
    """True for m.dist.xxx(...) or numpyro.sample(...)-style sampling calls."""
    try:
        name = ast.unparse(func_node)
    except Exception:
        return False
    parts = name.split(".")
    return any(p == "dist" for p in parts) or name in (
        "numpyro.sample", "sample")


def _is_deterministic_callee(func_node):
    try:
        name = ast.unparse(func_node)
    except Exception:
        return False
    return name in ("numpyro.deterministic", "deterministic")


def convert_line_names(line):
    tokens = re.split(r'(\W)', line)
    return ''.join([format_latex_var(t) if re.match(r'^[A-Za-z_]\w*$', t) else t for t in tokens])


def _is_broadcast_slice(slice_node):
    """Return True if a subscript slice is purely numpy/jax dimension expansion
    such as ``[..., None]``, ``[:, None]``, or ``[None, :]``."""
    elts = slice_node.elts if isinstance(slice_node, ast.Tuple) else [slice_node]
    has_none = False
    for e in elts:
        if isinstance(e, ast.Constant) and e.value is None:
            has_none = True
        elif isinstance(e, ast.Constant) and e.value is Ellipsis:
            continue
        elif isinstance(e, ast.Slice) and e.lower is None and e.upper is None and e.step is None:
            continue
        else:
            return False
    return has_none


def ast_to_latex(node):
    """Recursively convert a Python AST expression node into clean LaTeX math."""
    if node is None:
        return ""

    if isinstance(node, ast.Name):
        if node.id == "None":
            return r"\text{None}"
        if node.id in ("True", "False"):
            return f"\\text{{{node.id}}}"
        return format_latex_var(node.id)

    if isinstance(node, ast.Constant):
        if node.value is Ellipsis:
            return r"\dots"
        if node.value is None:
            return r"\text{None}"
        if isinstance(node.value, str):
            return f"\\text{{{_escape_text(node.value)}}}"
        return str(node.value)

    if isinstance(node, ast.Attribute):
        # Matrix transpose `.T`
        if node.attr == "T":
            base = ast_to_latex(node.value)
            if isinstance(node.value, (ast.BinOp, ast.Call)):
                return f"\\left({base}\\right)^{{\\top}}"
            return f"{base}^{{\\top}}"
        # Mathematical constants `jnp.pi`, `np.pi`, `math.pi`, `jnp.e`
        if node.attr == "pi":
            return r"\pi"
        if node.attr == "e" and isinstance(node.value, ast.Name) and node.value.id in ("jnp", "np", "math"):
            return "e"
        if node.attr == "shape":
            return f"\\text{{dim}}({ast_to_latex(node.value)})"
        return f"{ast_to_latex(node.value)}.{_escape_text(node.attr)}"

    if isinstance(node, ast.UnaryOp):
        operand = ast_to_latex(node.operand)
        if isinstance(node.operand, ast.BinOp) and isinstance(node.operand.op, (ast.Add, ast.Sub)):
            operand = f"\\left({operand}\\right)"
        if isinstance(node.op, ast.USub):
            return f"-{operand}"
        if isinstance(node.op, ast.UAdd):
            return f"+{operand}"
        if isinstance(node.op, ast.Not):
            return f"\\neg {operand}"
        return operand

    if isinstance(node, ast.BinOp):
        left = ast_to_latex(node.left)
        right = ast_to_latex(node.right)
        if isinstance(node.op, ast.Div):
            return f"\\frac{{{left}}}{{{right}}}"
        if isinstance(node.op, ast.Pow):
            if isinstance(node.left, (ast.BinOp, ast.UnaryOp, ast.Call)):
                left = f"\\left({left}\\right)"
            return f"{left}^{{{right}}}"
        if isinstance(node.op, (ast.Mult, ast.MatMult)):
            if isinstance(node.left, ast.BinOp) and isinstance(node.left.op, (ast.Add, ast.Sub)):
                left = f"\\left({left}\\right)"
            if isinstance(node.right, ast.BinOp) and isinstance(node.right.op, (ast.Add, ast.Sub)):
                right = f"\\left({right}\\right)"
            return f"{left} \\cdot {right}"
        if isinstance(node.op, ast.Sub):
            if isinstance(node.right, ast.BinOp) and isinstance(node.right.op, (ast.Add, ast.Sub)):
                right = f"\\left({right}\\right)"
            return f"{left} - {right}"
        op = {
            ast.Add: '+',
            ast.Mod: r'\bmod',
            ast.FloorDiv: r'\lfloor / \rfloor',
        }.get(type(node.op))
        if op is None:
            return convert_line_names(ast.unparse(node))
        return f"{left} {op} {right}"

    if isinstance(node, ast.Compare):
        left = ast_to_latex(node.left)
        parts = [left]
        op_map = {
            ast.Eq: '=',
            ast.NotEq: r'\neq',
            ast.Lt: '<',
            ast.LtE: r'\leq',
            ast.Gt: '>',
            ast.GtE: r'\geq',
            ast.In: r'\in',
            ast.NotIn: r'\notin',
        }
        for op, comp in zip(node.ops, node.comparators):
            parts.append(op_map.get(type(op), '='))
            parts.append(ast_to_latex(comp))
        return " ".join(parts)

    if isinstance(node, ast.BoolOp):
        sep = r" \land " if isinstance(node.op, ast.And) else r" \lor "
        return sep.join(ast_to_latex(v) for v in node.values)

    if isinstance(node, ast.IfExp):
        body = ast_to_latex(node.body)
        cond = ast_to_latex(node.test)
        orelse = ast_to_latex(node.orelse)
        return f"\\begin{{cases}} {body} & \\text{{if }} {cond} \\\\ {orelse} & \\text{{otherwise}} \\end{{cases}}"

    if isinstance(node, ast.Subscript):
        # Drop pure dimension-expansion indexing like `sigma[..., None]`
        if _is_broadcast_slice(node.slice):
            return ast_to_latex(node.value)
        # x.shape[0] -> N_x
        if (isinstance(node.value, ast.Attribute)
                and node.value.attr == "shape"
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == 0):
            return f"N_{{{ast_to_latex(node.value.value)}}}"
        value_latex = ast_to_latex(node.value)
        slice_latex = ast_to_latex(node.slice)
        return f"{value_latex}[{slice_latex}]"

    if isinstance(node, ast.Tuple):
        return ", ".join(ast_to_latex(e) for e in node.elts)

    if isinstance(node, ast.List):
        return "\\left[" + ", ".join(ast_to_latex(e) for e in node.elts) + "\\right]"

    if isinstance(node, ast.Slice):
        lower = ast_to_latex(node.lower)
        upper = ast_to_latex(node.upper)
        step = ast_to_latex(node.step)
        if step:
            return f"{lower}:{upper}:{step}"
        return f"{lower}:{upper}"

    if isinstance(node, ast.Call):
        return _call_to_latex(node)

    return convert_line_names(ast.unparse(node))


def _call_to_latex(node):
    """Translate an ``ast.Call`` node (JAX, math, link, or general function) to LaTeX."""
    full_name = _callee_str(node.func)
    if isinstance(node.func, ast.Attribute):
        fn = node.func.attr
    elif isinstance(node.func, ast.Name):
        fn = node.func.id
    else:
        fn = full_name

    # Filter out non-mathematical structural keyword arguments
    pos_args = [ast_to_latex(a) for a in node.args]
    kw_args = [
        f"{format_latex_var(kw.arg)}={ast_to_latex(kw.value)}"
        for kw in node.keywords
        if kw.arg not in _STRUCTURAL_KWARGS
    ]
    all_args = pos_args + kw_args
    arg0 = pos_args[0] if pos_args else (kw_args[0] if kw_args else "")

    # 1. Transparent reshaping / casting wrappers
    if fn in ("asarray", "array", "stack", "hstack", "vstack", "concatenate",
              "concat", "expand_dims", "squeeze", "ravel", "flatten", "astype"):
        if len(node.args) >= 1 and isinstance(node.args[0], (ast.List, ast.Tuple)):
            elts = [ast_to_latex(e) for e in node.args[0].elts]
            return "\\left[" + ", ".join(elts) + "\\right]"
        return arg0

    # 2. Array constructors
    if fn in ("zeros", "zeros_like"):
        return "0"
    if fn in ("ones", "ones_like"):
        return "1"
    if fn in ("eye", "identity"):
        return r"\mathbf{I}"
    if fn == "repeat" and len(pos_args) >= 1:
        return pos_args[0]

    # 3. Elementary math & JAX functions
    if fn == "exp":
        return f"\\exp\\left({arg0}\\right)"
    if fn in ("log", "ln"):
        return f"\\log\\left({arg0}\\right)"
    if fn == "log1p":
        return f"\\log\\left(1 + {arg0}\\right)"
    if fn == "log10":
        return f"\\log_{{10}}\\left({arg0}\\right)"
    if fn == "log2":
        return f"\\log_{{2}}\\left({arg0}\\right)"
    if fn == "expm1":
        return f"\\exp\\left({arg0}\\right) - 1"
    if fn == "sqrt":
        return f"\\sqrt{{{arg0}}}"
    if fn == "square":
        if len(node.args) == 1 and isinstance(node.args[0], (ast.BinOp, ast.UnaryOp)):
            return f"\\left({arg0}\\right)^{{2}}"
        return f"{arg0}^{{2}}"
    if fn in ("abs", "fabs"):
        return f"\\left|{arg0}\\right|"
    if fn in ("sin", "cos", "tan", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan"):
        return f"\\{fn}\\left({arg0}\\right)"

    # 4. Reductions & aggregations
    if fn in ("sum", "nansum"):
        if len(node.args) == 1 and isinstance(node.args[0], ast.BinOp):
            return f"\\sum \\left({arg0}\\right)"
        return f"\\sum {arg0}"
    if fn in ("prod", "nanprod"):
        if len(node.args) == 1 and isinstance(node.args[0], ast.BinOp):
            return f"\\prod \\left({arg0}\\right)"
        return f"\\prod {arg0}"
    if fn in ("mean", "nanmean"):
        if len(node.args) == 1 and isinstance(node.args[0], ast.Name):
            return f"\\overline{{{arg0}}}"
        return f"\\text{{mean}}\\left({arg0}\\right)"
    if fn in ("std", "nanstd"):
        return f"\\text{{sd}}\\left({arg0}\\right)"
    if fn in ("var", "nanvar"):
        return f"\\text{{Var}}\\left({arg0}\\right)"
    if fn in ("min", "minimum", "amin"):
        return f"\\min\\left({', '.join(pos_args)}\\right)"
    if fn in ("max", "maximum", "amax"):
        return f"\\max\\left({', '.join(pos_args)}\\right)"
    if fn == "clip":
        return f"\\text{{clip}}\\left({', '.join(pos_args)}\\right)"

    # 5. Linear algebra
    if fn in ("dot", "matmul", "tensordot") and len(pos_args) >= 2:
        return f"{pos_args[0]} \\cdot {pos_args[1]}"
    if fn == "outer" and len(pos_args) >= 2:
        return f"{pos_args[0]} \\otimes {pos_args[1]}"
    if fn == "einsum" and len(pos_args) >= 3:
        return " \\cdot ".join(pos_args[1:])
    if fn == "diag":
        return f"\\text{{diag}}\\left({arg0}\\right)"
    if fn == "transpose":
        return f"\\left({arg0}\\right)^{{\\top}}"
    if fn == "cholesky":
        return f"\\text{{Chol}}\\left({arg0}\\right)"
    if fn == "inv":
        return f"\\left({arg0}\\right)^{{-1}}"
    if fn in ("det", "slogdet"):
        return f"\\det\\left({arg0}\\right)"

    # 6. Conditional `jnp.where(cond, a, b)`
    if fn == "where" and len(pos_args) == 3:
        return (f"\\begin{{cases}} {pos_args[1]} & \\text{{if }} {pos_args[0]} \\\\ "
                f"{pos_args[2]} & \\text{{otherwise}} \\end{{cases}}")

    # 7. Link functions (`m.link.*`, `jax.nn.*`, `scipy.special.*`)
    if fn == "logit":
        return f"\\text{{logit}}\\left({arg0}\\right)"
    if fn in ("inv_logit", "expit", "sigmoid", "logistic"):
        return f"\\text{{logit}}^{{-1}}\\left({arg0}\\right)"
    if fn == "inv_logit_scale":
        return f"\\left(2 \\cdot \\text{{logit}}^{{-1}}\\left({arg0}\\right) - 1\\right)"
    if fn == "softmax":
        return f"\\text{{softmax}}\\left({arg0}\\right)"
    if fn == "log_softmax":
        return f"\\log\\text{{softmax}}\\left({arg0}\\right)"
    if fn in ("probit", "ppf", "ndtri"):
        return f"\\Phi^{{-1}}\\left({arg0}\\right)"
    if fn in ("inv_probit", "phi", "cdf", "ndtr"):
        return f"\\Phi\\left({arg0}\\right)"
    if fn == "cloglog":
        return f"\\text{{cloglog}}\\left({arg0}\\right)"
    if fn == "inv_cloglog":
        return f"\\text{{cloglog}}^{{-1}}\\left({arg0}\\right)"
    if fn == "reciprocal":
        return f"\\frac{{1}}{{{arg0}}}"
    if fn == "softplus":
        return f"\\text{{softplus}}\\left({arg0}\\right)"
    if fn == "relu":
        return f"\\text{{ReLU}}\\left({arg0}\\right)"

    # 8. BF Gaussian & Effects helpers used inside expressions
    if fn == "distance_matrix":
        return f"\\mathbf{{D}}\\left({arg0}\\right)"
    if fn == "cov_GPL2":
        return f"\\mathbf{{K}}_{{\\text{{SE}}}}\\left({', '.join(all_args)}\\right)"
    if fn == "kernel_sq_exp":
        return f"\\text{{GP}}_{{\\text{{SE}}}}\\left({', '.join(all_args)}\\right)"
    if fn == "kernel_exp":
        return f"\\text{{GP}}_{{\\text{{Exp}}}}\\left({', '.join(all_args)}\\right)"
    if fn == "kernel_periodic":
        return f"\\text{{GP}}_{{\\text{{Per}}}}\\left({', '.join(all_args)}\\right)"
    if fn == "kernel_periodic_local":
        return f"\\text{{GP}}_{{\\text{{LocPer}}}}\\left({', '.join(all_args)}\\right)"
    if fn == "diag_pre_multiply" and len(pos_args) == 2:
        return f"\\text{{diag}}\\left({pos_args[0]}\\right) \\cdot {pos_args[1]}"
    if fn == "random_centered" and len(pos_args) == 3:
        return f"\\left(\\text{{diag}}\\left({pos_args[0]}\\right) \\cdot {pos_args[1]} \\cdot {pos_args[2]}\\right)^{{\\top}}"

    # 9. General fallback with escaped underscores in `\text{...}`
    safe_name = _escape_text(fn.capitalize())
    args_str = ", ".join(all_args)
    return f"\\text{{{safe_name}}}({args_str})"


def _kw_latex(call, key, default_latex):
    for i, kw in enumerate(call.keywords):
        if kw.arg == key:
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return _escape_text(kw.value.value)
            return ast_to_latex(kw.value)
    return default_latex


def _pos_or_kw_latex(call, pos_idx, key, default_latex):
    if pos_idx < len(call.args):
        val = call.args[pos_idx]
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            return _escape_text(val.value)
        return ast_to_latex(val)
    return _kw_latex(call, key, default_latex)


def _kw_bool(call, key, default=False):
    for kw in call.keywords:
        if kw.arg == key and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool):
            return kw.value.value
    return default


def _expand_bf_block_latex(lhs_latex, call):
    """If ``call`` is a high-level BF building block (``m.effects.*``, ``m.net.*``,
    ``m.gaussian.*``), return a list of LaTeX lines in top-down mathematical
    order (target equation first, hyperpriors below). Returns ``None`` otherwise.
    """
    if not isinstance(call, ast.Call):
        return None
    cname = _callee_str(call.func)
    parts = cname.split(".")
    if len(parts) < 2:
        return None
    submod, fn = parts[-2], parts[-1]

    if submod == "effects":
        if fn in ("varying_effects", "varying_intercept_slope", "varying_intercept_slope_non_centered"):
            gname = _kw_latex(call, "group_name", "group")
            gid = _pos_or_kw_latex(call, 2 if fn == "varying_effects" else 1,
                                   "group_id" if fn == "varying_effects" else "group", "g")
            centered = _kw_bool(call, "centered", fn == "varying_intercept_slope")
            if centered:
                return [
                    f"{lhs_latex} = \\mathbf{{V}}_{{\\text{{{gname}}}[{gid}]}}, \\quad "
                    f"\\mathbf{{V}}_{{\\text{{{gname}}}}} \\sim \\text{{MultivariateNormal}}"
                    f"\\left(\\begin{{bmatrix}} \\bar{{\\alpha}} \\\\ \\bar{{\\beta}} \\end{{bmatrix}}, "
                    f"\\text{{diag}}(\\sigma_{{\\text{{{gname}}}}}) \\cdot \\mathbf{{R}}_{{\\text{{{gname}}}}} "
                    f"\\cdot \\text{{diag}}(\\sigma_{{\\text{{{gname}}}}})\\right)",
                    f"\\bar{{\\alpha}} \\sim \\text{{Normal}}(0, 2), \\quad "
                    f"\\bar{{\\beta}} \\sim \\text{{Normal}}(0, 1), \\quad "
                    f"\\sigma_{{\\text{{{gname}}}}} \\sim \\text{{Exponential}}(1), \\quad "
                    f"\\mathbf{{R}}_{{\\text{{{gname}}}}} \\sim \\text{{LKJ}}(2)",
                ]
            return [
                f"{lhs_latex} = \\begin{{bmatrix}} \\bar{{\\alpha}} \\\\ \\bar{{\\beta}} \\end{{bmatrix}} "
                f"+ \\left(\\text{{diag}}(\\sigma_{{\\text{{{gname}}}}}) \\cdot \\mathbf{{L}}_{{\\text{{{gname}}}}} "
                f"\\cdot \\mathbf{{Z}}_{{\\text{{{gname}}}}}\\right)^{{\\top}}_{{[{gid}]}}",
                f"\\mathbf{{Z}}_{{\\text{{{gname}}}}} \\sim \\text{{Normal}}(0, 1), \\quad "
                f"\\mathbf{{L}}_{{\\text{{{gname}}}}} \\sim \\text{{LKJCholesky}}(2), \\quad "
                f"\\sigma_{{\\text{{{gname}}}}} \\sim \\text{{Exponential}}(1), \\quad "
                f"\\bar{{\\alpha}} \\sim \\text{{Normal}}(0, 2), \\quad "
                f"\\bar{{\\beta}} \\sim \\text{{Normal}}(0, 1)",
            ]

        if fn in ("varying_intercept", "varying_intercept_non_centered"):
            gname = _kw_latex(call, "group_name", "group")
            gid = _pos_or_kw_latex(call, 1, "group_id" if fn == "varying_intercept" else "group_idx", "g")
            if fn == "varying_intercept":
                return [
                    f"{lhs_latex} = \\alpha_{{\\text{{{gname}}}[{gid}]}}, \\quad "
                    f"\\alpha_{{\\text{{{gname}}}}} \\sim \\text{{Normal}}\\left(\\bar{{\\alpha}}_{{\\text{{{gname}}}}}, \\sigma_{{\\text{{{gname}}}}}\\right)",
                    f"\\bar{{\\alpha}}_{{\\text{{{gname}}}}} \\sim \\text{{Normal}}(0, 2), \\quad "
                    f"\\sigma_{{\\text{{{gname}}}}} \\sim \\text{{Exponential}}(1)",
                ]
            return [
                f"{lhs_latex} = \\bar{{\\alpha}}_{{\\text{{{gname}}}}} + \\sigma_{{\\text{{{gname}}}}} \\cdot z_{{\\text{{{gname}}}[{gid}]}}",
                f"z_{{\\text{{{gname}}}}} \\sim \\text{{Normal}}(0, 1), \\quad "
                f"\\bar{{\\alpha}}_{{\\text{{{gname}}}}} \\sim \\text{{Normal}}(0, 2), \\quad "
                f"\\sigma_{{\\text{{{gname}}}}} \\sim \\text{{Exponential}}(1)",
            ]

        if fn in ("varying_slope", "varying_slope_non_centered"):
            gname = _kw_latex(call, "group_name", "group")
            cat_idx = _pos_or_kw_latex(call, 0, "category_idx", "c")
            grp_idx = _pos_or_kw_latex(call, 1, "grouping_idx", "g")
            return [
                f"{lhs_latex} = \\left(\\text{{diag}}(\\sigma_{{\\text{{{gname}}}}}) \\cdot \\mathbf{{L}}_{{\\text{{{gname}}}}} "
                f"\\cdot \\mathbf{{Z}}_{{\\text{{{gname}}}}}\\right)^{{\\top}}_{{[{grp_idx}, {cat_idx}]}}",
                f"\\mathbf{{Z}}_{{\\text{{{gname}}}}} \\sim \\text{{Normal}}(0, 1), \\quad "
                f"\\mathbf{{L}}_{{\\text{{{gname}}}}} \\sim \\text{{LKJCholesky}}(2), \\quad "
                f"\\sigma_{{\\text{{{gname}}}}} \\sim \\text{{Exponential}}(1)",
            ]

        if fn == "nested_varying_effects":
            return [
                f"{lhs_latex} = \\boldsymbol{{\\theta}}_{{\\text{{nested}}[g_{{1}}, \\dots, g_{{L}}]}}",
                r"\boldsymbol{\theta}_{\ell} = \boldsymbol{\theta}_{\ell - 1} + "
                r"\left(\text{diag}(\sigma_{\ell}) \cdot \mathbf{L}_{\ell} \cdot \mathbf{Z}_{\ell}\right)^{\top}, \quad "
                r"\mathbf{Z}_{\ell} \sim \text{Normal}(0, 1), \quad "
                r"\mathbf{L}_{\ell} \sim \text{LKJCholesky}(2), \quad "
                r"\sigma_{\ell} \sim \text{Exponential}(1)",
            ]

        if fn == "phylo_effect":
            L_mat = _pos_or_kw_latex(call, 0, "L", r"\mathbf{L}")
            sd_val = _pos_or_kw_latex(call, 1, "sd", r"\sigma_{\text{phylo}}")
            z_name = _kw_latex(call, "name", r"z\_phylo")
            return [
                f"{lhs_latex} = {sd_val} \\cdot {L_mat} \\cdot {z_name}, \\quad "
                f"{z_name} \\sim \\text{{Normal}}(0, 1)"
            ]

    if submod in ("net", "net2"):
        if fn == "sender_receiver":
            return [
                f"{lhs_latex} = \\begin{{bmatrix}} s_{{i}} + r_{{j}} & s_{{j}} + r_{{i}} \\end{{bmatrix}}, \\quad "
                r"\begin{bmatrix} s \\ r \end{bmatrix} = \mathbf{X}\beta_{sr} + "
                r"\left(\text{diag}(\sigma_{sr}) \cdot \mathbf{L}_{sr} \cdot \mathbf{Z}_{sr}\right)^{\top}",
                r"\mathbf{Z}_{sr} \sim \text{Normal}(0, 1), \quad "
                r"\sigma_{sr} \sim \text{TruncatedNormal}(0, 2.5), \quad "
                r"\mathbf{L}_{sr} \sim \text{LKJCholesky}(2, 2.5)",
            ]
        if fn == "nodes_random_effects":
            return [
                f"{lhs_latex} = \\left(\\text{{diag}}(\\sigma_{{sr}}) \\cdot \\mathbf{{L}}_{{sr}} \\cdot \\mathbf{{Z}}_{{sr}}\\right)^{{\\top}}",
                r"\mathbf{Z}_{sr} \sim \text{Normal}(0, 1), \quad "
                r"\sigma_{sr} \sim \text{TruncatedNormal}(0, 2.5), \quad "
                r"\mathbf{L}_{sr} \sim \text{LKJCholesky}(2, 2.5)",
            ]
        if fn == "dyadic_effect":
            return [
                f"{lhs_latex} = \\mathbf{{X}}_{{d}}\\beta_{{d}} + "
                r"\left(\text{diag}(\sigma_{dr}) \cdot \mathbf{L}_{dr} \cdot \mathbf{Z}_{dr}\right)^{\top}",
                r"\mathbf{Z}_{dr} \sim \text{Normal}(0, 1), \quad "
                r"\sigma_{dr} \sim \text{TruncatedNormal}(0, 2.5), \quad "
                r"\mathbf{L}_{dr} \sim \text{LKJCholesky}(2, 2.5)",
            ]
        if fn == "block_model":
            grp = _pos_or_kw_latex(call, 0, "group", "g")
            return [
                f"{lhs_latex} = \\begin{{bmatrix}} B_{{{grp}[i], {grp}[j]}} & B_{{{grp}[j], {grp}[i]}} \\end{{bmatrix}}, \\quad "
                r"B \sim \text{Normal}(\mu_{B}, \sigma_{B})"
            ]
        if fn == "edge_weights":
            ename = _kw_latex(call, "name", r"w")
            return [
                f"{lhs_latex} = \\mu_{{{ename}}} + \\sigma_{{{ename}}} \\cdot z_{{{ename}}}, \\quad "
                f"z_{{{ename}}} \\sim \\text{{Normal}}(0, 1), \\quad "
                f"\\sigma_{{{ename}}} \\sim \\text{{HalfNormal}}(2.5)"
            ]
        if fn == "nodal_fixed_effects":
            X = _pos_or_kw_latex(call, 0, "design_fixed", r"\mathbf{X}")
            bname = _kw_latex(call, "name", r"\beta_{\text{fixed}}")
            return [
                f"{lhs_latex} = {X} \\cdot {bname}, \\quad {bname} \\sim \\text{{Normal}}(0, 2.5)"
            ]
        if fn == "nodal_random_effects":
            X = _pos_or_kw_latex(call, 0, "design_random", r"\mathbf{X}_{\text{rand}}")
            rname = _kw_latex(call, "name", r"\beta_{\text{rand}}")
            return [
                f"{lhs_latex} = {X} \\cdot {rname}, \\quad "
                f"{rname} \\sim \\text{{Normal}}(\\mu_{{g}}, \\sigma_{{g}})"
            ]

    if submod == "gaussian" and fn == "gaussian_process":
        Dmat = _pos_or_kw_latex(call, 0, "Dmat", r"\mathbf{D}")
        return [
            f"{lhs_latex} \\sim \\text{{MultivariateNormal}}\\left(0, \\mathbf{{K}}_{{\\text{{SE}}}}({Dmat}, \\eta^{{2}}, \\rho^{{2}}, \\sigma^{{2}})\\right)",
            r"\eta^{2} \sim \text{Exponential}(2), \quad \rho^{2} \sim \text{Exponential}(0.5)",
        ]

    return None


def _process_dist_call(func_call_node):
    """Return ``(dist_name, args_str, obs_var, name_var)`` for a distribution call."""
    site_name = None
    if (_callee_str(func_call_node.func) in _NUMPYRO_SITE_FNS
            and func_call_node.args
            and isinstance(func_call_node.args[0], ast.Constant)
            and isinstance(func_call_node.args[0].value, str)):
        site_name = func_call_node.args[0].value
        inner = next((a for a in func_call_node.args[1:]
                      if isinstance(a, ast.Call)), None)
        if inner is not None:
            name, args_str, obs, _ = _process_dist_call(inner)
            for kw in func_call_node.keywords:
                if kw.arg == "obs" and not (isinstance(kw.value, ast.Constant) and kw.value.value is None):
                    obs = (format_latex_var(kw.value.id)
                           if isinstance(kw.value, ast.Name)
                           else ast_to_latex(kw.value))
            return name, args_str, obs, format_latex_var(site_name)

    if isinstance(func_call_node.func, ast.Attribute):
        func_name_str = func_call_node.func.attr
    else:
        func_name_str = getattr(func_call_node.func, "id", _callee_str(func_call_node.func))
    func_name = "".join([part.capitalize() for part in func_name_str.split('_')])
    obs_var = None
    name_var = format_latex_var(site_name) if site_name else None
    pos_args = [ast_to_latex(a) for a in func_call_node.args]
    kw_args = []
    for kw in func_call_node.keywords:
        if kw.arg == 'obs':
            if not (isinstance(kw.value, ast.Constant) and kw.value.value is None):
                obs_var = format_latex_var(kw.value.id) if isinstance(kw.value, ast.Name) else ast_to_latex(kw.value)
        elif kw.arg == 'name':
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                name_var = format_latex_var(kw.value.value)
        elif kw.arg not in _STRUCTURAL_KWARGS:
            key = format_latex_var(kw.arg)
            if '_' in kw.arg:
                key = f'\\text{{{_escape_text(kw.arg.replace("_", " "))}}}'
            value = ast_to_latex(kw.value)
            kw_args.append(f"{key}={value}")
    args_str = ", ".join(pos_args + kw_args)
    return func_name, args_str, obs_var, name_var


def _format_for_header(target_node, iter_node):
    """Format ``for <target> in <iter>:`` as clean mathematical range notation."""
    t_latex = ast_to_latex(target_node)
    if isinstance(iter_node, ast.Call) and _callee_str(iter_node.func) == "range":
        rargs = iter_node.args
        if len(rargs) == 1:
            end_node = rargs[0]
            if isinstance(end_node, ast.Constant) and isinstance(end_node.value, int):
                return f"\\text{{for }} {t_latex} \\in 0, \\dots, {end_node.value - 1}\\text{{:}}"
            return f"\\text{{for }} {t_latex} \\in 0, \\dots, {ast_to_latex(end_node)} - 1\\text{{:}}"
        if len(rargs) >= 2:
            start_s = ast_to_latex(rargs[0])
            end_node = rargs[1]
            if isinstance(end_node, ast.Constant) and isinstance(end_node.value, int):
                return f"\\text{{for }} {t_latex} \\in {start_s}, \\dots, {end_node.value - 1}\\text{{:}}"
            return f"\\text{{for }} {t_latex} \\in {start_s}, \\dots, {ast_to_latex(end_node)} - 1\\text{{:}}"
    return f"\\text{{for }} {t_latex} \\in {ast_to_latex(iter_node)}\\text{{:}}"


def _format_with_header(item_node):
    """Format ``with numpyro.plate(...):`` as mathematical plate notation."""
    ctx = item_node.context_expr
    if isinstance(ctx, ast.Call) and "plate" in _callee_str(ctx.func):
        opt_var = ast_to_latex(item_node.optional_vars) if item_node.optional_vars else None
        if len(ctx.args) >= 2:
            pname = opt_var or ast_to_latex(ctx.args[0])
            psize = ast_to_latex(ctx.args[1])
            return f"\\text{{for }} {pname} \\in 1, \\dots, {psize}\\text{{:}}"
        args = ", ".join(ast_to_latex(a) for a in ctx.args)
        return f"\\text{{for }} {args}\\text{{:}}"
    return f"\\text{{with }} {ast_to_latex(ctx)}\\text{{:}}"


def _lhs_target_latex(target):
    if isinstance(target, ast.Tuple):
        return ", ".join(_lhs_target_latex(t) for t in target.elts)
    if isinstance(target, ast.Name):
        return format_latex_var(target.id)
    return ast_to_latex(target)


def _stmt_to_latex_lines(stmt, loop_var=None):
    """Convert an AST statement (including ``For``, ``With``, ``If``, ``Assign``,
    ``AugAssign``, ``Expr``) into a list of LaTeX equation lines.

    For compound blocks (``for``, ``with plate``), the block header is kept
    FIRST and its body statements follow in natural order so loop recurrences
    read top-to-bottom.
    """
    # 1. For loop: keep header on top, translate `z.append(expr)` using `loop_var`
    if isinstance(stmt, ast.For):
        lv = stmt.target.id if isinstance(stmt.target, ast.Name) else "t"
        header = _format_for_header(stmt.target, stmt.iter)
        body_lines = []
        for s in stmt.body:
            body_lines.extend(_stmt_to_latex_lines(s, loop_var=lv))
        return [header] + [f"\\quad {bl}" for bl in body_lines if bl]

    # 2. With block (e.g. `with numpyro.plate("i", N):`)
    if isinstance(stmt, ast.With):
        headers = [_format_with_header(item) for item in stmt.items]
        body_lines = []
        for s in stmt.body:
            body_lines.extend(_stmt_to_latex_lines(s, loop_var=loop_var))
        # Reverse independent sampling statements inside a plate so likelihood is above priors
        return headers + [f"\\quad {bl}" for bl in reversed(body_lines) if bl]

    # 3. Assignment (`lhs = rhs`)
    if isinstance(stmt, ast.Assign):
        lhs = _lhs_target_latex(stmt.targets[0])
        val = stmt.value

        # Ignore empty list initializers `z = []` used before a `for` loop
        if isinstance(val, ast.List) and len(val.elts) == 0:
            return []
        # Single-element list initializer `z = [z0]` -> `z[0] = z_{0}`
        if isinstance(val, ast.List) and len(val.elts) == 1 and isinstance(stmt.targets[0], ast.Name):
            return [f"{lhs}[0] = {ast_to_latex(val.elts[0])}"]

        if isinstance(val, ast.Call):
            if _is_deterministic_callee(val.func) and len(val.args) >= 2:
                return [f"{lhs} = {ast_to_latex(val.args[1])}"]
            if _is_dist_callee(val.func):
                func_name, args_str, obs_var, _ = _process_dist_call(val)
                lhs_final = obs_var if obs_var else lhs
                return [f"{lhs_final} \\sim \\text{{{func_name}}}({args_str})"]
            block_lines = _expand_bf_block_latex(lhs, val)
            if block_lines is not None:
                return block_lines

        return [f"{lhs} = {ast_to_latex(val)}"]

    # 4. Augmented assignment (`mu += b * x`)
    if isinstance(stmt, ast.AugAssign):
        lhs = _lhs_target_latex(stmt.target)
        rhs = ast_to_latex(stmt.value)
        op = {
            ast.Add: '+', ast.Sub: '-', ast.Mult: r'\cdot', ast.Div: '/'
        }.get(type(stmt.op), '+')
        return [f"{lhs} \\leftarrow {lhs} {op} {rhs}"]

    # 5. Expression statement (`m.dist.normal(..., obs=y)` or `z.append(expr)`)
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        if _is_deterministic_callee(call.func) and len(call.args) >= 2:
            sname = (format_latex_var(call.args[0].value)
                     if isinstance(call.args[0], ast.Constant)
                     else ast_to_latex(call.args[0]))
            return [f"{sname} = {ast_to_latex(call.args[1])}"]
        if _is_dist_callee(call.func):
            func_name, args_str, obs_var, name_var = _process_dist_call(call)
            target_var = obs_var or name_var
            if target_var is not None:
                return [f"{target_var} \\sim \\text{{{func_name}}}({args_str})"]
        # `z.append(expr)` inside a loop -> `z[t] = expr`
        if (isinstance(call.func, ast.Attribute)
                and call.func.attr == "append"
                and len(call.args) == 1):
            seq_name = ast_to_latex(call.func.value)
            idx = format_latex_var(loop_var) if loop_var else "t"
            return [f"{seq_name}[{idx}] = {ast_to_latex(call.args[0])}"]

    # Ignore docstrings or `pass`/`return`
    if isinstance(stmt, (ast.Pass, ast.Return)):
        return []
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
        return []

    return [convert_line_names(ast.unparse(stmt))]


def extract_latex_line_final(line):
    """Convert a single line of Python model code into LaTeX."""
    leading_spaces = len(line) - len(line.lstrip())
    stripped_line = line.lstrip()
    if not stripped_line or stripped_line.startswith("#"):
        return ""

    # Block headers (`with numpyro.plate("i", N):`, `for i in range(N):`)
    if stripped_line.endswith(":"):
        try:
            dummy_tree = ast.parse(stripped_line + "\n    pass")
            stmt = dummy_tree.body[0]
            if isinstance(stmt, ast.For):
                return " " * leading_spaces + _format_for_header(stmt.target, stmt.iter)
            if isinstance(stmt, ast.With):
                return " " * leading_spaces + _format_with_header(stmt.items[0])
        except Exception:
            pass
        header = stripped_line[:-1].strip()
        return " " * leading_spaces + f"\\text{{{_escape_text(header)}}}"

    try:
        tree = ast.parse(stripped_line)
        if not tree.body:
            return ""
        lines = _stmt_to_latex_lines(tree.body[0])
        if not lines:
            return ""
        return (" \\\\\n" + " " * leading_spaces).join(
            " " * leading_spaces + l for l in lines
        )
    except Exception:
        return " " * leading_spaces + convert_line_names(stripped_line)


def _body_lines(code):
    """Source lines of the model function BODY, indentation preserved."""
    code = textwrap.dedent(code)
    try:
        tree = ast.parse(code)
        func = next(n for n in ast.walk(tree)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    except Exception:
        return [l.lstrip() for l in extract_lines(code)][1:]

    src_lines = code.split("\n")
    body_stmts = list(func.body)
    if (body_stmts and isinstance(body_stmts[0], ast.Expr)
            and isinstance(body_stmts[0].value, ast.Constant)
            and isinstance(body_stmts[0].value.value, str)):
        body_stmts = body_stmts[1:]
    if not body_stmts:
        return []

    start = body_stmts[0].lineno - 1
    end = max(getattr(s, "end_lineno", s.lineno) for s in body_stmts)
    body = src_lines[start:end]

    return [l.rstrip() for l in textwrap.dedent("\n".join(body)).split("\n")
            if l.strip() and not l.strip().startswith("#")]


def _format_scalar_or_parents(param):
    """Format a traced distribution :class:`Param` for trace-based LaTeX."""
    if param.parents:
        p_sorted = [format_latex_var(p) for p in sorted(param.parents)]
        if len(p_sorted) == 1:
            return p_sorted[0]
        return f"f\\left({', '.join(p_sorted)}\\right)"
    arr = np.asarray(param.value)
    if arr.ndim == 0 or arr.size == 1:
        val = arr.item()
        if isinstance(val, (int, np.integer)):
            return str(int(val))
        if isinstance(val, (float, np.floating)):
            return f"{val:g}"
    return format_latex_var(param.name)


def _latex_from_trace(model, data):
    """Build LaTeX equations directly from ``trace_model(model, data)`` when
    Python source is unavailable (e.g., models defined in R or Julia)."""
    from BayesForge.PostModel.model_trace import trace_model, DETERMINISTIC

    tr = trace_model(model, data)
    lines = []
    for name, site in reversed(list(tr.sites.items())):
        lhs = format_latex_var(name)
        if site.kind == DETERMINISTIC:
            ps = [format_latex_var(p) for p in sorted(tr.parents.get(name, ()))]
            rhs = f"f\\left({', '.join(ps)}\\right)" if ps else r"\dots"
            lines.append(f"{lhs} = {rhs}")
        else:
            p_strs = [_format_scalar_or_parents(p) for p in site.params]
            lines.append(f"{lhs} \\sim \\text{{{_escape_text(site.dist)}}}({', '.join(p_strs)})")
    return lines


def to_latex(model, data=None):
    """Render a BF/NumPyro model as a LaTeX ``align*`` environment.

    Uses full AST statement & block translation for Python models (preserving
    loop/plate block ordering and expanding ``jnp.*``, ``m.link.*``,
    ``m.effects.*``, and ``m.net.*`` calls into mathematical notation), and
    falls back to execution-trace introspection (``trace_model``) when Python
    source is unavailable (e.g., R or Julia models).
    """
    latex_lines = None
    try:
        code = textwrap.dedent(inspect.getsource(model))
        tree = ast.parse(code)
        func = next(n for n in ast.walk(tree)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        stmt_groups = []
        for stmt in func.body:
            group = _stmt_to_latex_lines(stmt)
            if group:
                stmt_groups.append(group)
        # Reverse top-level statement groups (likelihood at top, priors at bottom),
        # while keeping lines within each group (e.g. a `for` loop header + body)
        # in top-down order.
        latex_lines = []
        for group in reversed(stmt_groups):
            latex_lines.extend(group)
    except Exception:
        try:
            code = inspect.getsource(model)
            lines_clean = _body_lines(code)
            latex_lines = list(reversed([extract_latex_line_final(line) for line in lines_clean]))
        except Exception:
            if data is not None:
                latex_lines = _latex_from_trace(model, data)
            else:
                raise

    latex_str = "\\begin{gather*}\n"
    latex_str += " \\\\\n".join(filter(None, latex_lines))
    latex_str += "\\\\\n"
    latex_str += "\\end{gather*}\n"
    display(Latex(latex_str))
    return latex_str
