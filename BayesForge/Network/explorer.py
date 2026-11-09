"""NetExplorer data-handling port — R ``NetExplorer`` package -> BayesForge style.

Upstream: https://github.com/SebastianSosa/Network-Explorer  (R/Functions.R)
Pinned commit: e4000d3cc2b6d48246f568c5c31b744e75674773

Same job as the R package's data layer: take a **data frame of node
characteristics** + a **square adjacency matrix**, turn them into the
``nodes`` / ``links`` JSON that the bundled d3.js front-end reads, and splice
that into ``patron1.txt`` + tooltip JS + ``patron2.txt`` to emit a standalone
``NetExplorer.html``.

Design choices for the port:
  * node table stays a ``pandas.DataFrame`` (the R ``df``);
  * every matrix operation (matrix -> edge list, min-max scaling, layer
    intra/inter flags) runs on ``jax.numpy`` arrays (the R ``m``);
  * colour ramps use matplotlib's ``LinearSegmentedColormap`` in place of
    ``grDevices::colorRampPalette``.

R function  ->  method
  mat.to.edgl            -> NetExplorer.mat_to_edgl
  df.col.findId          -> NetExplorer._col_id
  colorize              -> NetExplorer._colorize
  shape                 -> NetExplorer._shape_codes
  vis.net.format.att    -> NetExplorer.format_att
  vis.net               -> NetExplorer.vis_net / NetExplorer.__call__

Reached from the BF object as ``m.net.viz`` (see BayesForge.Network.Net):
``m.net.viz(df, adj, col_id="id", col_size="strength", ...)`` renders the
network; ``m.net.viz.mat_to_edgl(adj)`` etc. for the individual helpers.
"""

from __future__ import annotations

import base64
import html as _html
import mimetypes
import os
import platform
import shutil
import subprocess
import webbrowser
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex

from BayesForge.Network import layouts as _lay
from BayesForge.Network import vizfeatures as _vf


def _categorical_hex(n: int) -> list[str]:
    """``n`` visually distinct hex colours for chord groups."""
    try:
        import matplotlib as _mpl

        cmap = _mpl.colormaps["tab10" if n <= 10 else "tab20" if n <= 20 else "hsv"]
        if n <= 20:
            return [to_hex(cmap(i)) for i in range(n)]
        return [to_hex(cmap(i / max(n, 1))) for i in range(n)]
    except Exception:
        cycle = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                 "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]
        return [cycle[i % len(cycle)] for i in range(n)]

_ASSETS = Path(__file__).with_name("assets") / "netexplorer"


# --------------------------------------------------------------------- #
# feature layer spliced into the vendored front-end
# --------------------------------------------------------------------- #
_FEATURES_CSS = """
.main-menu .tool { margin: 3px 4px 0 0; padding: 6px 11px; border: 1px solid var(--line);
  border-radius: 8px; background: #f7f9fb; color: var(--ink); font: inherit; font-size: 12px;
  cursor: pointer; }
.main-menu .tool:hover { background: var(--accent); color: #fff; border-color: var(--accent); }
#nodeSearch, #sizeBy, #colorBy, #axisX, #axisY { width: 100%; margin-top: 4px; padding: 6px 8px;
  border: 1px solid var(--line); border-radius: 8px; background: #fff; font: inherit;
  font-size: 12.5px; color: var(--ink); }
#axisCaption { position: absolute; left: 14px; bottom: 14px; z-index: 40;
  font-size: 11px; color: var(--muted); background: var(--panel-bg);
  border: 1px solid var(--line); border-radius: 8px; padding: 4px 9px;
  box-shadow: 0 4px 16px rgba(20,30,50,.12); }
#legendBox, #statsBox { position: absolute; z-index: 40; background: var(--panel-bg);
  border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px;
  box-shadow: 0 6px 24px rgba(20,30,50,.14); font-size: 12px; color: var(--ink);
  max-width: 240px; }
#legendBox { right: 14px; bottom: 14px; }
#statsBox  { right: 14px; top: 14px; }
#legendBox h4, #statsBox h4 { margin: 0 0 6px; font-size: 10.5px; letter-spacing: .09em;
  text-transform: uppercase; color: var(--muted); }
#legendBox h4:not(:first-child) { margin-top: 10px; }
#legendBox .row { display: flex; align-items: center; gap: 7px; margin: 3px 0; }
#legendBox .lgd-col { margin: -3px 0 4px; font-size: 11px; color: var(--muted); }
#legendBox .sw { width: 13px; height: 13px; border-radius: 3px;
  border: 1px solid rgba(0,0,0,.15); flex: 0 0 auto; }
#legendBox .grad { width: 120px; height: 10px; border-radius: 3px; }
#statsBox table { border-collapse: collapse; }
#statsBox td { padding: 1px 8px 1px 0; }
#statsBox td:last-child { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }

body[data-theme="dark"] { --bg:#12161c; --panel-bg:#1b212b; --ink:#e6eaf0;
  --muted:#8b95a3; --line:#2b333f; }
body[data-theme="dark"] svg#net { background: #12161c; }
body[data-theme="dark"] input[type=number],
body[data-theme="dark"] #nodeSearch,
body[data-theme="dark"] #sizeBy, body[data-theme="dark"] #colorBy,
body[data-theme="dark"] #axisX, body[data-theme="dark"] #axisY,
body[data-theme="dark"] .main-menu .tool { background: #232b36; color: var(--ink); }
"""

# injected into the native "Links" section (after the Curves control)
_FEATURES_LINKS_HTML = """
        <li class="darkerli"><span class="nav-text">Min edge weight</span>
          <form><div><span>0</span>
            <div class="slider-wrapper">
              <input id="edgeThresh" type="range" min="0" max="1" step="0.001" value="0"/>
            </div>
            <span id="edgeThreshV" style="min-width: 32px; text-align: right;">0</span></div></form></li>
        <li class="darkerlishadowdown" id="liArrow"><span class="nav-text">Arrow size</span>
          <form><div><span>0</span>
            <div class="slider-wrapper">
              <input id="arrowSize" type="range" min="0" max="100" step="1" value="9"/>
            </div>
            <span id="arrowSizeV" style="min-width: 32px; text-align: right;">9</span></div></form></li>
"""

_FEATURES_HTML = """
      <li class="darkerlishadow"><span class="nav-text"><b>Explore</b></span></li>
      <li class="darkerli"><span class="nav-text">Find node</span>
        <input id="nodeSearch" type="text" placeholder="node id, then Enter" autocomplete="off"/></li>
      <li class="darkerli"><span class="nav-text">Size by</span>
        <select id="sizeBy"></select></li>
      <li class="darkerli"><span class="nav-text">Colour by</span>
        <select id="colorBy"></select></li>
      <li class="darkerli" id="liAxisX"><span class="nav-text">X axis</span>
        <select id="axisX"></select></li>
      <li class="darkerli" id="liAxisY"><span class="nav-text">Y axis</span>
        <select id="axisY"></select></li>
      <li class="darkerlishadow"><span class="nav-text"><b>View</b></span></li>
      <li class="darkerli">
        <button class="tool" id="btnFit" type="button">Fit</button>
        <button class="tool" id="btnTheme" type="button">Dark</button>
        <button class="tool" id="btnPNG" type="button">PNG</button>
        <button class="tool" id="btnJSON" type="button">JSON</button>
        <button class="tool" id="btnCSV" type="button">CSV</button>
      </li>
"""

_FEATURES_JS = r"""
(function () {
  if (typeof graph === 'undefined' || !graph) return;
  var G = graph;
  var svgEl = document.querySelector('svg#net') || document.querySelector('svg');
  var gSel = (typeof svg !== 'undefined') ? svg : d3.select(svgEl).select('g');
  var haveLink = (typeof link !== 'undefined');

  /* background -------------------------------------------------- */
  if (G.background && !(G.theme === 'dark' && G.background === '#e6e6e8')) {
    svgEl.style.background = G.background;   // dark + default keeps the CSS dark canvas
    var cw0 = document.getElementById('colorWell');
    if (cw0 && /^#[0-9a-fA-F]{6}$/.test(G.background)) cw0.value = G.background;
  }

  /* theme -------------------------------------------------------- */
  document.body.setAttribute('data-theme', G.theme === 'dark' ? 'dark' : 'light');
  var bt = document.getElementById('btnTheme');
  function themeLbl() { if (bt) bt.textContent =
    document.body.getAttribute('data-theme') === 'dark' ? 'Light' : 'Dark'; }
  themeLbl();
  if (bt) bt.onclick = function () {
    document.body.setAttribute('data-theme',
      document.body.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
    themeLbl();
  };

  /* directed arrowheads + size slider ------------------------ */
  if (G.directed && haveLink) {
    var mk = d3.select(svgEl).append('defs').append('marker')
      .attr('id', 'arrow').attr('viewBox', '0 -5 10 10').attr('refY', 0)
      .attr('markerUnits', 'userSpaceOnUse').attr('orient', 'auto');
    mk.append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', '#666');
    link.attr('marker-end', 'url(#arrow)');
    function setArrow(s) {
      // markerUnits=userSpaceOnUse: refX is in viewBox (0..10) units, its
      // pull-back in px scales by s/10 -> keep the tip a roughly constant
      // ~12px off the node whatever the head size (s=0 just hides it).
      mk.attr('markerWidth', s).attr('markerHeight', s)
        .attr('refX', s > 0 ? 10 + 120 / s : 10);
    }
    var as = document.getElementById('arrowSize'),
        asv = document.getElementById('arrowSizeV'),
        a0 = as ? +as.value : 9;
    setArrow(a0);
    if (asv) asv.textContent = a0;
    if (as) as.oninput = function () {
      var s = +this.value; setArrow(s); if (asv) asv.textContent = s;
    };
  } else {
    var liA = document.getElementById('liArrow');
    if (liA) liA.style.display = 'none';
  }

  /* stats panel ---------------------------------------------- */
  if (G.stats) {
    var s = G.stats, sb = document.createElement('div'); sb.id = 'statsBox';
    var rr = [['nodes', s.nodes], ['edges', s.edges], ['directed', s.directed ? 'yes' : 'no'],
      ['density', s.density], ['mean degree', s.mean_degree], ['components', s.components],
      ['diameter', s.diameter == null ? '—' : s.diameter],
      ['clustering', s.global_clustering == null ? '—' : s.global_clustering]];
    sb.innerHTML = '<h4>Network</h4><table>' +
      rr.map(function (r) { return '<tr><td>' + r[0] + '</td><td>' + r[1] + '</td></tr>'; }).join('') +
      '</table>';
    document.body.appendChild(sb);
  }

  /* legend ------------------------------------------------- */
  if (G.legend && G.legend.length) {
    var lb = document.createElement('div'); lb.id = 'legendBox'; var h = '';
    G.legend.forEach(function (e) {
      h += '<h4>' + e.title + '</h4>';
      if (e.col) h += '<div class="row lgd-col">' + e.col + '</div>';
      if (e.kind === 'swatches') {
        e.items.forEach(function (it) {
          h += '<div class="row"><span class="sw" style="background:' + it.color + '"></span>' + it.label + '</div>';
        });
      } else if (e.kind === 'gradient') {
        h += '<div class="row"><span class="grad" style="background:linear-gradient(90deg,' +
          e.stops[0] + ',' + e.stops[1] + ')"></span></div>' +
          '<div class="row"><span>' + e.lo + '</span><span style="flex:1"></span><span>' + e.hi + '</span></div>';
      } else if (e.kind === 'shapes') {
        e.items.forEach(function (it) {
          h += '<div class="row"><svg width="15" height="15"><path transform="translate(7.5,7.5)" d="' +
            d3.symbol().type(d3.symbols[it.shape]).size(70)() + '" fill="#8a94a3"></path></svg>' + it.label + '</div>';
        });
      } else if (e.kind === 'size') {
        h += '<div class="row"><span>' + e.lo + '</span>&nbsp;–&nbsp;<span>' + e.hi + '</span></div>';
      } else if (e.kind === 'note') {
        h += '<div class="row">' + e.text + '</div>';
      }
    });
    lb.innerHTML = h; document.body.appendChild(lb);
  }

  /* size-by / colour-by (centralities) ------------------- */
  var opts = ['(none)'].concat(G.metricNames || []);
  function fillSel(sel) { if (!sel) return;
    sel.innerHTML = opts.map(function (m) { return '<option>' + m + '</option>'; }).join(''); }
  var sizeBy = document.getElementById('sizeBy'), colorBy = document.getElementById('colorBy');
  fillSel(sizeBy); fillSel(colorBy);
  function span(k) {
    var v = G.nodes.map(function (d) { return d.cent ? d.cent[k] : 0; });
    var mn = Math.min.apply(null, v), mx = Math.max.apply(null, v);
    return [mn, mx > mn ? mx : mn + 1];
  }
  if (sizeBy) sizeBy.onchange = function () {
    var k = this.value;
    if (k === '(none)') {
      node.attr('d', d3.symbol().type(function (d) { return d3.symbols[d.shape]; })
        .size(function (d) { return d.size * 100; }));
      return;
    }
    var e = span(k);
    node.attr('d', d3.symbol().type(function (d) { return d3.symbols[d.shape]; })
      .size(function (d) { var t = ((d.cent ? d.cent[k] : 0) - e[0]) / (e[1] - e[0]); return 40 + t * 360; }));
  };
  if (colorBy) colorBy.onchange = function () {
    var k = this.value;
    if (k === '(none)') {
      node.style('fill', function (d) { return d.color; });
      return;
    }
    var e = span(k), sc = d3.interpolateRgb('#3b4cc0', '#b40426');
    node.style('fill', function (d) { var t = ((d.cent ? d.cent[k] : 0) - e[0]) / (e[1] - e[0]); return sc(t); });
  };

  /* scatter axes: pin nodes by any df covariate --------- */
  (function () {
    var axisCols = G.axisCols || [];
    var axisX = document.getElementById('axisX'), axisY = document.getElementById('axisY');
    if (!axisX || !axisY) return;
    if (!axisCols.length || typeof simulation === 'undefined') {
      ['liAxisX', 'liAxisY'].forEach(function (id) {
        var li = document.getElementById(id); if (li) li.style.display = 'none';
      });
      return;
    }
    var aopts = ['(none)'].concat(axisCols.map(function (c) { return c.name; }));
    [axisX, axisY].forEach(function (sel) {
      sel.innerHTML = aopts.map(function (m) { return '<option>' + m + '</option>'; }).join('');
    });
    if (G.axisX && aopts.indexOf(G.axisX) !== -1) axisX.value = G.axisX;
    if (G.axisY && aopts.indexOf(G.axisY) !== -1) axisY.value = G.axisY;

    var cap = document.createElement('div');
    cap.id = 'axisCaption'; cap.style.display = 'none';
    document.body.appendChild(cap);

    function scaler(name) {
      if (!name || name === '(none)') return null;
      var meta = axisCols.filter(function (c) { return c.name === name; })[0];
      if (!meta) return null;
      if (meta.type === 'cat') {
        var cats = meta.cats || [];
        return function (d) {
          var i = cats.indexOf(d.av ? String(d.av[name]) : null);
          // band centres, inset from the edges so labels don't clip
          return (cats.length && i >= 0) ? (i + 0.5) / cats.length : 0.5;
        };
      }
      var vs = G.nodes.map(function (d) { return d.av ? d.av[name] : null; })
        .filter(function (v) { return v != null && isFinite(v); });
      var mn = Math.min.apply(null, vs), mx = Math.max.apply(null, vs);
      return function (d) {
        var v = d.av ? d.av[name] : null;
        return (v == null || !isFinite(v) || mx === mn) ? 0.5 : (v - mn) / (mx - mn);
      };
    }

    function applyScatter() {
      var kx = axisX.value, ky = axisY.value;
      if ((kx === '(none)' || !kx) && (ky === '(none)' || !ky)) {
        window.ScatterMode = false;
        G.nodes.forEach(function (d) { d.fx = null; d.fy = null; });
        cap.style.display = 'none';
        simulation.alpha(0.5).restart();
        return;
      }
      window.ScatterMode = true;
      var navW = 250, pad = 60,
          S = Math.min(width - navW, height) - 2 * pad,
          ox = navW + (width - navW - S) / 2, oy = (height - S) / 2;
      var fx = scaler(kx), fy = scaler(ky);
      G.nodes.forEach(function (d) {
        var px = fx ? fx(d) : 0.5, py = fy ? fy(d) : 0.5;
        d.fx = d.x = ox + px * S;
        d.fy = d.y = oy + (1 - py) * S;   // larger value -> higher on screen
      });
      cap.textContent = 'x: ' + (kx === '(none)' ? '—' : kx) +
                        '   ·   y: ' + (ky === '(none)' ? '—' : ky);
      cap.style.display = 'block';
      simulation.alpha(0.3).restart();
    }
    axisX.onchange = applyScatter; axisY.onchange = applyScatter;
    window._applyScatter = applyScatter;
    if (axisX.value !== '(none)' || axisY.value !== '(none)') applyScatter();
  })();

  /* edge weight threshold ------------------------------- */
  var et = document.getElementById('edgeThresh'), etv = document.getElementById('edgeThreshV');
  if (et && haveLink && G.links.length) {
    var ws = G.links.map(function (l) { return l.weigth; });
    et.min = Math.min.apply(null, ws); et.max = Math.max.apply(null, ws);
    et.step = ((et.max - et.min) / 200) || 0.001; et.value = et.min;
    if (etv) etv.textContent = (+et.min).toFixed(2);
    et.oninput = function () {
      var t = +this.value; if (etv) etv.textContent = t.toFixed(2);
      link.style('display', function (l) { return l.weigth < t ? 'none' : null; });
    };
  }

  /* fit to view --------------------------------------- */
  var bf = document.getElementById('btnFit');
  if (bf) bf.onclick = function () {
    var xs = G.nodes.map(function (d) { return d.x; }).filter(isFinite),
        ys = G.nodes.map(function (d) { return d.y; }).filter(isFinite);
    if (!xs.length) return;
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs),
        y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var w = (x1 - x0) || 1, hh = (y1 - y0) || 1, pad = 65;
    var k = Math.min((width - 2 * pad) / w, (height - 2 * pad) / hh, 4);
    var tx = width / 2 - k * (x0 + x1) / 2, ty = height / 2 - k * (y0 + y1) / 2;
    gSel.transition().duration(400).attr('transform', 'translate(' + tx + ',' + ty + ') scale(' + k + ')');
  };

  /* find node ---------------------------------------- */
  var ns = document.getElementById('nodeSearch');
  if (ns) ns.onkeydown = function (ev) {
    if (ev.key !== 'Enter') return;
    var q = this.value.trim();
    var hit = G.nodes.filter(function (d) { return String(d.id) === q; })[0];
    if (!hit) return;
    var k = 2, tx = width / 2 - k * hit.x, ty = height / 2 - k * hit.y;
    gSel.transition().duration(400).attr('transform', 'translate(' + tx + ',' + ty + ') scale(' + k + ')');
    node.filter(function (d) { return d === hit; }).transition().duration(150)
      .style('stroke', '#ff3b30').style('stroke-width', 4).transition().duration(700)
      .style('stroke', function (d) { return d.strokeCol; })
      .style('stroke-width', function (d) { return d.strokeW; });
  };

  /* click node -> brighten neighbour labels --------- */
  if (typeof texts !== 'undefined' && haveLink) {
    var nb = {};
    G.links.forEach(function (l) {
      var a = (l.source && l.source.id !== undefined) ? l.source.id : l.source;
      var b = (l.target && l.target.id !== undefined) ? l.target.id : l.target;
      (nb[a] = nb[a] || {})[b] = 1; (nb[b] = nb[b] || {})[a] = 1;
    });
    var pick = null;
    node.on('click.labels', function (d) {
      pick = (pick === d.id) ? null : d.id;
      texts.style('font-weight', function (o) {
        return pick === null ? null : ((o.id === pick || (nb[pick] && nb[pick][o.id])) ? '700' : '400');
      }).style('fill-opacity', function (o) {
        return pick === null ? 1 : ((o.id === pick || (nb[pick] && nb[pick][o.id])) ? 1 : 0.12);
      }).style('font-size', function (o) {
        return pick === null ? null : ((o.id === pick || (nb[pick] && nb[pick][o.id])) ? '15px' : null);
      });
    });
  }

  /* exports ---------------------------------------- */
  function dl(name, blob) {
    var a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = name; document.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }
  var bJ = document.getElementById('btnJSON');
  if (bJ) bJ.onclick = function () {
    dl('network.json', new Blob([JSON.stringify({
      nodes: G.nodes.map(function (d) { return { id: d.id, x: d.x, y: d.y, color: d.color, size: d.size, cent: d.cent }; }),
      links: G.links.map(function (l) {
        return { source: (l.source.id || l.source), target: (l.target.id || l.target), weight: l.weigth };
      })
    }, null, 1)], { type: 'application/json' }));
  };
  var bC = document.getElementById('btnCSV');
  if (bC) bC.onclick = function () {
    var cols = ['id', 'x', 'y', 'size', 'color'].concat(G.metricNames || []);
    var rows = [cols.join(',')];
    G.nodes.forEach(function (d) {
      rows.push(cols.map(function (c) {
        var v = (c in d) ? d[c] : (d.cent && (c in d.cent) ? d.cent[c] : '');
        return (typeof v === 'string' && v.indexOf(',') >= 0) ? '"' + v + '"' : v;
      }).join(','));
    });
    dl('nodes.csv', new Blob([rows.join('\n')], { type: 'text/csv' }));
  };
  var bP = document.getElementById('btnPNG');
  if (bP) bP.onclick = function () {
    var W = svgEl.clientWidth || width, H = svgEl.clientHeight || height;
    var str = new XMLSerializer().serializeToString(svgEl);
    var img = new Image();
    img.onload = function () {
      var c = document.createElement('canvas'); c.width = W * 2; c.height = H * 2;
      var ctx = c.getContext('2d'); ctx.scale(2, 2);
      ctx.fillStyle = getComputedStyle(svgEl).backgroundColor || '#fff';
      ctx.fillRect(0, 0, W, H); ctx.drawImage(img, 0, 0, W, H);
      c.toBlob(function (b) { dl('network.png', b); });
    };
    img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(str)));
  };

  /* canvas renderer (auto for very large graphs, or canvas=True) ---- */
  if (G.canvas && haveLink && typeof simulation !== 'undefined') {
    var cv = document.createElement('canvas');
    cv.style.position = 'fixed'; cv.style.pointerEvents = 'none'; cv.style.zIndex = 5;
    document.body.appendChild(cv);
    function fit() {
      var r = svgEl.getBoundingClientRect();
      var W = r.width || svgEl.clientWidth || (window.innerWidth - 260);
      var H = r.height || svgEl.clientHeight || window.innerHeight;
      cv.style.left = (r.left || 260) + 'px'; cv.style.top = (r.top || 0) + 'px';
      cv.width = W; cv.height = H;
    }
    fit(); window.addEventListener('resize', fit);
    if (!cv.width || !cv.height) { cv.remove(); return; }  // keep SVG if sizing failed
    // canvas owns rendering — drop the SVG marks
    d3.select(svgEl).selectAll('path.node, path.link, text.label').remove();
    var cx = cv.getContext('2d'), last = '\0';
    function drawCanvas() {
      var tr = gSel.attr('transform') || '';
      var mm = /translate\(([-\d.]+)[ ,]([-\d.]+)\)(?:\s*scale\(([-\d.]+)\))?/.exec(tr);
      cx.setTransform(1, 0, 0, 1, 0, 0);
      cx.clearRect(0, 0, cv.width, cv.height);
      if (mm) { cx.translate(+mm[1], +mm[2]); if (mm[3]) cx.scale(+mm[3], +mm[3]); }
      cx.globalAlpha = 0.35;
      cx.strokeStyle = (document.body.getAttribute('data-theme') === 'dark') ? '#5a6472' : '#9aa4b2';
      cx.lineWidth = 0.4; cx.beginPath();
      G.links.forEach(function (l) {
        if (!l.source || l.source.x == null) return;
        cx.moveTo(l.source.x, l.source.y); cx.lineTo(l.target.x, l.target.y);
      });
      cx.stroke();
      cx.globalAlpha = 1;
      G.nodes.forEach(function (d) {
        cx.beginPath(); cx.fillStyle = d.color || '#4e79a7';
        cx.arc(d.x, d.y, Math.max(1.5, Math.sqrt(d.size || 1) * 1.3), 0, 6.2832); cx.fill();
      });
    }
    function mnmx(a) {
      var lo = Infinity, hi = -Infinity;
      for (var i = 0; i < a.length; i++) { var v = a[i];
        if (v < lo) lo = v; if (v > hi) hi = v; }
      return [lo, hi];
    }
    function autoFit() {
      var xs = G.nodes.map(function (d) { return d.x; }).filter(isFinite);
      var ys = G.nodes.map(function (d) { return d.y; }).filter(isFinite);
      if (!xs.length) return;
      var X = mnmx(xs), Y = mnmx(ys), pad = 30;
      var w = (X[1] - X[0]) || 1, h = (Y[1] - Y[0]) || 1;
      var k = Math.min(cv.width / (w + 2 * pad), cv.height / (h + 2 * pad), 2);
      var tx = cv.width / 2 - k * (X[0] + X[1]) / 2, ty = cv.height / 2 - k * (Y[0] + Y[1]) / 2;
      gSel.attr('transform', 'translate(' + tx + ',' + ty + ') scale(' + k + ')');
      drawCanvas();
    }
    simulation.on('tick.cv', drawCanvas);
    simulation.on('end.cv', autoFit);
    setTimeout(autoFit, 400); setTimeout(autoFit, 1800); setTimeout(autoFit, 4000);
    // redraw on pan / zoom without an open rAF loop
    var pending = false;
    function schedule() {
      if (pending) return;
      pending = true;
      requestAnimationFrame(function () { pending = false; drawCanvas(); });
    }
    svgEl.addEventListener('wheel', schedule, { passive: true });
    svgEl.addEventListener('pointermove', function (e) { if (e.buttons) schedule(); });
    svgEl.addEventListener('pointerup', schedule);
    drawCanvas();
  }

  /* resizable left panel (splitter drag) -------------------------- */
  var resizer = document.getElementById('menuResizer');
  var menu = document.querySelector('.main-menu');
  if (resizer && menu) {
    var isDragging = false;
    var startX = 0;
    var startW = 0;

    resizer.addEventListener('mousedown', function (e) {
      if (e.button !== 0) return;
      isDragging = true;
      startX = e.clientX;
      startW = menu.getBoundingClientRect().width;
      resizer.classList.add('is-resizing');
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      e.preventDefault();
    });

    window.addEventListener('mousemove', function (e) {
      if (!isDragging) return;
      var newW = Math.max(200, Math.min(startW + (e.clientX - startX), window.innerWidth * 0.85));
      document.documentElement.style.setProperty('--panel-w', newW + 'px');
      menu.style.flex = '0 0 ' + newW + 'px';
      menu.style.width = newW + 'px';
      if (typeof autoFit === 'function') autoFit();
    });

    window.addEventListener('mouseup', function () {
      if (isDragging) {
        isDragging = false;
        resizer.classList.remove('is-resizing');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      }
    });
  }

  /* slider value tooltips (floating bubbles) --------------------- */
  function formatBubbleVal(v) {
    var num = parseFloat(v);
    if (isNaN(num)) return v;
    if (Math.abs(num) >= 1000) {
      if (num % 1000 === 0) return (num / 1000) + 'k';
      return (num / 1000).toFixed(1) + 'k';
    }
    if (Number.isInteger(num)) return '' + num;
    var s = num.toFixed(3);
    return s.replace(/\.?0+$/, '');
  }

  var rangeSliders = document.querySelectorAll('input[type="range"]');
  rangeSliders.forEach(function (slider) {
    var bubble = document.createElement('div');
    bubble.className = 'slider-bubble';
    var parent = slider.parentElement;
    if (parent) {
      if (parent.style.position !== 'relative' && parent.style.position !== 'absolute') {
        parent.style.position = 'relative';
      }
      parent.appendChild(bubble);
    }

    function updateBubble() {
      bubble.textContent = formatBubbleVal(slider.value);
      var min = parseFloat(slider.min) || 0;
      var max = parseFloat(slider.max) || 100;
      var val = parseFloat(slider.value) || 0;
      var pct = max > min ? (val - min) / (max - min) : 0;
      // Compenser la largeur du pouce (15px thumb)
      var thumbW = 15;
      var sliderW = slider.offsetWidth || 100;
      var px = pct * (sliderW - thumbW) + (thumbW / 2);
      bubble.style.left = px + 'px';
    }

    function showBubble() {
      updateBubble();
      bubble.classList.add('visible');
    }

    function hideBubble() {
      bubble.classList.remove('visible');
    }

    slider.addEventListener('input', updateBubble);
    slider.addEventListener('focus', showBubble);
    slider.addEventListener('blur', hideBubble);
    slider.addEventListener('mouseenter', showBubble);
    slider.addEventListener('mouseleave', hideBubble);
    slider.addEventListener('mousedown', showBubble);
  });

  /* loading spinner dismissal ------------------------------------ */
  var loader = document.getElementById('netLoader');
  if (loader) {
    function hideLoader() {
      if (loader && !loader.classList.contains('is-hidden')) {
        loader.classList.add('is-hidden');
        setTimeout(function () {
          if (loader && loader.parentNode) loader.parentNode.removeChild(loader);
        }, 500);
      }
    }

    if (typeof simulation !== 'undefined') {
      // D3 force simulation: hide spinner when simulation ends, or on tick when alpha is low enough
      var tickCount = 0;
      simulation.on('tick.loader', function () {
        tickCount++;
        // Once the network has stabilized a few frames or reached low energy
        if (tickCount > 15 || simulation.alpha() < 0.1) {
          simulation.on('tick.loader', null);
          hideLoader();
        }
      });
      simulation.on('end.loader', hideLoader);
    }
    // Safety fallback: ensure spinner never stays visible indefinitely
    setTimeout(hideLoader, 1200);
  }
})();
"""


def _in_notebook() -> bool:
    """True inside an IPython/Jupyter kernel with a rich display frontend."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _is_wsl() -> bool:
    return "microsoft" in platform.uname().release.lower() or "WSL" in platform.uname().release


def _open_in_browser(path: Path) -> bool:
    """Open a local file in the OS default browser. Handles WSL by handing the
    path to the Windows shell."""
    path = path.resolve()
    if _is_wsl():
        try:
            win = subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
            subprocess.Popen(
                ["cmd.exe", "/c", "start", "", win],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            pass
    try:
        return webbrowser.open(path.as_uri())
    except Exception:
        return False


def _data_uri(p: Path) -> str:
    mt = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    return f"data:{mt};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


class NetworkView:
    """Return value of :meth:`NetExplorer.vis_net`.

    * as the last expression in a Jupyter cell it renders the network inline
      (``_repr_html_`` -> ``<iframe srcdoc>`` with the self-contained HTML);
    * ``.open()`` (or ``vis_net(..., browser=True)``) opens it in the OS
      browser;
    * ``.path`` is the written ``.html`` file.
    """

    def __init__(self, path, html: str, height: int = 900):
        self.path = Path(path)
        self.html = html
        self.height = height

    def open(self) -> "NetworkView":
        _open_in_browser(self.path)
        return self

    def _repr_html_(self) -> str:
        srcdoc = _html.escape(self.html, quote=True)
        return (
            f'<iframe srcdoc="{srcdoc}" '
            f'style="width:100%;height:{self.height}px;border:1px solid #bbb;'
            f'border-radius:6px;" sandbox="allow-scripts"></iframe>'
        )

    def __repr__(self) -> str:
        return f"NetworkView(path={self.path!s}, {len(self.html)} bytes)"

# d3.symbols order in the bundled d3 build -> integer code the front-end wants.
_SHAPE_CODE = {
    "circle": 0,
    "cross": 1,
    "losange": 2,  # diamond
    "diamond": 2,
    "rectangle": 3,  # square
    "square": 3,
    "star": 4,
    "triangle": 5,
    "y": 6,  # wye
    "wye": 6,
}


class NetExplorer:
    """Build the NetExplorer visualisation HTML from a node DataFrame + a JAX
    adjacency matrix.

    Parameters
    ----------
    ids
        Optional node labels. If ``None`` they are taken from ``col_id`` at
        :meth:`vis_net` time, and failing that generated as ``n1..nN`` from the
        matrix size (mirrors the R behaviour of falling back to
        ``colnames(m)``).
    assets_dir
        Folder holding ``d3.min.js``, ``d3-tip.js``, ``style.css``,
        ``patron1.txt``, ``patron2.txt``, ``logo.jpg``, ``save.png``.
        Defaults to the ``assets/`` folder next to this file.
    """

    def __init__(self, ids: Sequence | None = None, assets_dir: str | os.PathLike | None = None):
        self.ids = None if ids is None else [str(x) for x in ids]
        self.assets_dir = Path(assets_dir) if assets_dir is not None else _ASSETS
        missing = [
            f
            for f in ("patron1.txt", "patron2.txt", "d3.min.js", "d3-tip.js", "style.css")
            if not (self.assets_dir / f).exists()
        ]
        if missing:
            raise FileNotFoundError(f"assets missing in {self.assets_dir}: {missing}")

    # ------------------------------------------------------------------ #
    # matrix -> edge list   (R: mat.to.edgl)
    # ------------------------------------------------------------------ #
    def mat_to_edgl(
        self, M, sym: bool = False, erase_diag: bool = True, ids: Sequence | None = None
    ) -> pd.DataFrame:
        """Square adjacency matrix -> ``from`` / ``to`` / ``weight`` DataFrame.

        Parameters
        ----------
        M
            ``(N, N)`` JAX array (or anything ``jnp.asarray`` accepts).
        sym
            If ``True`` keep the lower triangle only (undirected).
        erase_diag
            Drop self-loops.
        ids
            Per-call node labels (length ``N``). Overrides the constructor
            ``ids`` for this call only; falls back to ``self.ids`` then to
            ``"0" .. "N-1"`` when omitted.

        The directed path reproduces R's ``as.vector(M)`` column-major
        unrolling exactly: edge ``k`` is ``(row = k % N, col = k // N)``.
        """
        M = jnp.asarray(M)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("M must be a square 2D array")
        N = int(M.shape[0])
        ids = self._resolve_ids(N, ids)

        if sym:
            rows, cols = jnp.tril_indices(N, k=-1 if erase_diag else 0)
            weight = np.asarray(M[rows, cols])
            rows, cols = np.asarray(rows), np.asarray(cols)
            frm = [ids[i] for i in cols]  # R: colnames(M)[tmp[,1]] (col index first)
            to = [ids[i] for i in rows]
        else:
            idx = jnp.arange(N * N)
            from_i = np.asarray(idx % N)
            to_i = np.asarray(idx // N)
            weight = np.asarray(M.T.reshape(-1))  # column-major == R as.vector(M)
            keep = np.ones(N * N, dtype=bool)
            if erase_diag:
                keep = from_i != to_i
            from_i, to_i, weight = from_i[keep], to_i[keep], weight[keep]
            frm = [ids[i] for i in from_i]
            to = [ids[i] for i in to_i]

        return pd.DataFrame({"from": frm, "to": to, "weight": np.asarray(weight, dtype=float)})

    # ------------------------------------------------------------------ #
    # helpers  (R: df.col.findId / colorize / shape)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _col_id(df: pd.DataFrame, label) -> str:
        """Resolve a column name or 0-based index to a column name."""
        if isinstance(label, (int, np.integer)) and not isinstance(label, bool):
            if label < 0 or label >= df.shape[1]:
                raise IndexError(f"column index {label} out of bounds")
            return df.columns[int(label)]
        if label in df.columns:
            return label
        raise KeyError(f"'{label}' does not match any column of df: {list(df.columns)}")

    @staticmethod
    def _colorize(df: pd.DataFrame, col: str, colors: Sequence[str], new_col: str) -> pd.DataFrame:
        """Append a gradient hex column keyed on ``col``.

        Two mappings, chosen by dtype:

        * **numeric** ``col`` -> linear interpolation by *value* across
          ``[min, max]`` of the finite entries, so the fill matches the
          continuous gradient legend. Non-finite cells (NaN / ``±inf``) get
          ``NaN`` in ``new_col`` (the front-end then skips the channel), never
          a spurious ``colors[0]``.
        * **categorical** ``col`` -> one ramp stop per *distinct* value, by
          ascending rank (smallest -> ``colors[0]``); legend shows discrete
          swatches. Unknown / missing values map to ``NaN``.

        Unlike R ``colorize``, the frame is **not** re-sorted — rows stay
        aligned to the adjacency matrix; the ramp is a value->hex map so row
        order is irrelevant.
        """
        d = df.copy()
        cmap = LinearSegmentedColormap.from_list("netexp", list(colors))

        # Numeric column: interpolate by VALUE position across [min, max] so the
        # fill matches the linear gradient legend. Rank-based stops misplace a
        # skewed column (e.g. [0, 1, 2, 3, 100] -> value 3 drawn ~75% up the
        # ramp while the legend scale reads ~3%).
        if pd.api.types.is_numeric_dtype(d[col]):
            v = d[col].to_numpy(dtype=float)
            ok = np.isfinite(v)
            hexes = np.full(v.shape, np.nan, dtype=object)  # non-finite -> NaN
            if ok.any():
                lo, hi = v[ok].min(), v[ok].max()
                span = hi - lo
                frac = np.zeros(int(ok.sum())) if span == 0 else (v[ok] - lo) / span
                frac = np.clip(frac, 0.0, 1.0)
                hexes[ok] = [to_hex(cmap(float(f))) for f in frac]
            d[new_col] = hexes
            return d

        # Categorical column: one ramp stop per distinct value, by ascending
        # rank (smallest -> colors[0]); legend shows discrete swatches.
        try:
            levels = sorted(pd.unique(d[col]).tolist())
        except TypeError:
            levels = sorted(pd.unique(d[col]).tolist(), key=str)
        if len(levels) == 1:
            ramp = {levels[0]: to_hex(cmap(0.0))}
        else:
            ramp = {lv: to_hex(cmap(i / (len(levels) - 1))) for i, lv in enumerate(levels)}
        d[new_col] = d[col].map(ramp)
        return d

    @staticmethod
    def _gradient_stops(color: Sequence[str] | None, palette: str) -> list:
        """Gradient endpoints for ``_colorize``: the caller's ``color`` when it
        gives at least two stops, else the palette's sequential ramp."""
        if color is not None and not isinstance(color, str) and len(color) >= 2:
            return list(color)
        return list(_vf.palette(palette)["sequential"])

    @staticmethod
    def _shape_codes(vec, chars: Sequence[str]) -> np.ndarray:
        """Categorical vector -> d3 symbol codes via a per-category shape name."""
        vec = pd.Series(vec)
        cats = sorted(pd.unique(vec).tolist(), key=str)
        if len(cats) > 7:
            raise ValueError("at most 7 node shapes are available")
        if len(cats) != len(chars):
            raise ValueError(
                f"`shapes` has {len(chars)} entries but the column has {len(cats)} categories"
            )
        name_by_cat = dict(zip(cats, chars))
        try:
            return np.array([_SHAPE_CODE[name_by_cat[v].lower()] for v in vec], dtype=int)
        except KeyError as e:
            raise ValueError(f"unsupported shape name {e!s}; pick from {sorted(_SHAPE_CODE)}")

    # ------------------------------------------------------------------ #
    # node attribute formatting  (R: vis.net.format.att)
    # ------------------------------------------------------------------ #
    def format_att(
        self,
        df: pd.DataFrame,
        col_id=None,
        col_size=None,
        color_node: Sequence[str] | None = None,
        col_color=None,
        shapes: Sequence[str] | None = None,
        col_shape=None,
        color_stroke: Sequence[str] | str | None = None,
        col_strokeColor=None,
        col_stroke=None,
        node_opacity=None,
        palette: str = "default",
    ):
        """Normalise an arbitrary node table into the fixed column set the
        front-end reads: ``id, size, color, strokeCol, strokeW, shape,
        opacity`` plus the ``*Value`` columns used for tooltips.

        Returns ``(df2, ori)`` where ``ori`` is the list of the seven source
        column names ``[id, size, color, strokeCol, stroke, shape, opacity]``
        (``None`` where not supplied), used later for tooltip labels.

        ``color_node`` is the node-fill gradient, ``color_stroke`` the border
        gradient (or a single colour) and ``col_strokeColor`` the border-colour
        column. Supplying ``color_stroke`` or ``col_strokeColor`` turns borders
        on for every node at width 1 unless ``col_stroke`` gives an explicit
        width column.
        """
        d = df.copy()
        # `color_node` / `color_stroke` are pure colour ramps and never need an
        # id column — only per-node channels that would otherwise be resolved
        # positionally do. Guarding on those lets `vis_net(df, adj)` with no
        # styling args work id-free (the positional id fallback below applies).
        if col_id is None and any(
            x is not None for x in (col_shape, node_opacity)
        ):
            raise ValueError(
                "col_id cannot be None when col_shape / node_opacity is set"
            )

        # --- opacity -------------------------------------------------- #
        if node_opacity is not None:
            c = self._col_id(d, node_opacity)
            if not pd.api.types.is_numeric_dtype(d[c]):
                raise TypeError("node_opacity column must be numeric")
            ori_opacity = c
            v = jnp.asarray(d[c].to_numpy(dtype=float))
            v = (v - v.min()) / (v.max() - v.min())
            v = np.array(v)  # writable copy off the JAX buffer
            if len(v):
                v[np.argmin(v)] = 1e-3
            d["opacity"] = v
        else:
            ori_opacity = None
            d["opacity"] = 1.0

        # --- size --------------------------------------------------- #
        if col_size is not None:
            c = self._col_id(d, col_size)
            if not pd.api.types.is_numeric_dtype(d[c]):
                raise TypeError("col_size column must be numeric")
            ori_size = c
            s = d[c].to_numpy(dtype=float)
            fin = np.isfinite(s)
            if fin.any() and s[fin].min() <= 0:
                lo, hi = float(s[fin].min()), float(s[fin].max())
                s = np.where(fin, 1.0 + 4.0 * (s - lo) / (hi - lo if hi > lo else 1.0), 1.0)
            d["size"] = s
        else:
            ori_size = None
            d["size"] = 1.0

        # --- id ---------------------------------------------------- #
        if col_id is not None:
            c = self._col_id(d, col_id)
            ori_id = c
            d["id"] = d[c].astype(str)
        else:
            ori_id = None
            d["id"] = [str(i + 1) for i in range(len(d))]

        # --- shape ------------------------------------------------- #
        if col_shape is not None:
            c = self._col_id(d, col_shape)
            ori_shape = c
            if shapes is not None:
                d["shape"] = self._shape_codes(d[c], shapes)
            else:
                d["shape"] = pd.factorize(d[c], sort=True)[0]
        else:
            ori_shape = None
            d["shape"] = 0

        # --- stroke width + stroke colour ------------------------- #
        # Declaring `color_stroke` or `col_strokeColor` turns borders on for
        # every node: width 1 unless `col_stroke` gives an explicit width column.
        stroke_on = (
            col_stroke is not None
            or col_strokeColor is not None
            or color_stroke is not None
        )
        if col_stroke is not None:
            c = self._col_id(d, col_stroke)
            if not pd.api.types.is_numeric_dtype(d[c]):
                raise TypeError("col_stroke column must be numeric")
            ori_stroke = c
            d["strokeW"] = d[c].astype(float)
        elif stroke_on:
            ori_stroke = None
            d["strokeW"] = 1.0
        else:
            ori_stroke = None
            d["strokeW"] = 0.0

        if col_strokeColor is not None:
            cc = self._col_id(d, col_strokeColor)
            ori_strokeCol = cc
            stops = self._gradient_stops(color_stroke, palette)
            d = self._colorize(d, cc, stops, new_col="strokeCol")
        elif stroke_on:
            ori_strokeCol = None
            d["strokeCol"] = (
                color_stroke if isinstance(color_stroke, str)
                else (color_stroke[-1] if color_stroke else "black")
            )
        else:
            ori_strokeCol = None
            d["strokeCol"] = np.nan

        # --- node colour ----------------------------------------- #
        # numeric column  -> continuous gradient over `color`
        # non-numeric     -> `color` gradient sampled at k equidistant stops,
        #                    one per group (ascending order); legend shows swatches
        self._color_is_categorical = False
        if col_color is not None:
            cc = self._col_id(d, col_color)
            ori_color = cc
            stops = self._gradient_stops(color_node, palette)
            d = self._colorize(d, cc, stops, new_col="color")
            if not pd.api.types.is_numeric_dtype(d[cc]):
                self._color_is_categorical = True
        else:
            ori_color = None
            d["color"] = "black"

        # --- tooltip value columns ------------------------------- #
        for vcol, src in (
            ("sizeValue", ori_size),
            ("colorValue", ori_color),
            ("strokeColValue", ori_strokeCol),
            ("shapeValue", ori_shape),
        ):
            if src is not None:
                d[vcol] = d[src]
        d["strokeWValue"] = d["strokeW"]
        if ori_opacity is not None:
            d["opacityValue"] = d[ori_opacity]

        ori = [ori_id, ori_size, ori_color, ori_strokeCol, ori_stroke, ori_shape, ori_opacity]
        return d, ori

    # ------------------------------------------------------------------ #
    # full pipeline  (R: vis.net)
    # ------------------------------------------------------------------ #
    def vis_net(
        self,
        df: pd.DataFrame | None = None,
        m=None,
        col_id=None,
        col_size=None,
        color_node: Sequence[str] = ("black", "white"),
        col_color=None,
        col_shape=None,
        shapes: Sequence[str] | None = None,
        color_stroke: Sequence[str] | str | None = None,
        col_strokeColor=None,
        col_stroke=None,
        layers=None,
        node_opacity=None,
        link_opacity: bool = False,
        background: str = "grey",
        layout: str = "force",
        layout_col=None,
        layout_x=None,
        layout_y=None,
        axis_x=None,
        axis_y=None,
        directed: bool = True,
        metrics: bool = True,
        edge_color_col=None,
        weight_posterior=None,
        palette: str = "default",
        theme: str = "light",
        canvas: bool | None = None,
        out_dir: str | os.PathLike = "out",
        filename: str = "NetExplorer.html",
        inline: bool = True,
        browser: bool | None = None,
        height: int = 900,
    ) -> "NetworkView":
        """Write ``NetExplorer.html`` and return a :class:`NetworkView`.

        ``df`` is the node characteristics table, ``m`` a ``(N, N)`` JAX
        adjacency array. All styling arguments mirror the R ``vis.net`` names.

        Parameters
        ----------
        df
            Node characteristics. If ``None`` a placeholder frame of one
            all-ones column is built and ``col_id`` defaults to ``0`` — so
            ``m.net.viz(m=adj)`` works with no node table. (Node ids then fall
            back to ``n1..nN`` since that column is not unique.)
        inline
            Embed d3 / CSS / images into the HTML so the single file is
            portable (needed for Jupyter inline rendering). When ``False`` the
            assets are copied next to the file instead.
        browser
            Open the file in the OS default browser (WSL-aware). Default:
            ``True`` outside a notebook, ``False`` inside one (the cell renders
            it inline anyway — call ``.open()`` on the result to also pop a
            browser).
        layout
            Initial layout. ``force`` / ``circle`` / ``linear`` / ``multilayer``
            run live in the browser (unchanged). ``clustered`` / ``spectral`` /
            ``mds`` / ``radial`` / ``arc`` / ``layered`` / ``geo`` are computed
            here (see :mod:`BayesForge.Network.layouts`) and the page pins nodes
            to the result; ``chord`` draws a d3 ribbon chord of between-group
            flows. The dropdown offers the four live layouts plus whichever one
            you picked here.
        layout_col
            Column driving the chosen layout: the nodal metric for ``radial``
            (default: degree) and ``arc``, the grouping for ``clustered`` and
            ``chord`` (required for ``chord``).
        layout_x, layout_y
            Columns of fixed coordinates for ``layout="geo"`` (e.g. lon / lat).
        axis_x, axis_y
            Names of ``df`` columns to pin nodes on an invisible scatter grid
            (numeric -> min-max scaled, categorical -> evenly banded, larger =
            higher). Just the initial selection: the page exposes an **X axis**
            / **Y axis** dropdown over every usable ``df`` covariate, and
            ``(none)`` on both releases the pins back to the current layout.
        color_node, col_color
            Node-fill gradient and the column that drives it. A numeric column
            gives a continuous gradient over ``color_node``; a non-numeric one
            gives that ramp sampled at one equidistant stop per group.
        color_stroke, col_strokeColor, col_stroke
            Node border. ``col_strokeColor`` colours borders from a column
            (ramp = ``color_stroke``, else the palette sequential ramp);
            ``color_stroke`` alone (a 2-tuple or a single colour) paints every
            border one colour. Declaring either of those switches borders on at
            width 1 for all nodes unless ``col_stroke`` supplies a numeric
            width column.
        directed
            Treat the matrix as directed: draw arrowheads on links, and for
            ``layout="layered"`` use topological generations when acyclic.
        metrics
            Compute per-node centralities (degree / strength / eigenvector /
            betweenness / clustering, via ``m.net.met`` only) and a global
            summary. Drives the Size-by / Colour-by dropdowns and the Stats
            panel in the page.
        edge_color_col
            Column of ``df`` used to colour links (via the source node);
            otherwise links take the source node's colour.
        weight_posterior
            ``(draws, N, N)`` posterior edge-weight array. The posterior mean
            becomes the plotted network; each link also carries P(weight > 0)
            (shown as link opacity) and a 90% interval (in the tooltip).
        palette
            ``"default"`` or ``"cb"`` (Okabe-Ito categorical + viridis
            sequential, colour-blind safe).
        theme
            ``"light"`` or ``"dark"`` — initial page theme (toggle in the page).
        height
            Inline iframe height in px for the notebook view.
        """
        if browser is None:
            browser = not _in_notebook()

        # Convenience: viz(adj) with no node table — first positional is the
        # adjacency matrix, not a DataFrame.
        if m is None and df is not None and not isinstance(df, pd.DataFrame) \
                and getattr(df, "ndim", None) == 2 and df.shape[0] == df.shape[1]:
            m, df = df, None

        if m is None:
            raise ValueError("vis_net needs an adjacency matrix `m`")

        edge_prob = edge_lo = edge_hi = None
        if weight_posterior is not None:
            mean, edge_prob, edge_lo, edge_hi = _vf.edge_posterior_summary(weight_posterior)
            m = mean
        m = jnp.asarray(m)
        N = int(m.shape[0])

        if df is None:
            df = pd.DataFrame(np.ones((N, 1)))
            if col_id is None:
                col_id = 0

        d, ori = self.format_att(
            df,
            col_id=col_id,
            col_size=col_size,
            color_node=color_node,
            col_color=col_color,
            shapes=shapes,
            col_shape=col_shape,
            color_stroke=color_stroke,
            col_strokeColor=col_strokeColor,
            col_stroke=col_stroke,
            node_opacity=node_opacity,
            palette=palette,
        )
        # node labels: prefer col_id, then ctor ids, then an n1..nN fallback.
        # Kept in a local (not self.ids) and handed to mat_to_edgl below so the
        # edge labels match — without leaking this call's node table into the
        # next vis_net / mat_to_edgl call on the same NetExplorer instance.
        if ori[0] is not None:
            node_ids = d["id"].astype(str).tolist()
        else:
            node_ids = self._resolve_ids(N)
            d["id"] = list(node_ids)
        # ids must be unique for the d3 force graph — a constant/placeholder id
        # column (e.g. df=None -> ones) collapses every node onto one point and
        # hides the links. Fall back to positional labels.
        if pd.Index(d["id"]).duplicated().any():
            import warnings as _warnings

            _warnings.warn(
                "node id column is not unique; using positional ids n1..nN",
                stacklevel=2,
            )
            d["id"] = [f"n{i + 1}" for i in range(len(d))]
            node_ids = list(d["id"])

        # layer index per node
        if layers is not None:
            lc = self._col_id(d, layers)
            d["layers"] = pd.factorize(d[lc], sort=True)[0] + 1
        else:
            d["layers"] = 1

        # edge list from the matrix (JAX), drop zero links
        edgl = self.mat_to_edgl(m, ids=node_ids)
        edgl = edgl[edgl["weight"] != 0].reset_index(drop=True)

        # link opacity via min-max on the JAX weight vector
        if link_opacity and len(edgl):
            w = jnp.asarray(edgl["weight"].to_numpy())
            lo = (w - w.min()) / (w.max() - w.min())
            lo = np.array(lo)  # writable copy off the JAX buffer
            lo[np.argmin(lo)] = 1e-4
            edgl["lOpacity"] = lo
        else:
            edgl["lOpacity"] = 1.0

        # carry the source node's colour onto each edge (R merges 'color')
        color_by_id = dict(zip(d["id"], d["color"]))
        edgl["colorL"] = edgl["from"].map(color_by_id)

        # intra / inter layer flags (JAX equality on the layer vectors)
        if layers is not None:
            layer_by_id = dict(zip(d["id"], d["layers"]))
            src_layer = jnp.asarray([layer_by_id[x] for x in edgl["from"]])
            tgt_layer = jnp.asarray([layer_by_id[x] for x in edgl["to"]])
            same = np.asarray(src_layer == tgt_layer)
            edgl["intralayer"] = np.where(same, 1.0, np.nan)
            edgl["interlayer"] = np.where(same, np.nan, 1.0)
        else:
            edgl["intralayer"] = np.nan
            edgl["interlayer"] = np.nan

        adj_np = np.asarray(m, dtype=float)
        idx_by_id = {v: i for i, v in enumerate(d["id"])}
        si = edgl["from"].map(idx_by_id).to_numpy()
        ti = edgl["to"].map(idx_by_id).to_numpy()

        # posterior edge uncertainty: P(w>0) -> link opacity, 90% CI -> tooltip
        if edge_prob is not None and len(edgl):
            edgl["eprob"] = edge_prob[si, ti]
            edgl["elo"] = edge_lo[si, ti]
            edgl["ehi"] = edge_hi[si, ti]
            edgl["lOpacity"] = np.clip(edgl["eprob"].to_numpy(), 0.03, 1.0)

        # link colour from an edge attribute (via the source node's row):
        # numeric -> gradient, non-numeric -> gradient sampled per group
        if edge_color_col is not None:
            ec = self._col_id(d, edge_color_col)
            d = self._colorize(d, ec, _vf.palette(palette)["sequential"], "_edgeCol")
            edgl["colorL"] = edgl["from"].map(dict(zip(d["id"], d["_edgeCol"])))

        # per-node centralities + global summary (met only)
        node_metrics: dict[str, np.ndarray] = {}
        stats = None
        if metrics:
            try:
                node_metrics = _vf.centralities(adj_np, directed=directed)
            except Exception:
                node_metrics = {}
            try:
                stats = _vf.network_stats(adj_np)
            except Exception:
                stats = None

        # layouts: precompute coordinates for every applicable non-live layout
        # so the GUI dropdown can switch between them client-side. Big graphs
        # (N > 800) only get the one that was asked for.
        layout = (layout or "force").lower()
        grp_vals = (
            d[self._col_id(d, layout_col)].to_numpy() if layout_col is not None else None
        )
        geo_xy = None
        if layout_x is not None and layout_y is not None:
            geo_xy = (
                d[self._col_id(d, layout_x)].to_numpy(),
                d[self._col_id(d, layout_y)].to_numpy(),
            )

        wanted = _lay.PINNED if N <= 800 else (
            (layout,) if layout in _lay.PINNED else ()
        )
        all_pos: dict[str, tuple] = {}
        for name in wanted:
            try:
                if name == "geo":
                    if geo_xy is None:
                        continue
                    lx, ly = _lay.compute("geo", adj_np, x=geo_xy[0], y=geo_xy[1])
                elif name == "clustered":
                    lx, ly = _lay.compute("clustered", adj_np, groups=grp_vals)
                else:  # spectral / mds / radial / arc / layered — self-sufficient
                    lx, ly = _lay.compute(name, adj_np, directed=directed)
                all_pos[name] = (np.asarray(lx, float), np.asarray(ly, float))
            except Exception:
                continue
        if layout in _lay.PINNED and layout not in all_pos:
            raise ValueError(
                f"layout={layout!r} could not be computed"
                + (" (needs layout_x / layout_y)" if layout == "geo" else "")
            )

        chord_payload = None
        if grp_vals is not None:
            mat, labels = _lay.chord_matrix(grp_vals, adj_np)
            chord_payload = {
                "matrix": mat.tolist(),
                "labels": [str(x) for x in labels],
                "colors": _categorical_hex(len(labels)),
            }
        elif layout == "chord":
            raise ValueError("layout='chord' needs layout_col (the node grouping)")

        # legend entries for whatever encodings are actually in use
        legend = self._build_legend(d, ori, color_node, palette, edge_prob is not None)

        # invisible scatter axes: every usable df covariate becomes an X/Y option
        axis_cols, axis_vals = self._axis_cols(df, col_id)
        names = {c["name"] for c in axis_cols}
        ax_x = str(axis_x) if axis_x is not None and str(axis_x) in names else "(none)"
        ax_y = str(axis_y) if axis_y is not None and str(axis_y) in names else "(none)"

        use_canvas = (N > 1500) if canvas is None else bool(canvas)

        # assemble + write
        html = self._assemble_html(
            d, edgl, ori, layout=layout, chord=chord_payload, all_pos=all_pos,
            stats=stats, node_metrics=node_metrics, legend=legend,
            theme=theme, palette=palette, directed=bool(directed),
            has_posterior=edge_prob is not None, canvas=use_canvas,
            background=background,
            axis_cols=axis_cols, axis_vals=axis_vals, axis_x=ax_x, axis_y=ax_y,
        )
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if inline:
            html = self._inline_assets(html)
        else:
            for f in ("d3.min.js", "d3-tip.js", "style.css", "logo.jpg", "save.png"):
                src = self.assets_dir / f
                if src.exists():
                    shutil.copy(src, out_dir / f)
        target = out_dir / filename
        target.write_text(html, encoding="utf-8")

        view = NetworkView(target, html, height=height)
        if browser:
            _open_in_browser(target)
        # In a notebook the returned view renders itself once via _repr_html_
        # when it is the last expression in the cell. Do NOT also display() it
        # here — that draws the network twice. If the call is assigned
        # (`v = m.net.viz(...)`), evaluate `v` on its own line to show it.
        return view

    # `m.net.viz(df, adj, ...)` is the common path -> alias __call__ to vis_net
    __call__ = vis_net

    def _inline_assets(self, html: str) -> str:
        """Fold ``d3.min.js`` / ``d3-tip.js`` / ``style.css`` / images into the
        HTML so it is a single portable file."""
        d3 = (self.assets_dir / "d3.min.js").read_text(encoding="utf-8")
        tip = (self.assets_dir / "d3-tip.js").read_text(encoding="utf-8")
        css = (self.assets_dir / "style.css").read_text(encoding="utf-8")
        html = html.replace(
            '<link rel="stylesheet" type="text/css" href="style.css">',
            f"<style>\n{css}\n</style>",
        )
        html = html.replace("<script src='d3.min.js'></script>", f"<script>\n{d3}\n</script>")
        html = html.replace("<script src='d3-tip.js'></script>", f"<script>\n{tip}\n</script>")
        for img in ("logo.jpg", "save.png"):
            p = self.assets_dir / img
            if p.exists():
                html = html.replace(f'src="{img}"', f'src="{_data_uri(p)}"')
        return html

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _resolve_ids(self, N: int, ids: Sequence | None = None) -> list[str]:
        # Never mutate call state: ``ids`` (this call) wins, else the
        # ctor-supplied labels, else positional ``n1..nN``. Nothing is cached
        # onto ``self``, so a later call with a different-sized matrix or a
        # different node table is unaffected by this one.
        src = ids if ids is not None else self.ids
        if src is None:
            return [f"n{i + 1}" for i in range(N)]
        src = [str(x) for x in src]
        if len(src) != N:
            raise ValueError(f"ids has length {len(src)} but matrix is {N}x{N}")
        return src

    @staticmethod
    def _js_val(v) -> str:
        """Python scalar -> JS literal (numbers bare, NaN -> NaN, else quoted)."""
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            f = float(v)
            if not np.isfinite(f):        # NaN / inf / -inf -> JS has no literal
                return "NaN"
            return repr(int(f)) if f.is_integer() else repr(f)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "NaN"
        return "'" + str(v).replace("'", "\\'") + "'"

    def _tooltip_js(self, ori) -> str:
        """The ``tooltip.html(...)`` argument list appended after patron1."""
        names = dict(zip(("id", "size", "color", "strokeCol", "stroke", "shape", "opacity"), ori))
        s = "\n'id : '+ d.id "
        if names["size"] is not None:
            s += f"+ '<p/>{names['size']} (size) : '+ d.sizeValue "
        if names["color"] is not None:
            s += f"+ '<p/>{names['color']} (color) : '+ d.colorValue "
        if names["strokeCol"] is not None:
            s += f"+ '<p/>{names['strokeCol']} (stroke color) : '+ d.strokeColValue "
        if names["stroke"] is not None:
            s += f"+ '<p/>{names['stroke']} (stroke width) : '+ d.strokeWValue "
        if names["shape"] is not None:
            s += f"+ '<p/>{names['shape']} (shape) : '+ d.shapeValue \n"
        s += (
            ").style( 'left' ,(d3.event.pageX) + 'px')\n"
            ".style( 'top' ,(d3.event.pageY) + 'px');\n})\n"
        )
        return s

    def _axis_cols(self, df: pd.DataFrame, col_id) -> tuple[list, dict]:
        """Every usable ``df`` covariate -> a scatter-axis option.

        Returns ``(meta, values)``: ``meta`` a list of
        ``{"name", "type": "num"|"cat", "cats": [...]}`` and ``values`` a
        ``name -> [per-node value]`` map aligned to the node (matrix) order.
        Columns with < 2 distinct non-null values (e.g. the ``df=None`` ones
        placeholder) are skipped.
        """
        try:
            idc = self._col_id(df, col_id) if col_id is not None else None
        except Exception:
            idc = None
        meta: list = []
        values: dict = {}
        for c in df.columns:
            if c == idc:
                continue
            s = df[c]
            if s.nunique(dropna=True) < 2:
                continue
            name = str(c)
            if pd.api.types.is_numeric_dtype(s):
                meta.append({"name": name, "type": "num"})
                values[name] = [None if pd.isna(v) else float(v) for v in s]
            else:
                cats = sorted({str(v) for v in s.dropna().unique()})
                meta.append({"name": name, "type": "cat", "cats": cats})
                values[name] = [None if pd.isna(v) else str(v) for v in s]
        return meta, values

    def _nodes_json(self, d: pd.DataFrame, ori, all_pos: dict | None = None,
                    node_metrics: dict | None = None,
                    axis_vals: dict | None = None) -> str:
        import json as _json

        has = dict(zip(("id", "size", "color", "strokeCol", "stroke", "shape", "opacity"), ori))
        all_pos = all_pos or {}
        node_metrics = node_metrics or {}
        axis_vals = axis_vals or {}
        rows = []
        for i, (_, r) in enumerate(d.iterrows()):
            parts = [
                f"'id':{self._js_val(r['id'])}",
                f"'size':{self._js_val(r['size'])}",
                f"'color':{self._js_val(r['color'])}",
                f"'strokeCol':{self._js_val(r['strokeCol'])}",
                f"'strokeW':{self._js_val(r['strokeW'])}",
                f"'shape':{self._js_val(r['shape'])}",
                f"'opacity':{self._js_val(r['opacity'])}",
                f"'layers':{self._js_val(r['layers'])}",
            ]
            if has["size"] is not None:
                parts.append(f"'sizeValue':{self._js_val(r['sizeValue'])}")
            if has["color"] is not None:
                parts.append(f"'colorValue':{self._js_val(r['colorValue'])}")
            if has["strokeCol"] is not None:
                parts.append(f"'strokeColValue':{self._js_val(r['strokeColValue'])}")
            if has["stroke"] is not None:
                parts.append(f"'strokeWValue':{self._js_val(r['strokeWValue'])}")
            if has["shape"] is not None:
                parts.append(f"'shapeValue':{self._js_val(r['shapeValue'])}")
            if has["opacity"] is not None:
                parts.append(f"'opacityValue':{self._js_val(r['opacityValue'])}")
            if all_pos:
                pos = ",".join(
                    f"'{name}':[{float(lx[i]):.5f},{float(ly[i]):.5f}]"
                    for name, (lx, ly) in all_pos.items()
                    if np.isfinite(lx[i]) and np.isfinite(ly[i])
                )
                parts.append("'pos':{" + pos + "}")
            if node_metrics:
                cent = ",".join(
                    f"'{name}':{float(v[i]):.5f}" for name, v in node_metrics.items()
                )
                parts.append("'cent':{" + cent + "}")
            if axis_vals:
                av = ",".join(
                    f"{_json.dumps(name)}:{self._js_val(vals[i])}"
                    for name, vals in axis_vals.items()
                )
                parts.append("'av':{" + av + "}")
            rows.append("{" + ",".join(parts) + "},")
        return "\n".join(rows)

    def _links_json(self, edgl: pd.DataFrame) -> str:
        post = {"eprob", "elo", "ehi"}.issubset(edgl.columns)
        rows = []
        for _, e in edgl.iterrows():
            row = (
                "{"
                f"'source':{self._js_val(e['from'])},"
                f"'target':{self._js_val(e['to'])},"
                f"'colorL':{self._js_val(e['colorL'])},"
                f"'lOpacity':{self._js_val(e['lOpacity'])},"
                f"'weigth':{self._js_val(e['weight'])},"
                f"'intralayer':{self._js_val(e['intralayer'])},"
                f"'interlayer':{self._js_val(e['interlayer'])}"
            )
            if post:
                row += (
                    f",'eprob':{self._js_val(e['eprob'])}"
                    f",'elo':{self._js_val(e['elo'])}"
                    f",'ehi':{self._js_val(e['ehi'])}"
                )
            rows.append(row + "},")
        return "\n".join(rows)

    def _assemble_html(
        self,
        d: pd.DataFrame,
        edgl: pd.DataFrame,
        ori,
        layout: str = "force",
        chord: dict | None = None,
        all_pos: dict | None = None,
        stats: dict | None = None,
        node_metrics: dict | None = None,
        legend: list | None = None,
        theme: str = "light",
        palette: str = "default",
        directed: bool = False,
        has_posterior: bool = False,
        canvas: bool = False,
        background: str = "grey",
        axis_cols: list | None = None,
        axis_vals: dict | None = None,
        axis_x: str = "(none)",
        axis_y: str = "(none)",
    ) -> str:
        import json as _json

        p1 = (self.assets_dir / "patron1.txt").read_text(encoding="utf-8")
        p2 = (self.assets_dir / "patron2.txt").read_text(encoding="utf-8")
        p2 = self._inject_layouts(
            p2, layout, computed=list((all_pos or {}).keys()), chord=chord
        )
        p1 = self._inject_features(p1)

        extra = {
            "stats": stats,
            "legend": legend or [],
            "theme": theme,
            "palette": palette,
            "directed": bool(directed),
            "hasPosterior": bool(has_posterior),
            "canvas": bool(canvas),
            "background": "#e6e6e8" if str(background).lower() == "grey" else background,
            "metricNames": list((node_metrics or {}).keys()),
            "axisCols": axis_cols or [],
            "axisX": axis_x,
            "axisY": axis_y,
        }
        if chord is not None:
            extra["chord"] = chord
        extra_js = "".join(
            f"json[{_json.dumps(k)}] = {_json.dumps(v)};\n" for k, v in extra.items()
        )
        return (
            p1
            + self._tooltip_js(ori)
            + p2
            + "\n           function getData() {\n   let json = { 'nodes':[\n"
            + self._nodes_json(d, ori, all_pos, node_metrics, axis_vals)
            + "\n],\n'links':[\n"
            + self._links_json(edgl)
            + "\n]}\n"
            + extra_js
            + "return json;\n}\n</script>\n"
            + "<script>\n" + _FEATURES_JS + "\n</script>\n"
        )

    def _build_legend(self, d, ori, color_node, palette, has_post) -> list:
        """One entry per *encoding channel* actually in use. ``title`` is the
        channel ("Size", "Colour", "Border width", ...); ``col`` is the source
        ``df`` column driving it (rendered as a sub-label, e.g. "Size /
        strength")."""
        ln, sn, cn, scn, stn, shn, on = ori
        ent: list = []

        # node fill colour
        if cn is not None and "colorValue" in d.columns:
            if getattr(self, "_color_is_categorical", False):
                seen = {}
                for v, hx in zip(d["colorValue"], d["color"]):
                    seen.setdefault(str(v), hx)
                ent.append({"kind": "swatches", "title": "Colour", "col": str(cn),
                            "items": [{"label": k, "color": v} for k, v in seen.items()]})
            else:
                ent.append({"kind": "gradient", "title": "Colour", "col": str(cn),
                            "stops": list(color_node),
                            "lo": round(float(d["colorValue"].min()), 3),
                            "hi": round(float(d["colorValue"].max()), 3)})

        # node shape
        if shn is not None and "shapeValue" in d.columns:
            seen = {}
            for v, code in zip(d["shapeValue"], d["shape"]):
                seen.setdefault(str(v), int(code))
            ent.append({"kind": "shapes", "title": "Shape", "col": str(shn),
                        "items": [{"label": k, "shape": v} for k, v in seen.items()]})

        # node size
        if sn is not None and "sizeValue" in d.columns:
            ent.append({"kind": "size", "title": "Size", "col": str(sn),
                        "lo": round(float(d["sizeValue"].min()), 3),
                        "hi": round(float(d["sizeValue"].max()), 3)})

        # node border colour
        if scn is not None and "strokeColValue" in d.columns:
            sv = d["strokeColValue"]
            if pd.api.types.is_numeric_dtype(sv):
                arr = sv.to_numpy(dtype=float)
                fin = np.flatnonzero(np.isfinite(arr))
                if fin.size:
                    # rank finite rows only: np.argsort sends NaN to the end, so
                    # order[-1] would otherwise index a NaN row whose strokeCol
                    # is NaN, collapsing the swatch while lo/hi show the real range
                    order = fin[np.argsort(arr[fin])]
                    stops = [str(d["strokeCol"].iloc[int(order[0])]),
                             str(d["strokeCol"].iloc[int(order[-1])])]
                    ent.append({"kind": "gradient", "title": "Border colour", "col": str(scn),
                                "stops": stops,
                                "lo": round(float(np.nanmin(arr)), 3),
                                "hi": round(float(np.nanmax(arr)), 3)})
            else:
                seen = {}
                for v, hx in zip(sv, d["strokeCol"]):
                    seen.setdefault(str(v), hx)
                ent.append({"kind": "swatches", "title": "Border colour", "col": str(scn),
                            "items": [{"label": k, "color": v} for k, v in seen.items()]})

        # node border width
        if stn is not None and "strokeWValue" in d.columns:
            wv = d["strokeWValue"]
            ent.append({"kind": "size", "title": "Border width", "col": str(stn),
                        "lo": round(float(wv.min()), 3),
                        "hi": round(float(wv.max()), 3)})

        if has_post:
            ent.append({"kind": "note", "title": "Link opacity", "text": "P(weight > 0)"})
        return ent

    def _inject_features(self, p1: str) -> str:
        p1 = p1.replace("</head>", f"<style>\n{_FEATURES_CSS}\n</style>\n</head>", 1)
        # Arrow size / Min edge weight belong in the native "Links" section:
        # splice them in right after the Curves control (before the spacer li).
        p1 = p1.replace(
            '        <li style="height: 1%;"></li>',
            _FEATURES_LINKS_HTML + '        <li style="height: 1%;"></li>',
            1,
        )
        p1 = p1.replace("    </nav>", _FEATURES_HTML + "\n    </nav>", 1)
        return p1

    _LIVE_LAYOUTS = ("force", "circle", "linear", "multilayer")

    @staticmethod
    def _lay_label(name: str) -> str:
        return "MDS" if name == "mds" else name.capitalize()

    def _inject_layouts(self, p2: str, layout: str, computed: list[str], chord: dict | None) -> str:
        """Extend the front-end: put every precomputed layout (and Chord, when a
        payload is present) in the dropdown; switch between them client-side."""
        want = ["Force", "Circle", "Linear", "Multilayer"]
        want += [self._lay_label(n) for n in computed if n not in self._LIVE_LAYOUTS]
        if chord is not None:
            want.append("Chord")
        initial = (
            layout.capitalize() if layout in self._LIVE_LAYOUTS
            else "Chord" if layout == "chord"
            else self._lay_label(layout)
        )
        if initial not in want:
            initial = "Force"

        p2 = p2.replace("var Layout = 'Force'", f"var Layout = '{initial}'", 1)
        p2 = p2.replace(
            'layouts = ["Force", "Circle", "Linear", "Multilayer"]',
            "layouts = " + repr(want).replace("'", '"'),
            1,
        )

        # pin nodes to whichever precomputed layout is selected + arc link paths
        pin_js = """
    {
      // map pos[layout] in [0,1] onto a centred square so aspect is preserved, scaled by Spread slider (dist)
      var _pk = Layout.toLowerCase(),
          _navW = 250, _pad = 40,
          _spread = (typeof dist !== 'undefined' && dist > 0) ? (dist / 500) : 1,
          _baseS = Math.min(width - _navW, height) - 2 * _pad,
          _S = _baseS * _spread,
          _cx = _navW + (width - _navW) / 2,
          _cy = height / 2,
          _ox = _cx - _S / 2,
          _oy = _cy - _S / 2;
      if (!window.ScatterMode) graph.nodes.forEach(function(d) {
        var p = d.pos && d.pos[_pk];
        if (p) { d.x = _ox + p[0] * _S; d.y = _oy + p[1] * _S; d.fx = d.x; d.fy = d.y; }
      });
    }
    if (Layout === 'Chord') { drawChordOnce(); }
    else { d3.select('#chordG').style('display', 'none');
           link.style('display', null); node.style('display', null); texts.style('display', null); }
"""
        p2 = p2.replace("    if(Layout == 'Circle'){", pin_js + "\n    if(Layout == 'Circle'){", 1)

        arc_link = """      // Curve (semicircular arcs under the Arc layout)
      link.attr('d', function(d) {
        if (Layout === 'Arc') {
          var r = Math.abs(d.target.x - d.source.x) / 2,
              y = Math.max(d.source.y, d.target.y),
              sweep = d.source.x < d.target.x ? 1 : 0;
          return 'M' + d.source.x + ',' + y + ' A' + r + ',' + r + ' 0 0,' + sweep + ' ' + d.target.x + ',' + y;
        }
        var dx = d.target.x - d.source.x,
            dy = d.target.y - d.source.y,
            dr = Math.sqrt(dx * dx + dy * dy)
        return 'M' + d.source.x + ',' + d.source.y + 'A' + dr + ',' + dr + ' 0 0,1 ' + d.target.x + ',' + d.target.y;
      });"""
        old_link = """      // Curve
      link.attr('d', function(d) {
        var dx = d.target.x - d.source.x,
            dy = d.target.y - d.source.y,
            dr = Math.sqrt(dx * dx + dy * dy)
        return 'M' + d.source.x + ',' + d.source.y + 'A' + dr + ',' + dr + ' 0 0,1 ' + d.target.x + ',' + d.target.y;
      });"""
        p2 = p2.replace(old_link, arc_link, 1)

        # show the initial layout as the selected <option>
        p2 = p2.replace(
            '.attr("value", function (d) { return d; }) // corresponding value returned by the button',
            '.attr("value", function (d) { return d; });\n'
            '      d3.select("#dataviz_builtWithD3 select").property("value", Layout);',
            1,
        )

        # dropdown: clear pins when returning to a live layout; accept the extra
        old_handler = """          if(selectedLayout == 'Circle'){Layout = 'Circle';simulation.alpha(0.5).restart();}
          if(selectedLayout == 'Linear'){Layout = 'Linear';simulation.alpha(0.5).restart();}
          if(selectedLayout == 'Force'){Layout = 'Force2';simulation.alpha(0.5).restart();}
          if(selectedLayout == 'Multilayer'){Layout = 'Multilayer';simulation.alpha(0.5).restart();}"""
        new_handler = """          window.ScatterMode = false;
          (function(){ var _ax=document.getElementById('axisX'), _ay=document.getElementById('axisY'),
              _cap=document.getElementById('axisCaption');
            if(_ax) _ax.value='(none)'; if(_ay) _ay.value='(none)'; if(_cap) _cap.style.display='none'; })();
          if(["Circle","Linear","Multilayer","Force"].indexOf(selectedLayout) !== -1){
            graph.nodes.forEach(function(d){ d.fx = null; d.fy = null; });
          }
          if(selectedLayout == 'Circle'){Layout = 'Circle';simulation.alpha(0.5).restart();}
          if(selectedLayout == 'Linear'){Layout = 'Linear';simulation.alpha(0.5).restart();}
          if(selectedLayout == 'Force'){Layout = 'Force2';simulation.alpha(0.5).restart();}
          if(selectedLayout == 'Multilayer'){Layout = 'Multilayer';simulation.alpha(0.5).restart();}
          if(["Circle","Linear","Multilayer","Force"].indexOf(selectedLayout) === -1){
            Layout = selectedLayout; simulation.alpha(0.5).restart();
          }"""
        p2 = p2.replace(old_handler, new_handler, 1)

        # chord helper (no-op unless a chord payload was emitted)
        chord_fn = """
    function drawChordOnce() {
      link.style('display','none'); node.style('display','none'); texts.style('display','none');
      if (window._chordDrawn || !graph.chord || typeof d3.chord !== 'function') return;
      window._chordDrawn = true;
      var cd = graph.chord, W = Math.min(width, height), R = W/2 - 70;
      var g = d3.select('svg').append('g').attr('id','chordG')
        .attr('transform','translate(' + (width/2) + ',' + (height/2) + ')');
      var chords = d3.chord().padAngle(0.045).sortSubgroups(d3.descending)(cd.matrix);
      var arcGen = d3.arc().innerRadius(R).outerRadius(R + 16);
      var ribGen = d3.ribbon().radius(R);
      var col = function(i){ return cd.colors[i % cd.colors.length]; };
      g.append('g').selectAll('path').data(chords.groups).enter().append('path')
        .attr('d', arcGen).style('fill', function(d){ return col(d.index); }).style('stroke','#333');
      g.append('g').style('opacity',0.72).selectAll('path').data(chords).enter().append('path')
        .attr('d', ribGen).style('fill', function(d){ return col(d.source.index); }).style('stroke','#333');
      g.append('g').selectAll('text').data(chords.groups).enter().append('text')
        .each(function(d){ d.a = (d.startAngle + d.endAngle) / 2; })
        .attr('dy','0.35em')
        .attr('transform', function(d){
          return 'rotate(' + (d.a * 180 / Math.PI - 90) + ') translate(' + (R + 22) + ')'
            + (d.a > Math.PI ? ' rotate(180)' : ''); })
        .attr('text-anchor', function(d){ return d.a > Math.PI ? 'end' : null; })
        .text(function(d){ return cd.labels[d.index]; })
        .style('font','12px sans-serif').style('fill','#111');
    }
"""
        p2 = p2.replace("var Layout = '", chord_fn + "\nvar Layout = '", 1)
        return p2
