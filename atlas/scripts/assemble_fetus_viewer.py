# -*- coding: utf-8 -*-
import io, json, base64, argparse

_cli = argparse.ArgumentParser(description=__doc__)
_cli.add_argument("--three-js", required=True, help="Path to three.min.js to inline")
_cli.add_argument("--geometry", required=True, help="Path to the combined fetus_atlas_geometry.bin")
_cli.add_argument("--manifest", required=True, help="Path to the part->group manifest JSON")
_cli.add_argument("--out", required=True, help="Output viewer HTML path")
_args = _cli.parse_args()

three_js = io.open(_args.three_js, encoding='utf-8').read()
geo_b64 = base64.b64encode(open(_args.geometry, 'rb').read()).decode('ascii')
manifest = json.load(open(_args.manifest))

# per-organ system grouping + curated colour + default opacity/visibility.
# "shell" organs (torso/head/limbs) default to a translucent skin so the
# internal anatomy reads immediately; everything else is opaque by default.
ORGAN_STYLE = {
    'torso':       {'color': 0xe8c9a0, 'opacity': 0.12, 'system': 'body shell'},
    'head':        {'color': 0xe8c9a0, 'opacity': 0.12, 'system': 'body shell'},
    'arm_left':    {'color': 0xdbb98a, 'opacity': 0.18, 'system': 'body shell'},
    'arm_right':   {'color': 0xdbb98a, 'opacity': 0.18, 'system': 'body shell'},
    'leg_left':    {'color': 0xdbb98a, 'opacity': 0.18, 'system': 'body shell'},
    'leg_right':   {'color': 0xdbb98a, 'opacity': 0.18, 'system': 'body shell'},
    'heart':       {'color': 0xc0392b, 'opacity': 1.0, 'system': 'cardiovascular'},
    'aorta':       {'color': 0xd64545, 'opacity': 1.0, 'system': 'cardiovascular'},
    'svc':         {'color': 0x3f6fae, 'opacity': 1.0, 'system': 'cardiovascular'},
    'portal':      {'color': 0x7a52a3, 'opacity': 1.0, 'system': 'cardiovascular'},
    'uv':          {'color': 0xd9534f, 'opacity': 1.0, 'system': 'umbilical'},
    'cord':        {'color': 0xcbb89a, 'opacity': 1.0, 'system': 'umbilical'},
    'placenta':    {'color': 0x8b2f2f, 'opacity': 1.0, 'system': 'umbilical'},
    'liver':       {'color': 0x8b4a3c, 'opacity': 1.0, 'system': 'abdominal'},
    'kidney':      {'color': 0x7a4b6e, 'opacity': 1.0, 'system': 'abdominal'},
    'bladder':     {'color': 0xd4b83f, 'opacity': 1.0, 'system': 'abdominal'},
    'lung':        {'color': 0xd98a9c, 'opacity': 1.0, 'system': 'thoracic'},
    'sex_organ_m': {'color': 0xb98a6e, 'opacity': 1.0, 'system': 'abdominal'},
    'womb':        {'color': 0xc77a95, 'opacity': 0.35, 'system': 'maternal'},
    'brain_left':  {'color': 0xd88fb0, 'opacity': 1.0, 'system': 'neuro'},
    'brain_right': {'color': 0xd88fb0, 'opacity': 1.0, 'system': 'neuro'},
    'cerebellum':  {'color': 0xc77a9e, 'opacity': 1.0, 'system': 'neuro'},
    'eye':         {'color': 0xeef2f5, 'opacity': 1.0, 'system': 'neuro'},
}

groups = sorted(ORGAN_STYLE.keys())
tri_by_group = {}
for m in manifest:
    tri_by_group[m['group']] = tri_by_group.get(m['group'], 0) + m['n_triangles']

# legend checkboxes, grouped by system for a scannable HUD
systems_order = ['cardiovascular', 'umbilical', 'abdominal', 'thoracic', 'neuro', 'maternal', 'body shell']
by_system = {}
for g in groups:
    by_system.setdefault(ORGAN_STYLE[g]['system'], []).append(g)

legend_html = ''
for sysname in systems_order:
    if sysname not in by_system: continue
    legend_html += '<div class="sysgroup"><div class="sysname">%s</div>' % sysname
    for g in sorted(by_system[sysname]):
        style = ORGAN_STYLE[g]
        hexcol = '#%06x' % style['color']
        n_tris = tri_by_group.get(g, 0)
        legend_html += (
            '<div class="organ-row">'
            '<label><input type="checkbox" class="organ-toggle" data-organ="%s" checked>'
            '<span class="sw" style="background:%s"></span>%s '
            '<span class="n">%s tri</span></label>'
            '<input type="range" class="organ-opacity" data-organ="%s" min="0" max="100" value="%d" title="opacity">'
            '</div>'
        ) % (g, hexcol, g, format(n_tris, ','), g, round(style['opacity'] * 100))
    legend_html += '</div>'

organ_style_json = json.dumps({g: {'color': v['color'], 'opacity': v['opacity']} for g, v in ORGAN_STYLE.items()})
total_tris = sum(tri_by_group.values())

html = r'''<title>Fetal Anatomy Atlas</title>
<style>
:root {
  --bg: #10151b; --panel: #171e26; --border: #29323d;
  --text: #e7edf2; --text-muted: #8b98a6; --accent: #e0a84f;
  --mono: ui-monospace, "SF Mono", "Cascadia Mono", Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
}
:root[data-theme="light"] { --bg: #f2f5f7; --panel: #ffffff; --border: #dbe2e8; --text: #131a20; --text-muted: #5b6a76; --accent: #a3721f; }
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) { --bg: #f2f5f7; --panel: #ffffff; --border: #dbe2e8; --text: #131a20; --text-muted: #5b6a76; --accent: #a3721f; }
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; height: 100%; background: var(--bg); color: var(--text); font-family: var(--sans); overflow: hidden; }
#viewport { position: fixed; inset: 0; }
#viewport canvas { display: block; width: 100%; height: 100%; }
header {
  position: fixed; top: 0; left: 0; right: 0; z-index: 10;
  display: flex; align-items: baseline; gap: 0.75rem; padding: 0.85rem 1.1rem;
  background: linear-gradient(to bottom, color-mix(in srgb, var(--bg) 88%, transparent), transparent);
  pointer-events: none;
}
header h1 { font-family: var(--mono); font-size: 0.8rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--text); margin: 0; }
header .src { font-family: var(--mono); font-size: 0.72rem; color: var(--text-muted); }
#hud {
  position: fixed; top: 3.2rem; left: 1rem; bottom: 1rem; z-index: 10;
  background: color-mix(in srgb, var(--panel) 90%, transparent);
  border: 1px solid var(--border); border-radius: 6px; padding: 0.75rem 0.85rem;
  font-family: var(--mono); font-size: 0.7rem; line-height: 1.5; color: var(--text-muted);
  width: 250px; backdrop-filter: blur(6px); overflow-y: auto;
}
#hud .toolbar { display: flex; gap: 0.4rem; margin-bottom: 0.6rem; }
#hud .toolbar button {
  flex: 1; background: transparent; border: 1px solid var(--border); color: var(--text-muted);
  font-family: var(--mono); font-size: 0.66rem; padding: 0.3rem 0.4rem; border-radius: 4px; cursor: pointer;
}
#hud .toolbar button:hover { color: var(--text); border-color: var(--accent); }
.sysgroup { margin-bottom: 0.65rem; }
.sysname { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--accent); margin-bottom: 0.25rem; }
#hud label { display: flex; align-items: center; gap: 6px; cursor: pointer; padding: 0.08rem 0; }
#hud input { accent-color: var(--accent); cursor: pointer; }
#hud .sw { display: inline-block; width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
#hud .n { margin-left: auto; color: var(--text-muted); font-size: 0.62rem; }
.organ-row { margin-bottom: 0.2rem; }
.organ-opacity { width: 100%; height: 10px; margin: 0.02rem 0 0.2rem; }
#hud .divider { height: 1px; background: var(--border); margin: 0.55rem 0; }
#hud .note { color: var(--text-muted); font-size: 0.64rem; line-height: 1.55; }
#hint { position: fixed; bottom: 1rem; right: 1rem; z-index: 10; font-family: var(--mono); font-size: 0.68rem; color: var(--text-muted); text-align: right; line-height: 1.6; }
#hint kbd { font-family: var(--mono); color: var(--accent); }
#theme-toggle {
  position: fixed; top: 0.85rem; right: 1.1rem; z-index: 10; pointer-events: auto;
  background: transparent; border: 1px solid var(--border); color: var(--text-muted);
  font-family: var(--mono); font-size: 0.68rem; padding: 0.3rem 0.55rem; border-radius: 4px; cursor: pointer;
}
#theme-toggle:hover { color: var(--text); border-color: var(--accent); }
#loading { position: fixed; inset: 0; z-index: 20; display: flex; align-items: center; justify-content: center; background: var(--bg); font-family: var(--mono); font-size: 0.78rem; color: var(--text-muted); letter-spacing: 0.04em; }
#footer { position: fixed; left: 0; right: 0; bottom: 0.4rem; z-index: 10; text-align: center; font-family: var(--mono); font-size: 0.66rem; color: var(--text-muted); pointer-events: none; }
#footer a { color: var(--text-muted); pointer-events: auto; }
#footer a:hover { color: var(--accent); }
</style>

<div id="loading">rendering fetal atlas (''' + format(total_tris, ',') + r''' triangles, 23 structures)</div>
<div id="viewport"></div>
<header>
  <h1>Fetal anatomy atlas</h1>
  <span class="src">MR-digitized fetal anatomy, 23 structures &mdash; self-contained three.js viewer, geometry kept simulation-ready</span>
</header>
<button id="theme-toggle" type="button" aria-label="Toggle theme">&#9680; theme</button>
<div id="hud">
  <div class="toolbar">
    <button id="show-all">Show all</button>
    <button id="hide-all">Hide all</button>
  </div>
  ''' + legend_html + r'''
</div>
<div id="hint">drag &mdash; <kbd>rotate</kbd><br>scroll &mdash; <kbd>zoom</kbd></div>
<div id="footer">MIT License &mdash; questions to <a id="contact-email" href="#">Dr Harvey Ho</a></div>

<script>
window.GEO_B64 = "''' + geo_b64 + r'''";
window.ORGAN_STYLE = ''' + organ_style_json + r''';
</script>
<script>
''' + three_js + r'''
</script>
<script>
(function () {
  "use strict";
  // built at runtime, not present as a literal string in the page source;
  // link text stays "Dr Harvey Ho" -- the address itself is never displayed
  var eu = ["harvey", ".nz"].join(""), ed = ["gmail", ".com"].join("");
  var contactEl = document.getElementById("contact-email");
  contactEl.href = "mailto:" + eu + "@" + ed;

  var root = document.documentElement;
  document.getElementById("theme-toggle").addEventListener("click", function () {
    var cur = root.getAttribute("data-theme");
    var next = cur === "dark" ? "light" : cur === "light" ? null : "dark";
    if (next) root.setAttribute("data-theme", next); else root.removeAttribute("data-theme");
  });

  function base64ToArrayBuffer(b64) {
    var binary = atob(b64), len = binary.length, bytes = new Uint8Array(len);
    for (var i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
    return bytes.buffer;
  }

  // parse the combined geometry blob: uint32 numGroups, then per group:
  // uint32 nameLen, name bytes, uint32 triCount, triCount * (pos3+norm3)*3 verts
  function parseGeometryBlob(buffer) {
    var view = new DataView(buffer);
    var off = 0;
    var numGroups = view.getUint32(off, true); off += 4;
    var groups = {};
    var decoder = new TextDecoder("utf-8");
    for (var g = 0; g < numGroups; g++) {
      var nameLen = view.getUint32(off, true); off += 4;
      var nameBytes = new Uint8Array(buffer, off, nameLen); off += nameLen;
      var name = decoder.decode(nameBytes);
      var triCount = view.getUint32(off, true); off += 4;
      var positions = new Float32Array(triCount * 9);
      var normals = new Float32Array(triCount * 9);
      for (var t = 0; t < triCount; t++) {
        for (var v = 0; v < 3; v++) {
          var base = t*9 + v*3;
          positions[base] = view.getFloat32(off, true);
          positions[base+1] = view.getFloat32(off+4, true);
          positions[base+2] = view.getFloat32(off+8, true);
          off += 12;
          normals[base] = view.getFloat32(off, true);
          normals[base+1] = view.getFloat32(off+4, true);
          normals[base+2] = view.getFloat32(off+8, true);
          off += 12;
        }
      }
      groups[name] = { positions: positions, normals: normals, triCount: triCount };
    }
    return groups;
  }

  var rawGroups = parseGeometryBlob(base64ToArrayBuffer(window.GEO_B64));

  var viewportEl = document.getElementById("viewport");
  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(38, window.innerWidth/window.innerHeight, 0.1, 8000);

  var renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  viewportEl.appendChild(renderer.domElement);

  // compute a shared bounding box (from torso, the largest reference shell)
  // and centre every group on it so the whole atlas sits at the origin
  var refGeo = new THREE.BufferGeometry();
  refGeo.setAttribute("position", new THREE.BufferAttribute(rawGroups["torso"].positions, 3));
  refGeo.computeBoundingBox();
  var bbox = refGeo.boundingBox;
  var size = new THREE.Vector3(); bbox.getSize(size);
  var center = new THREE.Vector3(); bbox.getCenter(center);
  var maxDim = Math.max(size.x, size.y, size.z);
  var camDist = maxDim * 1.9;

  function bgColor() {
    var isLight = root.getAttribute("data-theme") === "light" ||
      (!root.getAttribute("data-theme") && window.matchMedia("(prefers-color-scheme: light)").matches);
    return isLight ? 0xf2f5f7 : 0x10151b;
  }
  scene.background = new THREE.Color(bgColor());
  scene.fog = new THREE.Fog(bgColor(), camDist * 1.5, camDist * 3.6);

  scene.add(new THREE.HemisphereLight(0xdfeaf0, 0x1a1410, 0.95));
  var key = new THREE.DirectionalLight(0xeaf2ff, 1.15);
  key.position.set(maxDim*1.3, maxDim*1.7, maxDim*1.1);
  scene.add(key);
  var rim = new THREE.DirectionalLight(0xffd8b0, 0.35);
  rim.position.set(-maxDim, -maxDim*0.5, -maxDim*0.9);
  scene.add(rim);

  var meshesByOrgan = {};
  Object.keys(rawGroups).forEach(function (name) {
    var raw = rawGroups[name];
    var geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(raw.positions, 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(raw.normals, 3));
    geo.translate(-center.x, -center.y, -center.z);
    var style = window.ORGAN_STYLE[name] || { color: 0x999999, opacity: 1.0 };
    var transparent = style.opacity < 1.0;
    var mat = new THREE.MeshStandardMaterial({
      color: style.color, roughness: 0.5, metalness: 0.05, side: THREE.DoubleSide,
      transparent: transparent, opacity: style.opacity, depthWrite: !transparent
    });
    var mesh = new THREE.Mesh(geo, mat);
    mesh.renderOrder = transparent ? 1 : 0;
    scene.add(mesh);
    meshesByOrgan[name] = mesh;
  });

  document.querySelectorAll(".organ-toggle").forEach(function (cb) {
    cb.addEventListener("change", function () {
      var m = meshesByOrgan[cb.dataset.organ];
      if (m) m.visible = cb.checked;
    });
  });
  document.querySelectorAll(".organ-opacity").forEach(function (sl) {
    sl.addEventListener("input", function () {
      var m = meshesByOrgan[sl.dataset.organ];
      if (!m) return;
      var op = sl.value / 100;
      m.material.opacity = op;
      var isTransparent = op < 1.0;
      if (m.material.transparent !== isTransparent) {
        m.material.transparent = isTransparent;
        m.material.depthWrite = !isTransparent;
        m.material.needsUpdate = true;
      }
      m.renderOrder = isTransparent ? 1 : 0;
    });
  });
  document.getElementById("show-all").addEventListener("click", function () {
    document.querySelectorAll(".organ-toggle").forEach(function (cb) {
      cb.checked = true;
      var m = meshesByOrgan[cb.dataset.organ];
      if (m) m.visible = true;
    });
  });
  document.getElementById("hide-all").addEventListener("click", function () {
    document.querySelectorAll(".organ-toggle").forEach(function (cb) {
      cb.checked = false;
      var m = meshesByOrgan[cb.dataset.organ];
      if (m) m.visible = false;
    });
  });

  var rotX = -0.15, rotY = 0.5, radius = camDist;
  var dragging = false, lastX = 0, lastY = 0, autoRotate = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  function updateCamera() {
    var x = radius * Math.cos(rotX) * Math.sin(rotY);
    var y = radius * Math.sin(rotX);
    var z = radius * Math.cos(rotX) * Math.cos(rotY);
    camera.position.set(x, y, z);
    camera.lookAt(0, 0, 0);
  }
  updateCamera();
  renderer.domElement.style.cursor = "grab";
  renderer.domElement.addEventListener("pointerdown", function (e) {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    renderer.domElement.style.cursor = "grabbing";
    renderer.domElement.setPointerCapture(e.pointerId);
    autoRotate = false;
  });
  renderer.domElement.addEventListener("pointerup", function () { dragging = false; renderer.domElement.style.cursor = "grab"; });
  renderer.domElement.addEventListener("pointermove", function (e) {
    if (!dragging) return;
    var dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    rotY -= dx * 0.0065; rotX += dy * 0.0065;
    rotX = Math.max(-1.45, Math.min(1.45, rotX));
    updateCamera();
  });
  renderer.domElement.addEventListener("wheel", function (e) {
    e.preventDefault();
    radius *= (1 + Math.sign(e.deltaY) * 0.08);
    radius = Math.max(maxDim*0.4, Math.min(maxDim*6, radius));
    updateCamera();
  }, { passive: false });

  window.addEventListener("resize", function () {
    camera.aspect = window.innerWidth/window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  var mq = window.matchMedia("(prefers-color-scheme: light)");
  function refreshBg() {
    var c = bgColor();
    scene.background = new THREE.Color(c);
    scene.fog.color = new THREE.Color(c);
  }
  mq.addEventListener("change", refreshBg);
  document.getElementById("theme-toggle").addEventListener("click", refreshBg);

  document.getElementById("loading").style.display = "none";

  function animate() {
    requestAnimationFrame(animate);
    if (autoRotate) { rotY += 0.0018; updateCamera(); }
    renderer.render(scene, camera);
  }
  animate();
})();
</script>
'''

with open(_args.out, 'w', encoding='utf-8') as f:
    f.write(html)
print('wrote', len(html), 'bytes')
