(function () {
  "use strict";

  var root = document.documentElement;
  document.getElementById("theme-toggle").addEventListener("click", function () {
    var cur = root.getAttribute("data-theme");
    var next = cur === "dark" ? "light" : cur === "light" ? null : "dark";
    if (next) root.setAttribute("data-theme", next); else root.removeAttribute("data-theme");
    scene.background = new THREE.Color(bgColor());
  });

  function b64ToFloat32(b64) {
    var bin = atob(b64), len = bin.length, bytes = new Uint8Array(len);
    for (var i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }
  function b64ToUint32(b64) {
    var bin = atob(b64), len = bin.length, bytes = new Uint8Array(len);
    for (var i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
    return new Uint32Array(bytes.buffer);
  }
  var D = window.SCENE_DATA;
  var cordVerts = b64ToFloat32(D.cord_verts_b64), cordRadial = b64ToFloat32(D.cord_radial_b64);
  var cordCentre = b64ToFloat32(D.cord_centre_b64), cordFaces = b64ToUint32(D.cord_faces_b64);
  var CORD_NV = D.cord_n_verts, CORD_NF = D.cord_n_faces;
  var placentaPos = b64ToFloat32(D.placenta_pos_b64), placentaNorm = b64ToFloat32(D.placenta_norm_b64);
  var uvPos = b64ToFloat32(D.uv_pos_b64), UV_N = D.uv_n;
  var localCordDist = b64ToFloat32(D.local_cord_dist_b64);
  var taper = b64ToFloat32(D.taper_b64);
  var GC = D.growth_curves;
  var META = D.meta;

  // ---------------------------------------------------------------------
  // PCHIP (Fritsch-Carlson monotone cubic Hermite), general N points
  // ---------------------------------------------------------------------
  function pchipTangents(x, y) {
    var n = x.length, h = [], delta = [], m = new Array(n);
    for (var i = 0; i < n - 1; i++) { h.push(x[i + 1] - x[i]); delta.push((y[i + 1] - y[i]) / h[i]); }
    for (var i = 1; i < n - 1; i++) {
      if (delta[i - 1] * delta[i] <= 0) { m[i] = 0; }
      else {
        var w1 = 2 * h[i] + h[i - 1], w2 = h[i] + 2 * h[i - 1];
        m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
      }
    }
    function edge(h0, h1, d0, d1) {
      var m0 = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
      if (m0 * d0 <= 0) return 0;
      if (d0 * d1 <= 0 && Math.abs(m0) > 3 * Math.abs(d0)) return 3 * d0;
      return m0;
    }
    m[0] = edge(h[0], h[1], delta[0], delta[1]);
    m[n - 1] = edge(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
    return m;
  }
  function pchipEval(x, y, m, xq) {
    var n = x.length;
    // linear extrapolation beyond the anchored range, using the boundary
    // tangent (matches scipy's PchipInterpolator(extrapolate=True))
    if (xq < x[0]) return y[0] + m[0] * (xq - x[0]);
    if (xq > x[n - 1]) return y[n - 1] + m[n - 1] * (xq - x[n - 1]);
    var i = 0;
    while (i < n - 2 && xq > x[i + 1]) i++;
    var h = x[i + 1] - x[i], t = (xq - x[i]) / h;
    var h00 = 2 * t ** 3 - 3 * t ** 2 + 1, h10 = t ** 3 - 2 * t ** 2 + t;
    var h01 = -2 * t ** 3 + 3 * t ** 2, h11 = t ** 3 - t ** 2;
    return h00 * y[i] + h10 * h * m[i] + h01 * y[i + 1] + h11 * h * m[i + 1];
  }
  var uvM = pchipTangents(GC.uv_ga, GC.uv_diam_mm);
  var uaM = pchipTangents(GC.ua_ga, GC.ua_diam_mm);
  function uvDiameterMm(ga) { return pchipEval(GC.uv_ga, GC.uv_diam_mm, uvM, ga); }
  function uaDiameterMm(ga) { return pchipEval(GC.ua_ga, GC.ua_diam_mm, uaM, ga); }

  // ---------------------------------------------------------------------
  // Bishop (rotation-minimizing) frame along the fixed UV centreline
  // ---------------------------------------------------------------------
  function bishopFrame(pos, n) {
    var T = new Float32Array(n * 3), N = new Float32Array(n * 3), B = new Float32Array(n * 3);
    for (var i = 0; i < n - 1; i++) {
      var ax = pos[(i + 1) * 3] - pos[i * 3], ay = pos[(i + 1) * 3 + 1] - pos[i * 3 + 1], az = pos[(i + 1) * 3 + 2] - pos[i * 3 + 2];
      var len = Math.hypot(ax, ay, az) || 1e-9;
      T[i * 3] = ax / len; T[i * 3 + 1] = ay / len; T[i * 3 + 2] = az / len;
    }
    T[(n - 1) * 3] = T[(n - 2) * 3]; T[(n - 1) * 3 + 1] = T[(n - 2) * 3 + 1]; T[(n - 1) * 3 + 2] = T[(n - 2) * 3 + 2];
    var ref = Math.abs(T[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
    var t0 = [T[0], T[1], T[2]];
    var dot0 = ref[0]*t0[0]+ref[1]*t0[1]+ref[2]*t0[2];
    var n0 = [ref[0]-dot0*t0[0], ref[1]-dot0*t0[1], ref[2]-dot0*t0[2]];
    var nlen = Math.hypot(n0[0],n0[1],n0[2]) || 1e-9;
    N[0]=n0[0]/nlen; N[1]=n0[1]/nlen; N[2]=n0[2]/nlen;
    function cross(a,b){ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
    var b0 = cross(t0, [N[0],N[1],N[2]]); B[0]=b0[0]; B[1]=b0[1]; B[2]=b0[2];
    for (var i2 = 0; i2 < n - 1; i2++) {
      var ta=[T[i2*3],T[i2*3+1],T[i2*3+2]], tb=[T[(i2+1)*3],T[(i2+1)*3+1],T[(i2+1)*3+2]];
      var v = cross(ta, tb); var s = Math.hypot(v[0],v[1],v[2]); var c = ta[0]*tb[0]+ta[1]*tb[1]+ta[2]*tb[2];
      var Ncur=[N[i2*3],N[i2*3+1],N[i2*3+2]], Nnext;
      if (s < 1e-9) { Nnext = Ncur; }
      else {
        var axis=[v[0]/s,v[1]/s,v[2]/s]; var angle = Math.atan2(s, c); var cosA=Math.cos(angle), sinA=Math.sin(angle);
        var crossAN = cross(axis, Ncur); var dotAN = axis[0]*Ncur[0]+axis[1]*Ncur[1]+axis[2]*Ncur[2];
        Nnext = [Ncur[0]*cosA + crossAN[0]*sinA + axis[0]*dotAN*(1-cosA),
                 Ncur[1]*cosA + crossAN[1]*sinA + axis[1]*dotAN*(1-cosA),
                 Ncur[2]*cosA + crossAN[2]*sinA + axis[2]*dotAN*(1-cosA)];
        var dTn = Nnext[0]*tb[0]+Nnext[1]*tb[1]+Nnext[2]*tb[2];
        Nnext = [Nnext[0]-dTn*tb[0], Nnext[1]-dTn*tb[1], Nnext[2]-dTn*tb[2]];
        var nl = Math.hypot(Nnext[0],Nnext[1],Nnext[2]) || 1e-9;
        Nnext = [Nnext[0]/nl, Nnext[1]/nl, Nnext[2]/nl];
      }
      N[(i2+1)*3]=Nnext[0]; N[(i2+1)*3+1]=Nnext[1]; N[(i2+1)*3+2]=Nnext[2];
      var bnext = cross(tb, Nnext); B[(i2+1)*3]=bnext[0]; B[(i2+1)*3+1]=bnext[1]; B[(i2+1)*3+2]=bnext[2];
    }
    return { T: T, N: N, B: B };
  }
  var uvFrame = bishopFrame(uvPos, UV_N);
  var uvArcLen = (function () {
    var arc = new Float32Array(UV_N); arc[0] = 0;
    for (var i = 1; i < UV_N; i++) {
      var dx = uvPos[i*3]-uvPos[(i-1)*3], dy = uvPos[i*3+1]-uvPos[(i-1)*3+1], dz = uvPos[i*3+2]-uvPos[(i-1)*3+2];
      arc[i] = arc[i-1] + Math.hypot(dx,dy,dz);
    }
    return arc;
  })();

  // ---------------------------------------------------------------------
  // live UA-pair path generation (helix around the fixed UV centreline)
  // ---------------------------------------------------------------------
  function computeUAPaths(ga, pitchMm) {
    var uvR = uvDiameterMm(ga) / 2, uaR = uaDiameterMm(ga) / 2;
    var literatureOffset = uvR + META.gap_mm + uaR;
    var omega = 2 * Math.PI / pitchMm * -1.0;  // left-handed
    var deltaTheta = META.delta_theta_deg * Math.PI / 180;
    var baseAngle = -deltaTheta / 2;
    var a1 = new Float32Array(UV_N * 3), a2 = new Float32Array(UV_N * 3);
    for (var i = 0; i < UV_N; i++) {
      var localMax = Math.max(localCordDist[i] - 0.3 - uaR, uaR * 1.2);
      var offset = Math.min(literatureOffset, localMax) * taper[i];
      var ang = baseAngle + omega * uvArcLen[i];
      var nx = uvFrame.N[i*3], ny = uvFrame.N[i*3+1], nz = uvFrame.N[i*3+2];
      var bx = uvFrame.B[i*3], by = uvFrame.B[i*3+1], bz = uvFrame.B[i*3+2];
      var c1 = Math.cos(ang), s1 = Math.sin(ang);
      a1[i*3] = uvPos[i*3] + offset*(c1*nx + s1*bx);
      a1[i*3+1] = uvPos[i*3+1] + offset*(c1*ny + s1*by);
      a1[i*3+2] = uvPos[i*3+2] + offset*(c1*nz + s1*bz);
      var ang2 = ang + deltaTheta, c2 = Math.cos(ang2), s2 = Math.sin(ang2);
      a2[i*3] = uvPos[i*3] + offset*(c2*nx + s2*bx);
      a2[i*3+1] = uvPos[i*3+1] + offset*(c2*ny + s2*by);
      a2[i*3+2] = uvPos[i*3+2] + offset*(c2*nz + s2*bz);
    }
    return { a1: a1, a2: a2, uvR: uvR, uaR: uaR };
  }

  // ---------------------------------------------------------------------
  // tube mesh builder (UV, UA1, UA2)
  // ---------------------------------------------------------------------
  var RADIAL_SEG = 10;
  function buildTube(pos, n, frame, radius, colorHex) {
    var positions = new Float32Array(n * RADIAL_SEG * 3);
    var normals = new Float32Array(n * RADIAL_SEG * 3);
    for (var i = 0; i < n; i++) {
      var nx = frame.N[i*3], ny = frame.N[i*3+1], nz = frame.N[i*3+2];
      var bx = frame.B[i*3], by = frame.B[i*3+1], bz = frame.B[i*3+2];
      for (var j = 0; j < RADIAL_SEG; j++) {
        var ang = (2 * Math.PI * j) / RADIAL_SEG, ca = Math.cos(ang), sa = Math.sin(ang);
        var ox = ca * nx + sa * bx, oy = ca * ny + sa * by, oz = ca * nz + sa * bz;
        var idx = (i * RADIAL_SEG + j) * 3;
        positions[idx] = pos[i*3] + radius * ox; positions[idx+1] = pos[i*3+1] + radius * oy; positions[idx+2] = pos[i*3+2] + radius * oz;
        normals[idx] = ox; normals[idx+1] = oy; normals[idx+2] = oz;
      }
    }
    var indices = [];
    for (var i2 = 0; i2 < n - 1; i2++) {
      for (var j2 = 0; j2 < RADIAL_SEG; j2++) {
        var a = i2 * RADIAL_SEG + j2, b = i2 * RADIAL_SEG + (j2+1) % RADIAL_SEG;
        var c = (i2+1) * RADIAL_SEG + j2, d = (i2+1) * RADIAL_SEG + (j2+1) % RADIAL_SEG;
        indices.push(a, c, b, b, c, d);
      }
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    geo.setIndex(indices);
    var mat = new THREE.MeshStandardMaterial({ color: colorHex, roughness: 0.4, metalness: 0.05, side: THREE.DoubleSide });
    return new THREE.Mesh(geo, mat);
  }

  // ---------------------------------------------------------------------
  // scene setup
  // ---------------------------------------------------------------------
  var viewportEl = document.getElementById("viewport");
  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.1, 20000);
  var renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  viewportEl.appendChild(renderer.domElement);

  function bgColor() {
    var isLight = root.getAttribute("data-theme") === "light" ||
      (!root.getAttribute("data-theme") && window.matchMedia("(prefers-color-scheme: light)").matches);
    return isLight ? 0xeef2f4 : 0x0c1119;
  }
  scene.background = new THREE.Color(bgColor());
  scene.add(new THREE.HemisphereLight(0xdfeaf0, 0x151b23, 1.05));
  var key = new THREE.DirectionalLight(0xffffff, 1.2); key.position.set(300, 400, 300); scene.add(key);
  var rim = new THREE.DirectionalLight(0xbfe0ff, 0.4); rim.position.set(-300, -150, -250); scene.add(rim);

  var allPos = new Float32Array(cordVerts.length + placentaPos.length);
  allPos.set(cordVerts, 0); allPos.set(placentaPos, cordVerts.length);
  var bboxGeo = new THREE.BufferGeometry();
  bboxGeo.setAttribute("position", new THREE.BufferAttribute(allPos, 3));
  bboxGeo.computeBoundingBox();
  var bbox = bboxGeo.boundingBox;
  var size = new THREE.Vector3(); bbox.getSize(size);
  var center = new THREE.Vector3(); bbox.getCenter(center);
  var maxDim = Math.max(size.x, size.y, size.z);

  function recenter(arr, n) {
    var out = new Float32Array(arr.length);
    for (var i = 0; i < n; i++) { out[i*3]=arr[i*3]-center.x; out[i*3+1]=arr[i*3+1]-center.y; out[i*3+2]=arr[i*3+2]-center.z; }
    return out;
  }
  var cordVertsC = recenter(cordVerts, CORD_NV), cordCentreC = recenter(cordCentre, CORD_NV);
  var placentaPosC = recenter(placentaPos, placentaPos.length / 3);
  var uvPosC = recenter(uvPos, UV_N);

  // cord: indexed geometry, positions rebuilt each time GA changes
  var cordGeo = new THREE.BufferGeometry();
  var cordPositions = new THREE.BufferAttribute(new Float32Array(CORD_NV * 3), 3);
  cordGeo.setAttribute("position", cordPositions);
  cordGeo.setIndex(Array.from(cordFaces));
  var cordMat = new THREE.MeshStandardMaterial({ color: 0xc9a876, roughness: 0.55, metalness: 0.03, side: THREE.DoubleSide, transparent: true, opacity: 0.35, depthWrite: false });
  var cordMesh = new THREE.Mesh(cordGeo, cordMat);
  scene.add(cordMesh);

  function updateCordGeometry(scale) {
    var arr = cordPositions.array;
    for (var i = 0; i < CORD_NV; i++) {
      arr[i*3] = cordCentreC[i*3] + cordRadial[i*3] * scale;
      arr[i*3+1] = cordCentreC[i*3+1] + cordRadial[i*3+1] * scale;
      arr[i*3+2] = cordCentreC[i*3+2] + cordRadial[i*3+2] * scale;
    }
    cordPositions.needsUpdate = true;
    cordGeo.computeVertexNormals();
  }

  var placentaMat = new THREE.MeshStandardMaterial({ color: 0x8b2f2f, roughness: 0.55, metalness: 0.03, side: THREE.DoubleSide, transparent: true, opacity: 0.45, depthWrite: false });
  var placentaGeo = new THREE.BufferGeometry();
  placentaGeo.setAttribute("position", new THREE.BufferAttribute(placentaPosC, 3));
  placentaGeo.setAttribute("normal", new THREE.BufferAttribute(placentaNorm, 3));
  var placentaMesh = new THREE.Mesh(placentaGeo, placentaMat);
  scene.add(placentaMesh);

  var uvMesh = null, ua1Mesh = null, ua2Mesh = null;
  var uvRefDiam = uvDiameterMm(META.reference_ga);
  var uaRefDiam = uaDiameterMm(META.reference_ga);

  function rebuildVessels(ga, pitchMm) {
    var ua = computeUAPaths(ga, pitchMm);
    var a1c = recenter(ua.a1, UV_N), a2c = recenter(ua.a2, UV_N);
    var ua1Frame = bishopFrame(a1c, UV_N), ua2Frame = bishopFrame(a2c, UV_N);

    if (uvMesh) { scene.remove(uvMesh); uvMesh.geometry.dispose(); }
    if (ua1Mesh) { scene.remove(ua1Mesh); ua1Mesh.geometry.dispose(); }
    if (ua2Mesh) { scene.remove(ua2Mesh); ua2Mesh.geometry.dispose(); }
    uvMesh = buildTube(uvPosC, UV_N, uvFrame, ua.uvR, 0x2dc7b4);
    ua1Mesh = buildTube(a1c, UV_N, ua1Frame, ua.uaR, 0xe8a13d);
    ua2Mesh = buildTube(a2c, UV_N, ua2Frame, ua.uaR, 0xe8a13d);
    uvMesh.visible = layerState.uv; ua1Mesh.visible = layerState.ua1; ua2Mesh.visible = layerState.ua2;
    scene.add(uvMesh); scene.add(ua1Mesh); scene.add(ua2Mesh);

    // cord girth scales with the UV growth curve, relative to this specimen's own GA
    var cordScale = uvDiameterMm(ga) / uvRefDiam;
    updateCordGeometry(cordScale);
    cordMesh.visible = layerState.cord;

    document.getElementById("uv-diam-val").textContent = (ua.uvR * 2).toFixed(2);
    document.getElementById("ua-diam-val").textContent = (ua.uaR * 2).toFixed(2);
    var spiralsPerCm = 10 / pitchMm;
    document.getElementById("coil-val").textContent = spiralsPerCm.toFixed(2);
  }

  var layerState = { cord: true, placenta: true, uv: true, ua1: true, ua2: true };
  document.querySelectorAll("[data-layer]").forEach(function (cb) {
    cb.addEventListener("change", function () {
      layerState[cb.dataset.layer] = cb.checked;
      var m = { cord: cordMesh, placenta: placentaMesh, uv: uvMesh, ua1: ua1Mesh, ua2: ua2Mesh }[cb.dataset.layer];
      if (m) m.visible = cb.checked;
    });
  });
  document.querySelectorAll("[data-opacity]").forEach(function (sl) {
    sl.addEventListener("input", function () {
      var m = { cord: cordMesh, placenta: placentaMesh }[sl.dataset.opacity];
      if (m) m.material.opacity = sl.value / 100;
    });
  });

  var gaSlider = document.getElementById("ga-slider");
  var pitchSlider = document.getElementById("pitch-slider");
  function onSlidersChange() {
    var ga = parseFloat(gaSlider.value), pitch = parseFloat(pitchSlider.value);
    document.getElementById("ga-val").textContent = ga.toFixed(0);
    document.getElementById("pitch-val").textContent = pitch.toFixed(0);
    rebuildVessels(ga, pitch);
  }
  gaSlider.addEventListener("input", onSlidersChange);
  pitchSlider.addEventListener("input", onSlidersChange);

  rebuildVessels(META.reference_ga, META.default_pitch_mm);

  // ---------------------------------------------------------------------
  // camera orbit + pan controls
  //   drag                  -> rotate
  //   right/middle drag, or shift+drag -> pan
  //   two-finger drag (touch) -> pan (+ pinch to zoom)
  //   scroll / pinch        -> zoom
  // ---------------------------------------------------------------------
  var camState = { az: 0.6, pol: 1.15, radius: maxDim * 1.7, target: new THREE.Vector3(0, 0, 0) };
  var pointers = new Map();   // pointerId -> {x, y}
  var mode = null;            // "rotate" | "pan"
  var lastX = 0, lastY = 0;
  var panMid = { x: 0, y: 0 }, pinchDist = 0;

  function cameraBasis() {
    var fwd = new THREE.Vector3().subVectors(camState.target, camera.position).normalize();
    var right = new THREE.Vector3().crossVectors(fwd, camera.up).normalize();
    var up = new THREE.Vector3().crossVectors(right, fwd).normalize();
    return { right: right, up: up };
  }
  function panBy(dx, dy) {
    var b = cameraBasis(), s = camState.radius * 0.0016;
    camState.target.addScaledVector(b.right, -dx * s);
    camState.target.addScaledVector(b.up, dy * s);
  }

  viewportEl.addEventListener("pointerdown", function (e) {
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    viewportEl.setPointerCapture(e.pointerId);
    if (pointers.size === 1) {
      mode = (e.button === 2 || e.button === 1 || e.shiftKey) ? "pan" : "rotate";
      lastX = e.clientX; lastY = e.clientY;
    } else if (pointers.size === 2) {
      mode = "pan";
      var p = Array.from(pointers.values());
      panMid = { x: (p[0].x + p[1].x) / 2, y: (p[0].y + p[1].y) / 2 };
      pinchDist = Math.hypot(p[1].x - p[0].x, p[1].y - p[0].y);
    }
    viewportEl.classList.add("dragging");
  });
  viewportEl.addEventListener("pointermove", function (e) {
    if (!pointers.has(e.pointerId)) return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 1) {
      var dx = e.clientX - lastX, dy = e.clientY - lastY; lastX = e.clientX; lastY = e.clientY;
      if (mode === "pan") panBy(dx, dy);
      else {
        camState.az -= dx * 0.0055;
        camState.pol = Math.max(0.12, Math.min(Math.PI - 0.12, camState.pol - dy * 0.0055));
      }
    } else if (pointers.size === 2) {
      var p = Array.from(pointers.values());
      var mx = (p[0].x + p[1].x) / 2, my = (p[0].y + p[1].y) / 2;
      panBy(mx - panMid.x, my - panMid.y);
      panMid = { x: mx, y: my };
      var d = Math.hypot(p[1].x - p[0].x, p[1].y - p[0].y);
      if (pinchDist > 0) {
        camState.radius = Math.max(maxDim * 0.15, Math.min(maxDim * 6, camState.radius * (pinchDist / d)));
      }
      pinchDist = d;
    }
  });
  function endPointer(e) {
    pointers.delete(e.pointerId);
    if (pointers.size === 0) {
      mode = null;
      viewportEl.classList.remove("dragging");
    } else if (pointers.size === 1) {
      var p = Array.from(pointers.values())[0];
      lastX = p.x; lastY = p.y;
      mode = "rotate";
    }
  }
  window.addEventListener("pointerup", endPointer);
  window.addEventListener("pointercancel", endPointer);
  viewportEl.addEventListener("contextmenu", function (e) { e.preventDefault(); });

  viewportEl.addEventListener("wheel", function (e) {
    e.preventDefault();
    camState.radius = Math.max(maxDim * 0.15, Math.min(maxDim * 6, camState.radius * (1 + e.deltaY * 0.0012)));
  }, { passive: false });

  function updateCamera() {
    var r = camState.radius, az = camState.az, pol = camState.pol;
    camera.position.set(camState.target.x + r * Math.sin(pol) * Math.sin(az),
                         camState.target.y + r * Math.cos(pol),
                         camState.target.z + r * Math.sin(pol) * Math.cos(az));
    camera.lookAt(camState.target);
  }
  window.addEventListener("resize", function () {
    camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  (function animate() {
    requestAnimationFrame(animate);
    updateCamera();
    renderer.render(scene, camera);
  })();
})();
