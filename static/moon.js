/*
 * Live moon background: a real-time WebGL2 shader, no libraries.
 *
 * It ray-casts a sphere per pixel, wraps NASA's Lunar Reconnaissance Orbiter colour map around it, lights it
 * with a normal map built from LOLA laser-altimeter heights, and uses the Lommel-Seeliger law (how the real
 * Moon scatters sunlight: no limb darkening, bright relief at grazing light). The moon turns slowly, the stars
 * twinkle and drift, and the whole thing eases toward the mouse a little. If WebGL2 is missing the page keeps its
 * static background image.
 *
 * Imagery: NASA/GSFC/Arizona State University (LROC colour), NASA/GSFC/MIT/SVS (LOLA elevation), public domain.
 */
(function () {
  "use strict";
  var self = document.currentScript;
  var BASE = (self && self.dataset.base) || "/app/static/";
  var doc = document;
  if (window.__moon) return; // already running (Streamlit reruns re-inject this file)

  var host = doc.querySelector(".stApp") || doc.body;
  var canvas = doc.createElement("canvas");
  canvas.id = "moon-bg";
  canvas.setAttribute("aria-hidden", "true");
  canvas.style.cssText = "position:fixed;inset:0;width:100vw;height:100vh;z-index:0;pointer-events:none;display:block;";
  var gl = canvas.getContext("webgl2", { antialias: false, alpha: false, powerPreference: "high-performance" });
  if (!gl) return;
  host.insertBefore(canvas, host.firstChild);

  var reduced = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;

  var VERT = "#version 300 es\nin vec2 a;\nvoid main(){gl_Position=vec4(a,0.,1.);}\n";

  var FRAG = [
    "#version 300 es",
    "precision highp float;",
    "uniform vec2 u_res; uniform float u_time; uniform vec2 u_center; uniform float u_radius; uniform float u_dpr;",
    "uniform mat3 u_R; uniform vec3 u_L; uniform vec2 u_par;",
    "uniform sampler2D u_color; uniform sampler2D u_normal;",
    "out vec4 outColor;",
    "const float PI = 3.14159265359;",
    "float hash(vec2 p){ p = fract(p*vec2(123.34, 456.21)); p += dot(p, p+45.32); return fract(p.x*p.y); }",
    "float noise(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.-2.*f);",
    "  return mix(mix(hash(i),hash(i+vec2(1,0)),f.x), mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x), f.y); }",
    "float fbm(vec2 p){ float s=0., a=.5; for(int i=0;i<5;i++){ s+=a*noise(p); p*=2.03; a*=.5; } return s; }",
    // one layer of stars: a point per grid cell, most cells empty, each with its own size, tint and twinkle
    "vec3 stars(vec2 px, float cell, float density, float seed){",
    "  vec2 g = px / cell; vec2 id = floor(g); vec2 f = fract(g) - .5;",
    "  float h = hash(id + seed); if (h > density) return vec3(0.);",
    "  vec2 off = (vec2(hash(id+seed+1.7), hash(id+seed+9.2)) - .5) * .7;",
    "  float d = length(f - off);",
    "  float size = mix(.012, .05, pow(hash(id+seed+3.3), 3.)) * (cell/40.);",   // mostly pin-points, a few brighter ones
    "  float tw = .72 + .28*sin(u_time*(.8+3.*hash(id+seed+5.1)) + 6.2831*hash(id+seed+7.7));",
    "  float b = smoothstep(size, 0., d) * tw * (.35 + .65*hash(id+seed+2.2));",
    "  vec3 tint = mix(vec3(.75,.85,1.), vec3(1.,.9,.75), hash(id+seed+4.4));",
    "  return tint * b * 2.4;",
    "}",
    "void main(){",
    "  vec2 frag = gl_FragCoord.xy;",
    "  vec2 p = (frag - u_center) / u_radius;",
    "  float r = length(p);",
    // ---- sky: faint nebula, two star layers drifting at different speeds ----
    "  vec3 col = vec3(.0016,.0018,.0028);",
    "  if (r > .985) {",                                 // sky is only needed outside the moon; skip the work behind it
    "    vec2 sp = frag + u_par * 18. + vec2(u_time*1.6, u_time*.5);",
    "    float n = fbm(frag/(u_res.y*.9) + vec2(1.7,.3) + u_time*.002);",
    "    col += vec3(.03,.055,.13) * smoothstep(.45,.85,n) * .22 * (1. - smoothstep(0., 1.2, frag.x/u_res.x + (1.-frag.y/u_res.y)*.2));",
    "    col += stars(sp*1.0, 52.*u_dpr, .20, 0.) + stars((sp+u_par*10.)*.62 + 77., 34.*u_dpr, .14, 11.)*.6;",
    "  }",
    // ---- thin atmosphere-like limb glow, strongest on the sunlit side ----
    "  float limb = r - 1.;",
    "  float sunSide = max(dot(normalize(p + 1e-5), normalize(u_L.xy + 1e-5)), 0.);",
    "  col += vec3(.22,.4,.85) * exp(-max(limb,0.)*38.) * (.04 + .22*sunSide) * step(0.,limb);",
    "  if (r < 1.0) {",
    "    float z = sqrt(1. - r*r);",
    "    vec3 N0 = vec3(p, z);",
    "    vec3 b = transpose(u_R) * N0;",              // body-space position on the unit sphere
    "    float lat = asin(clamp(b.y, -1., 1.));",
    "    float lon = atan(b.x, b.z);",
    "    float u = lon/(2.*PI) + .5, v = .5 - lat/PI;",
    // pick the smoother of two longitude parametrisations for the mip level, which hides the wrap-around seam
    "    float u2 = fract(u + .5);",
    "    float w1 = abs(dFdx(u)) + abs(dFdy(u)), w2 = abs(dFdx(u2)) + abs(dFdy(u2));",
    "    vec2 ddx = w1 <= w2 ? vec2(dFdx(u), dFdx(v)) : vec2(dFdx(u2), dFdx(v));",
    "    vec2 ddy = w1 <= w2 ? vec2(dFdy(u), dFdy(v)) : vec2(dFdy(u2), dFdy(v));",
    "    vec2 uv = vec2(u, v);",
    "    vec3 alb = textureGrad(u_color, uv, ddx, ddy).rgb;",
    "    vec3 nm  = textureGrad(u_normal, uv, ddx, ddy).rgb * 2. - 1.;",
    "    vec3 east = normalize(vec3(b.z, 0., -b.x) + vec3(1e-5,0.,0.));",
    "    vec3 north = cross(b, east);",
    "    vec3 nb = normalize(east*nm.x + north*nm.y + b*nm.z);",
    "    vec3 n = normalize(u_R * nb);",                // perturbed normal back in view space
    "    vec3 V = vec3(0.,0.,1.);",
    "    float mu0 = max(dot(n, u_L), 0.);",
    "    float mu  = max(dot(n, V), .05);",
    "    float ls = mu0 / (mu0 + mu);",                  // Lommel-Seeliger: how a dusty airless surface scatters light
    "    float geo = smoothstep(-.03, .1, dot(N0, u_L));",
    "    vec3 lin = pow(alb, vec3(2.2));",
    "    vec3 sunlit = lin * ls * geo * 8.2;",
    "    vec3 earthshine = lin * vec3(.45,.62,1.) * .035;",
    "    vec3 moon = sunlit + earthshine;",
    "    float edge = 1. - smoothstep(1. - 1.5/u_radius, 1., r);",   // 1.5px anti-aliased rim (edge0 must be < edge1)
    "    col = mix(col, moon, edge);",
    "  }",
    "  col = 1. - exp(-col * 1.15);",                    // gentle filmic roll-off so highlights don't clip
    "  col = pow(col, vec3(1./2.2));",
    "  col += (hash(frag + u_time) - .5) / 255.;",       // dither: no banding in the dark gradients
    "  outColor = vec4(col, 1.);",
    "}"
  ].join("\n");

  function compile(type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.warn("[moon] shader error:", gl.getShaderInfoLog(s));
      return null;
    }
    return s;
  }
  var vs = compile(gl.VERTEX_SHADER, VERT), fs = compile(gl.FRAGMENT_SHADER, FRAG);
  if (!vs || !fs) { canvas.remove(); return; }
  var prog = gl.createProgram();
  gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { canvas.remove(); return; }
  gl.useProgram(prog);

  var buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  var loc = gl.getAttribLocation(prog, "a");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  var U = {};
  ["u_res", "u_time", "u_center", "u_radius", "u_dpr", "u_R", "u_L", "u_par", "u_color", "u_normal"].forEach(function (n) {
    U[n] = gl.getUniformLocation(prog, n);
  });

  // ---- textures: pick sizes the GPU and screen can handle ----
  var maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  var small = Math.min(screen.width, screen.height) < 700;
  var colorFile = (maxTex >= 8192 && !small) ? "moon_color_8k.jpg" : "moon_color_4k.jpg";
  var normalFile = (maxTex >= 4096 && !small) ? "moon_normal_4k.jpg" : "moon_normal_2k.jpg";
  var aniso = gl.getExtension("EXT_texture_filter_anisotropic");

  function loadTexture(unit, file, onReady) {
    var tex = gl.createTexture();
    var img = new Image();
    img.onload = function () {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      if (aniso) gl.texParameterf(gl.TEXTURE_2D, aniso.TEXTURE_MAX_ANISOTROPY_EXT,
        Math.min(8, gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT)));
      onReady();
    };
    img.onerror = function () { console.warn("[moon] could not load", file); };
    img.src = BASE + file;
  }

  var ready = 0, started = false, fade = 0;
  function oneReady() { ready++; if (ready === 2 && !started) { started = true; requestAnimationFrame(frame); } }
  loadTexture(0, colorFile, oneReady);
  loadTexture(1, normalFile, oneReady);
  gl.uniform1i(U.u_color, 0);
  gl.uniform1i(U.u_normal, 1);

  // ---- layout: the moon sits right of centre and bleeds off the edge, like a planet behind the page ----
  var W = 0, H = 0, dpr = 1, cx = 0, cy = 0, rad = 1;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    W = Math.max(2, Math.floor(window.innerWidth * dpr));
    H = Math.max(2, Math.floor(window.innerHeight * dpr));
    canvas.width = W; canvas.height = H;
    gl.viewport(0, 0, W, H);
    var land = W >= H * 1.1;
    rad = land ? Math.min(H * 0.64, W * 0.42) : Math.min(W * 0.78, H * 0.4);
    cx = land ? W * 0.80 : W * 0.84;
    cy = land ? H * 0.50 : H * 0.70; // gl_FragCoord origin is bottom-left, so 0.70 puts the moon in the upper part on phones
  }
  window.addEventListener("resize", resize);
  resize();

  // ---- motion ----
  var mx = 0, my = 0, tx = 0, ty = 0;
  window.addEventListener("pointermove", function (e) {
    tx = (e.clientX / window.innerWidth - 0.5) * 2;
    ty = (e.clientY / window.innerHeight - 0.5) * -2;
  }, { passive: true });

  function rotY(a) { var c = Math.cos(a), s = Math.sin(a); return [c, 0, -s, 0, 1, 0, s, 0, c]; }       // column-major
  function rotX(a) { var c = Math.cos(a), s = Math.sin(a); return [1, 0, 0, 0, c, s, 0, -s, c]; }
  function mul(A, B) { // A*B, column-major 3x3
    var o = new Array(9);
    for (var c = 0; c < 3; c++) for (var r = 0; r < 3; r++) {
      o[c * 3 + r] = A[0 * 3 + r] * B[c * 3 + 0] + A[1 * 3 + r] * B[c * 3 + 1] + A[2 * 3 + r] * B[c * 3 + 2];
    }
    return o;
  }

  var t0 = performance.now(), last = t0, spin = 0.3, SPEED = reduced ? 0 : 0.0125; // radians/s: one turn in ~8 min
  function frame(now) {
    var dt = Math.min((now - last) / 1000, 0.25); last = now;
    var t = (now - t0) / 1000;
    spin += dt * SPEED;
    mx += (tx - mx) * Math.min(dt * 2.0, 1); my += (ty - my) * Math.min(dt * 2.0, 1);
    fade = Math.min(fade + dt * 0.8, 1);

    var tilt = 0.12 + 0.03 * Math.sin(t * 0.05);           // slow libration
    var R = mul(rotX(tilt + my * 0.04), rotY(spin + mx * 0.05));
    var Lx = -0.88 + mx * 0.05, Ly = 0.20 + my * 0.04, Lz = 0.22;
    var Ln = Math.hypot(Lx, Ly, Lz);

    gl.uniform2f(U.u_res, W, H);
    gl.uniform1f(U.u_time, t);
    gl.uniform2f(U.u_center, cx + mx * 14 * dpr, cy + my * 10 * dpr);
    gl.uniform1f(U.u_radius, rad);
    gl.uniform1f(U.u_dpr, dpr);
    gl.uniformMatrix3fv(U.u_R, false, new Float32Array(R));
    gl.uniform3f(U.u_L, Lx / Ln, Ly / Ln, Lz / Ln);
    gl.uniform2f(U.u_par, mx, my);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    if (fade < 1) canvas.style.opacity = String(fade);
    else if (canvas.style.opacity !== "1") canvas.style.opacity = "1";
    requestAnimationFrame(frame);
  }
  canvas.style.opacity = "0";
  doc.documentElement.classList.add("moon-live");
  window.__moon = { canvas: canvas };
})();
