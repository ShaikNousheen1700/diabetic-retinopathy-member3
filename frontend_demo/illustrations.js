/* ==========================================================================
   Conceptual retina illustrations (SVG), drawn in code.
   retinaSVG(stage) returns an SVG string for stage 0–4:
     0 healthy · 1 tiny red dots (microaneurysms) · 2 + small bleeds and yellow deposits
     3 + many bleeds across the retina · 4 + fragile new vessels near the optic disc
   These are simplified drawings for education — NOT real images, NOT diagnostic.
   A fixed random seed per stage keeps every drawing identical on each page load.
   ========================================================================== */

(function () {
  let uid = 0;

  // Small deterministic random generator (same seed -> same "random" positions)
  function seeded(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Random point inside the retina, away from the optic disc
  function point(rand, maxR = 78) {
    for (;;) {
      const a = rand() * Math.PI * 2, r = Math.sqrt(rand()) * maxR;
      const x = 100 + r * Math.cos(a), y = 100 + r * Math.sin(a);
      if (Math.hypot(x - 140, y - 96) > 22) return [x, y];
    }
  }

  // Main vessels leaving the optic disc (at 140, 96)
  const VESSELS = [
    ["M140 92 C 120 70, 95 58, 60 48 S 22 50, 14 62", 2.8],
    ["M140 100 C 120 124, 95 138, 60 150 S 24 150, 14 138", 2.8],
    ["M140 90 C 128 66, 118 46, 112 22", 2.2],
    ["M140 102 C 128 128, 118 150, 114 178", 2.2],
    ["M138 94 C 112 84, 96 82, 70 86", 1.4],
    ["M138 99 C 112 110, 96 114, 70 112", 1.4],
    ["M95 60 C 80 70, 66 76, 40 82", 1.2],
    ["M95 140 C 80 130, 66 124, 40 118", 1.2],
    ["M150 90 C 165 74, 176 66, 186 70", 1.6],
    ["M150 102 C 165 120, 176 130, 186 128", 1.6],
  ];

  function retinaSVG(stage, opts = {}) {
    const id = "r" + (++uid);
    const rand = seeded(17 + stage * 101);
    const parts = [];

    // stage 1+: microaneurysms (tiny red dots)
    if (stage >= 1) {
      const n = [0, 7, 12, 16, 16][stage];
      for (let i = 0; i < n; i++) {
        const [x, y] = point(rand);
        parts.push(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2" fill="#a51414"/>`);
      }
    }
    // stage 2+: hard exudates (yellow deposits) and small bleeds
    if (stage >= 2) {
      const ex = [0, 0, 12, 20, 18][stage];
      for (let i = 0; i < ex; i++) {
        const [x, y] = point(rand, 60);
        parts.push(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${(1.2 + rand() * 1.6).toFixed(1)}" fill="#f6dd7a" opacity=".95"/>`);
      }
      const hb = [0, 0, 6, 16, 14][stage];
      for (let i = 0; i < hb; i++) {
        const [x, y] = point(rand);
        const r = 2.2 + rand() * (stage >= 3 ? 3.4 : 2);
        parts.push(`<ellipse cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" rx="${r.toFixed(1)}" ry="${(r * .75).toFixed(1)}" fill="#6e0d0d" opacity=".85"/>`);
      }
      // cotton-wool spots (pale, soft patches)
      const cw = [0, 0, 2, 4, 3][stage];
      for (let i = 0; i < cw; i++) {
        const [x, y] = point(rand, 64);
        parts.push(`<ellipse cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" rx="5" ry="3.4" fill="#fbeedd" opacity=".75" filter="url(#${id}-soft)"/>`);
      }
    }
    // stage 4: new fragile vessels near the disc + a larger bleed
    if (stage >= 4) {
      parts.push(`<g fill="none" stroke="#9b1c1c" stroke-width=".9" opacity=".95">
        <path d="M128 78 q6 -8 12 -2 t10 4 t-4 10 t-12 -2"/>
        <path d="M150 112 q8 4 6 12 t-10 6 t-6 -10 t8 -6"/>
        <path d="M124 104 q-6 6 -2 12 t10 2 t2 -10"/>
        <path d="M146 76 q-2 -8 6 -10 t10 6"/>
      </g>`);
      parts.push(`<path d="M70 150 C 80 140, 104 140, 112 152 C 104 160, 80 160, 70 150 Z" fill="#5c0a0a" opacity=".8"/>`);
    }

    const vessels = VESSELS.map(([d, w]) =>
      `<path d="${d}" stroke="#8c2a1a" stroke-width="${w}" fill="none" stroke-linecap="round" opacity=".85"/>`).join("");

    return `<svg viewBox="0 0 200 200" role="img" aria-label="${opts.label || "Conceptual retina illustration"}">
      <defs>
        <radialGradient id="${id}-bg" cx="45%" cy="48%" r="60%">
          <stop offset="0" stop-color="#e9a66b"/><stop offset=".6" stop-color="#c8643b"/><stop offset="1" stop-color="#8f3a1f"/>
        </radialGradient>
        <radialGradient id="${id}-disc"><stop offset="0" stop-color="#fff2cf"/><stop offset=".7" stop-color="#f3c98b"/><stop offset="1" stop-color="#e9a66b" stop-opacity="0"/></radialGradient>
        <radialGradient id="${id}-mac"><stop offset="0" stop-color="#7a2e16" stop-opacity=".75"/><stop offset="1" stop-color="#7a2e16" stop-opacity="0"/></radialGradient>
        <filter id="${id}-soft"><feGaussianBlur stdDeviation="1.2"/></filter>
        <clipPath id="${id}-clip"><circle cx="100" cy="100" r="96"/></clipPath>
      </defs>
      <circle cx="100" cy="100" r="98" fill="#0d0b0a"/>
      <g clip-path="url(#${id}-clip)">
        <circle cx="100" cy="100" r="96" fill="url(#${id}-bg)"/>
        <circle cx="84" cy="102" r="20" fill="url(#${id}-mac)"/>
        ${vessels}
        <circle cx="140" cy="96" r="17" fill="url(#${id}-disc)"/>
        ${parts.join("")}
      </g>
    </svg>`;
  }

  window.retinaSVG = retinaSVG;
})();
