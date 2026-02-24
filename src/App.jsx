import { useState, useEffect, createContext, useContext, useCallback } from "react";

// ─── KaTeX dynamic loader ─────────────────────────────────────────────────────
const KatexCtx = createContext(false);

function useKatexLoader() {
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    if (window.katex) { setLoaded(true); return; }
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css";
    link.crossOrigin = "anonymous";
    document.head.appendChild(link);
    const script = document.createElement("script");
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js";
    script.crossOrigin = "anonymous";
    script.onload = () => setLoaded(true);
    script.onerror = () => console.warn("KaTeX failed to load — math will render as code");
    document.head.appendChild(script);
  }, []);
  return loaded;
}

// T — renders LaTeX, falls back to <code> gracefully
function T({ tex, d = false, style }) {
  const loaded = useContext(KatexCtx);
  if (!loaded) return <code style={{ fontFamily: "monospace", fontSize: "0.85em", opacity: 0.8, ...style }}>{tex}</code>;
  try {
    const html = window.katex.renderToString(tex, { throwOnError: false, displayMode: !!d });
    return <span style={style} dangerouslySetInnerHTML={{ __html: html }} />;
  } catch (e) {
    return <code style={{ color: "#f87171", fontFamily: "monospace", fontSize: "0.8em", ...style }}>{tex}</code>;
  }
}

// Dim — dimension badge (defined at module level to avoid React remount issues)
function Dim({ tex }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center",
      background: "rgba(255,255,255,0.035)", border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 6, padding: "3px 9px", margin: "2px 3px",
    }}>
      <T tex={tex} />
    </span>
  );
}

// ─── Constants ────────────────────────────────────────────────────────────────
const ACTIVATIONS = ["ReLU", "Sigmoid", "Tanh", "Linear"];

const ACT = {
  ReLU:    { color: "#fb923c", tex: "\\max(0,z)",
             desc: () => <>Zeroes negative pre-activations; cheap and avoids saturation.</> },
  Sigmoid: { color: "#22d3ee", tex: "\\dfrac{1}{1+e^{-z}}",
             desc: () => <>Squashes to <T tex="(0,1)" />. Prone to vanishing gradients in deep nets.</> },
  Tanh:    { color: "#c084fc", tex: "\\dfrac{e^{z}-e^{-z}}{e^{z}+e^{-z}}",
             desc: () => <>Zero-centred squash to <T tex="(-1,1)" />. Stronger gradients than sigmoid.</> },
  Linear:  { color: "#94a3b8", tex: "z",
             desc: () => <>Identity &#8212; no non-linearity. Collapses to a single affine map.</> },
  Softmax: { color: "#34d399", tex: "\\dfrac{e^{z_i}}{\\sum_j e^{z_j}}",
             desc: () => <>Normalises logits to a probability simplex.</> },
};

const MAX_NODES_SHOWN = 6;
const MAX_NODES  = 8;
const MIN_NODES  = 1;
const MAX_LAYERS = 6;

function defaultLayers() {
  return [
    { nodes: 3, activation: "Linear"  },
    { nodes: 5, activation: "ReLU"    },
    { nodes: 4, activation: "Tanh"    },
    { nodes: 2, activation: "Softmax" },
  ];
}

// ─── Loss catalogue ───────────────────────────────────────────────────────────
// note / gradNote / body are all () => JSX so math renders via <T>.
const LOSSES = {
  ce: {
    label: "Cross-Entropy",
    tag: "Standard multiclass",
    color: "#34d399",
    requiresBinary: false,
    params: [],
    fullFormula: (K) => `\\mathcal{L}(\\hat{y}, y) = -\\sum_{i=1}^{${K}} y_i \\log \\hat{y}_i`,
    gradFormula: () => `\\frac{\\partial \\mathcal{L}}{\\partial z^{(L)}_i} = \\hat{y}_i - y_i`,
    gradNote: () => <>Clean residual — all exponentials cancel. This is why softmax&nbsp;+&nbsp;CE is the standard choice.</>,
    note: (K) => <><T tex="\hat{y}" /> is a one-hot vector in <T tex={`\\mathbb{R}^{${K}}`} />. Only the term for the correct class survives the sum.</>,
    derivation: [
      { heading: "Maximum-likelihood view",
        body: () => <>Assume the true class is drawn from a categorical distribution parameterised by the network output. The log-likelihood of the true label <T tex="y" /> under that distribution is <T tex="\log \hat{y}_{y^*}" />. Minimising the negative log-likelihood is therefore equivalent to minimising the cross-entropy.</> },
      { heading: "Information-theoretic view",
        body: () => <>Cross-entropy <T tex="H(p,q) = -\sum_x p(x)\log q(x)" /> measures expected bits to encode events from <T tex="p" /> using a code optimised for <T tex="q" />. Minimising CE pushes the model distribution <T tex="q = \hat{y}" /> towards the data distribution <T tex="p = y" />.</> },
      { heading: "Why softmax + CE?",
        body: () => <>The gradient of CE with respect to the pre-softmax logits <T tex="z" /> simplifies to <T tex="\hat{y}_i - y_i" />. No exponentials remain — the training signal is just the residual between predicted and true probability. This is unusually clean compared to other loss/output combinations.</> },
    ],
  },

  bce: {
    label: "Binary CE",
    tag: "Binary (K = 2)",
    color: "#38bdf8",
    requiresBinary: true,
    params: [],
    fullFormula: () => `\\mathcal{L}(\\hat{y}, y) = -y\\log\\hat{y} - (1-y)\\log(1-\\hat{y})`,
    gradFormula: () => `\\frac{\\partial \\mathcal{L}}{\\partial z} = \\hat{y} - y`,
    gradNote: () => <>Same elegant residual form as multiclass CE, now with a single sigmoid output.</>,
    note: () => <>Use when <T tex="K = 2" />. Replace the softmax head with a single sigmoid neuron.</>,
    derivation: [
      { heading: "Bernoulli log-likelihood",
        body: () => <>For binary classification the output is a single sigmoid neuron <T tex="\hat{y} \in (0,1)" /> representing <T tex="P(y=1 \mid x)" />. The Bernoulli likelihood is <T tex="\hat{y}^{\,y}(1-\hat{y})^{1-y}" />. Taking the negative log gives the BCE formula — it is the <T tex="K=2" /> special case of cross-entropy.</> },
      { heading: "Relationship to multiclass CE",
        body: () => <>BCE with a single sigmoid output is mathematically equivalent to CE with a 2-neuron softmax. The sigmoid is computationally cheaper for binary problems, and its gradient has the same residual form <T tex="\hat{y} - y" />.</> },
    ],
  },

  focal: {
    label: "Focal Loss",
    tag: "Class imbalance",
    color: "#f472b6",
    requiresBinary: false,
    params: [{ key: "gamma", label: "\u03b3 (focusing)", min: 0, max: 5, step: 0.5, default: 2 }],
    fullFormula: (K, p) => `\\mathcal{L} = -\\sum_{i=1}^{${K}} y_i\\,(1-\\hat{y}_i)^{${p.gamma}}\\log \\hat{y}_i`,
    gradFormula: (_, p) => `\\frac{\\partial \\mathcal{L}}{\\partial z_i} = (1-\\hat{y}_{y^*})^{${p.gamma}}(\\hat{y}_i - y_i) - ${p.gamma}(1-\\hat{y}_{y^*})^{${Math.max(0, p.gamma - 1)}}\\log(\\hat{y}_{y^*})\\hat{y}_{y^*}(\\delta_{i,y^*} - \\hat{y}_i)`,
    gradNote: () => <>The modulating factor suppresses gradient magnitude for well-classified (easy) examples.</>,
    note: (_, p) => <><T tex={`\\gamma = ${p.gamma}`} />. When <T tex="\gamma > 0" />, easy examples (<T tex="\hat{y}" /> close to 1) contribute very little loss.</>,
    derivation: [
      { heading: "Motivation: easy negatives dominate",
        body: () => <>In heavily imbalanced datasets, correctly-classified background examples dominate the gradient and overwhelm signal from rare positives. The modulating factor <T tex="(1-\hat{y})^\gamma" /> scales down well-classified examples so training focuses on hard ones.</> },
      { heading: "Effect of \u03b3",
        body: (p) => <><T tex="\gamma = 0" /> recovers standard CE. As <T tex="\gamma" /> increases, hard examples get relatively more weight. <T tex="\gamma = 2" /> is the standard recommendation (Lin et al., 2017, RetinaNet). For a correct prediction with <T tex="\hat{y} = 0.9" />, the modulating factor is <T tex="(0.1)^2 = 0.01" /> — a 100&times; down-weight versus CE.</> },
    ],
  },

  ls: {
    label: "Label Smoothing",
    tag: "Regularisation",
    color: "#a78bfa",
    requiresBinary: false,
    params: [{ key: "eps", label: "\u03b5 (smoothing)", min: 0.01, max: 0.3, step: 0.01, default: 0.1 }],
    fullFormula: (K, p) => `\\tilde{y}_i = (1-${p.eps})y_i + \\tfrac{${p.eps}}{${K}},\\quad \\mathcal{L} = -\\sum_{i=1}^{${K}} \\tilde{y}_i \\log \\hat{y}_i`,
    gradFormula: () => `\\frac{\\partial \\mathcal{L}}{\\partial z^{(L)}_i} = \\hat{y}_i - \\tilde{y}_i`,
    gradNote: () => <>Same residual form, but the smoothed target <T tex="\tilde{y}" /> puts a floor — prevents logits from growing unboundedly.</>,
    note: (K, p) => <><T tex={`\\varepsilon = ${p.eps}`} />. Wrong-class target: <T tex={`${(p.eps / K).toFixed(4)}`} />. Correct-class target: <T tex={`${(1 - p.eps + p.eps / K).toFixed(4)}`} />.</>,
    derivation: [
      { heading: "The overconfidence problem",
        body: () => <>Standard CE encourages the model to push the correct-class logit to <T tex="+\infty" /> (<T tex="\hat{y} \to 1" />, all others <T tex="\to 0" />). This hurts calibration and can cause poor generalisation since the model becomes arbitrarily confident.</> },
      { heading: "Smoothing as a prior",
        body: (_, K) => <>Label smoothing redistributes a small mass <T tex="\varepsilon" /> uniformly across all <T tex="K" /> classes. The correct-class target becomes <T tex="1 - \varepsilon + \varepsilon/K" /> instead of 1, and <T tex="\varepsilon/K" /> for wrong classes. This is equivalent to adding <T tex="\mathrm{KL}(\mathrm{Uniform}\,\|\,\hat{y})" /> to the loss, acting as a calibration regulariser.</> },
      { heading: "Effect on gradients",
        body: () => <>The gradient is <T tex="\hat{y}_i - \tilde{y}_i" />. The smoothed target <T tex="\tilde{y}" /> puts a floor on the reward signal — the network is prevented from becoming arbitrarily certain, improving held-out calibration.</> },
    ],
  },
};

// ─── Helpers ──────────────────────────────────────────────────────────────────
function computeParams(layers) {
  const breakdown = [];
  let totalW = 0, totalB = 0;
  for (let i = 1; i < layers.length; i++) {
    const nin = layers[i - 1].nodes, nout = layers[i].nodes;
    const W = nin * nout, B = nout;
    totalW += W; totalB += B;
    breakdown.push({ layer: i, nin, nout, W, B });
  }
  return { totalW, totalB, total: totalW + totalB, breakdown };
}

function seedVal(li, r, c) {
  const x = Math.sin(li * 7919 + r * 1013 + c * 431) * 43758.5453;
  return parseFloat(((x - Math.floor(x)) * 2 - 1).toFixed(2));
}

const btnSm = {
  width: 20, height: 20,
  background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)",
  color: "rgba(255,255,255,0.55)", borderRadius: 5, cursor: "pointer", fontSize: 14,
  display: "flex", alignItems: "center", justifyContent: "center", lineHeight: 1, padding: 0,
};

const secLabel = {
  fontSize: 9, fontWeight: 700, letterSpacing: "0.14em",
  textTransform: "uppercase", color: "rgba(255,255,255,0.22)", marginBottom: 10,
};

// ─── Callout ──────────────────────────────────────────────────────────────────
function Callout({ icon, color, title, children }) {
  return (
    <div style={{
      background: `${color}09`, border: `1px solid ${color}28`,
      borderLeft: `3px solid ${color}88`, borderRadius: "0 8px 8px 0",
      padding: "10px 14px", marginTop: 8,
    }}>
      <div style={{ fontSize: 11, fontWeight: 600, color, marginBottom: 5 }}>{icon} {title}</div>
      <div style={{ fontSize: 11, color: "rgba(255,255,255,0.52)", lineHeight: 1.7 }}>{children}</div>
    </div>
  );
}

// ─── Weight matrix grid ───────────────────────────────────────────────────────
function WeightMatrix({ nin, nout, layerIdx }) {
  const VR = Math.min(nout, 5), VC = Math.min(nin, 5);
  return (
    <div>
      <div style={{ fontSize: 10, color: "rgba(255,255,255,0.28)", fontFamily: "monospace", marginBottom: 6 }}>
        Illustrative entries of <T tex={`W^{(${layerIdx})} \\in \\mathbb{R}^{${nout}\\times${nin}}`} />
      </div>
      <div style={{ display: "inline-block", background: "rgba(0,0,0,0.35)", borderRadius: 8, padding: "8px 10px", border: "1px solid rgba(255,255,255,0.05)" }}>
        {Array.from({ length: VR }, (_, r) => (
          <div key={r} style={{ display: "flex", gap: 3, marginBottom: r < VR - 1 ? 3 : 0 }}>
            {Array.from({ length: VC }, (_, c) => {
              const v = seedVal(layerIdx, r, c);
              const a = 0.08 + Math.abs(v) * 0.26;
              return (
                <div key={c}
                  title={`W[${r+1},${c+1}]: weight from neuron ${c+1} (layer ${layerIdx-1}) to neuron ${r+1} (layer ${layerIdx})`}
                  style={{
                    width: 40, height: 26, borderRadius: 4,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 9, fontFamily: "monospace", cursor: "help",
                    background: `rgba(99,102,241,${a})`,
                    color: `rgba(180,190,255,${0.35 + Math.abs(v) * 0.55})`,
                    border: "1px solid rgba(99,102,241,0.13)",
                  }}>{v}</div>
              );
            })}
            {nin > 5 && <div style={{ width: 18, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, color: "rgba(255,255,255,0.18)" }}>&#8943;</div>}
          </div>
        ))}
        {nout > 5 && <div style={{ textAlign: "center", fontSize: 13, color: "rgba(255,255,255,0.18)", marginTop: 3 }}>&#8942;</div>}
      </div>
      <div style={{ fontSize: 9, color: "rgba(255,255,255,0.2)", fontFamily: "monospace", marginTop: 5 }}>
        Hover a cell for its meaning. Showing {VR}&times;{VC} of {nout}&times;{nin}.
      </div>
    </div>
  );
}

// ─── Transform card ───────────────────────────────────────────────────────────
function TransformCard({ fromLayer, toLayer, layerIdx, isLast, startOpen }) {
  const [open, setOpen]       = useState(startOpen);
  const [showMat, setShowMat] = useState(false);
  const [showWhy, setShowWhy] = useState(false);

  const ac    = toLayer.activation;
  const color = ACT[ac]?.color ?? "#94a3b8";
  const nin   = fromLayer.nodes;
  const nout  = toLayer.nodes;
  const l     = layerIdx;
  const lPrev = l - 1;   // NOTE: plain ASCII minus throughout — no Unicode

  return (
    <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 12, marginBottom: 10, overflow: "hidden" }}>

      {/* Header */}
      <button onClick={() => setOpen(o => !o)}
        style={{ width: "100%", background: "none", border: "none", cursor: "pointer", padding: "11px 14px", display: "flex", alignItems: "center", justifyContent: "space-between", color: "inherit" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: color, boxShadow: `0 0 6px ${color}88`, flexShrink: 0 }} />
          <span style={{ fontFamily: "monospace", fontSize: 12, fontWeight: 600, color: "rgba(255,255,255,0.78)" }}>Layer {l}</span>
          <T tex={`a^{(${lPrev})} \\to a^{(${l})}`} style={{ fontSize: 13 }} />
          <span style={{ fontSize: 10, background: `${color}22`, border: `1px solid ${color}44`, color, padding: "1px 9px", borderRadius: 20, fontFamily: "monospace", marginLeft: 4 }}>{ac}</span>
        </div>
        <span style={{ fontSize: 13, color: "rgba(255,255,255,0.3)", display: "inline-block", transform: open ? "rotate(180deg)" : "none", transition: "transform 0.2s" }}>&#9662;</span>
      </button>

      {open && (
        <div style={{ padding: "0 14px 14px" }}>

          {/* Dimension badges */}
          <div style={{ marginBottom: 12, lineHeight: 2.2 }}>
            <Dim tex={`W^{(${l})} \\in \\mathbb{R}^{${nout}\\times${nin}}`} />
            <Dim tex={`b^{(${l})} \\in \\mathbb{R}^{${nout}}`} />
            <Dim tex={`a^{(${lPrev})} \\in \\mathbb{R}^{${nin}}`} />
            <Dim tex={`z^{(${l})} \\in \\mathbb{R}^{${nout}}`} />
            <Dim tex={`a^{(${l})} \\in \\mathbb{R}^{${nout}}`} />
          </div>

          {/* Equations */}
          <div style={{ background: "rgba(0,0,0,0.3)", borderRadius: 8, padding: "14px 18px", borderLeft: `3px solid ${color}77`, display: "flex", flexDirection: "column", gap: 16 }}>
            <T d tex={`z^{(${l})} = W^{(${l})} a^{(${lPrev})} + b^{(${l})}`} />
            <T d tex={`a^{(${l})} = \\sigma\\!\\bigl(z^{(${l})}\\bigr), \\qquad \\sigma(z) = ${ACT[ac]?.tex ?? "z"}`} />
          </div>
          <div style={{ marginTop: 8, fontSize: 11, color: `${color}cc`, paddingLeft: 4 }}>{ACT[ac]?.desc()}</div>

          {/* Reasoning toggle */}
          <button onClick={() => setShowWhy(s => !s)}
            style={{ marginTop: 12, background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.1)", color: showWhy ? "#a5b4fc" : "rgba(255,255,255,0.4)", borderRadius: 7, padding: "5px 12px", cursor: "pointer", fontSize: 11, fontFamily: "monospace" }}>
            {showWhy ? "hide" : "show"} reasoning
          </button>

          {showWhy && (
            <div style={{ marginTop: 10 }}>

              <Callout icon="&#128290;" color="#6366f1" title={<>Why is <T tex={`W^{(${l})}`} /> shaped <T tex={`${nout} \\times ${nin}`} />?</>}>
                The weight matrix must map an input vector of size <strong style={{ color: "#94a3b8" }}>{nin}</strong> (the
                previous layer) to an output vector of size <strong style={{ color }}>{nout}</strong> (this layer).
                Multiplying a ({nout}&times;{nin}) matrix by a ({nin}&times;1) column vector yields a ({nout}&times;1) vector
                -- one scalar per output neuron.
                Entry <T tex={`W^{(${l})}_{ij}`} /> is the connection strength from neuron <em>j</em> in
                layer {lPrev} to neuron <em>i</em> in layer {l}.
                Row <em>i</em> of <T tex={`W^{(${l})}`} /> is therefore the learned "template"
                that neuron <em>i</em> matches against its {nin} inputs.
              </Callout>

              <Callout icon="&#128208;" color="#f59e0b" title={<>What does <T tex={`W^{(${l})}a^{(${lPrev})} + b^{(${l})}`} /> compute geometrically?</>}>
                <T tex={`W^{(${l})} a^{(${lPrev})} + b^{(${l})}`} /> is an <strong style={{ color: "#fcd34d" }}>affine transformation</strong>:
                a linear map (rotation, scaling, shearing) followed by a translation.
                For neuron <em>i</em>: <T tex={`z^{(${l})}_i = \\sum_{j=1}^{${nin}} W^{(${l})}_{ij} a^{(${lPrev})}_j + b^{(${l})}_i`} />.
                This is a weighted sum of all {nin} inputs, with the bias acting as a learnable threshold.
                Without the bias, every neuron's decision hyperplane is forced to pass through the origin.
              </Callout>

              <Callout icon="&#9889;" color={color} title={<>Why apply <T tex={`\\sigma_{\\text{${ac}}}`} /> after the linear step?</>}>
                {ac === "Linear" ? (
                  <>
                    <strong style={{ color: "#f87171" }}>Warning: no non-linearity here.</strong> Composing
                    purely linear layers is equivalent to a <em>single</em> linear layer regardless of depth --
                    the product <T tex={`W^{(L)}\\cdots W^{(1)}`} /> is just another matrix, with the same
                    expressiveness as a one-layer network. Linear activations are occasionally
                    useful in a <em>final regression head</em>, but not in hidden layers of a classifier.
                  </>
                ) : (
                  <>
                    Stacking affine maps without non-linearity collapses to a single affine map --
                    the network could only learn linear decision boundaries.
                    Applying <strong style={{ color }}>{ac}</strong> element-wise introduces the
                    non-linearity required to approximate complex functions.
                    By the Universal Approximation Theorem, a network with at least one hidden layer
                    using a non-linear activation can represent any continuous function on a compact domain.
                  </>
                )}
              </Callout>

              <Callout icon="&#8596;" color="#94a3b8" title="Step-by-step dimension flow">
                <div style={{ fontFamily: "monospace", fontSize: 11, lineHeight: 2.1 }}>
                  <div>
                    <T tex={`a^{(${lPrev})} \\in \\mathbb{R}^{${nin}}`} style={{ marginRight: 8 }} />
                    input to this layer
                  </div>
                  <div>
                    <T tex={`W^{(${l})} a^{(${lPrev})} \\in \\mathbb{R}^{${nout}}`} style={{ marginRight: 8 }} />
                    ({nout}&times;{nin}) &middot; ({nin}&times;1) = ({nout}&times;1)
                  </div>
                  <div>
                    <T tex={`z^{(${l})} = W^{(${l})} a^{(${lPrev})} + b^{(${l})} \\in \\mathbb{R}^{${nout}}`} style={{ marginRight: 8 }} />
                    add bias
                  </div>
                  <div>
                    <T tex={`a^{(${l})} = \\sigma(z^{(${l})}) \\in \\mathbb{R}^{${nout}}`} style={{ marginRight: 8 }} />
                    element-wise {ac}
                  </div>
                </div>
              </Callout>

              {isLast && (
                <Callout icon="&#127937;" color="#34d399" title="Output layer: why softmax?">
                  The final linear step produces raw <em>logits</em> <T tex={`z^{(${l})} \\in \\mathbb{R}^{${nout}}`} /> --
                  real numbers with no probabilistic meaning.
                  Softmax exponentiates and normalises them so all outputs are positive and sum to 1,
                  giving them a direct interpretation as class probabilities.
                  The exponential also amplifies differences between logits, producing confident predictions
                  when one class dominates.
                </Callout>
              )}
            </div>
          )}

          {/* Matrix toggle */}
          <button onClick={() => setShowMat(s => !s)}
            style={{ marginTop: 10, background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.1)", color: "rgba(255,255,255,0.4)", borderRadius: 7, padding: "5px 12px", cursor: "pointer", fontSize: 11, fontFamily: "monospace" }}>
            {showMat ? "hide" : "show"} weight matrix
          </button>
          {showMat && <div style={{ marginTop: 12 }}><WeightMatrix nin={nin} nout={nout} layerIdx={l} /></div>}
        </div>
      )}
    </div>
  );
}

// ─── Loss panel ───────────────────────────────────────────────────────────────
function LossPanel({ K }) {
  const [lossKey, setLossKey] = useState("ce");
  const [params, setParams]   = useState({ gamma: 2, eps: 0.1 });
  const [tab, setTab]         = useState(0); // 0 = formula, 1 = derivation

  const loss       = LOSSES[lossKey];
  const isDisabled = (key) => LOSSES[key].requiresBinary && K !== 2;

  // Auto-switch away from binary-only loss when K changes
  useEffect(() => {
    if (LOSSES[lossKey].requiresBinary && K !== 2) setLossKey("ce");
  }, [K, lossKey]);

  const setParam = (key, val) => setParams(p => ({ ...p, [key]: val }));

  return (
    <div style={{ background: "rgba(52,211,153,0.03)", border: "1px solid rgba(52,211,153,0.15)", borderRadius: 12, padding: "16px 18px", marginTop: 4 }}>
      <div style={{ fontWeight: 700, color: "#34d399", fontSize: 13, marginBottom: 14, letterSpacing: "-0.01em" }}>
        Classification Objective
      </div>

      {/* Loss selector */}
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 14 }}>
        {Object.entries(LOSSES).map(([key, l]) => {
          const disabled = isDisabled(key);
          const active   = lossKey === key;
          return (
            <button key={key}
              onClick={() => { if (!disabled) setLossKey(key); }}
              title={disabled ? `Requires K=2 (current K=${K})` : l.label}
              style={{
                background: active ? `${l.color}22` : "rgba(255,255,255,0.03)",
                border: `1px solid ${active ? `${l.color}66` : "rgba(255,255,255,0.08)"}`,
                borderRadius: 8, padding: "5px 12px",
                cursor: disabled ? "not-allowed" : "pointer",
                opacity: disabled ? 0.35 : 1, transition: "all 0.15s",
              }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: active ? l.color : "rgba(255,255,255,0.55)" }}>{l.label}</div>
              <div style={{ fontSize: 9, color: active ? `${l.color}99` : "rgba(255,255,255,0.25)", marginTop: 1 }}>{l.tag}</div>
            </button>
          );
        })}
      </div>

      {/* Hyperparameter sliders */}
      {loss.params.length > 0 && (
        <div style={{ background: "rgba(0,0,0,0.2)", borderRadius: 8, padding: "10px 14px", marginBottom: 14, display: "flex", flexDirection: "column", gap: 10 }}>
          {loss.params.map(p => (
            <div key={p.key}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", fontFamily: "monospace" }}>{p.label}</span>
                <span style={{ fontSize: 11, color: loss.color, fontFamily: "monospace", fontWeight: 700 }}>{params[p.key]}</span>
              </div>
              <input type="range" min={p.min} max={p.max} step={p.step} value={params[p.key]}
                onChange={e => setParam(p.key, parseFloat(e.target.value))}
                style={{ width: "100%", accentColor: loss.color, cursor: "pointer" }} />
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "rgba(255,255,255,0.2)", fontFamily: "monospace", marginTop: 2 }}>
                <span>{p.min}</span><span>{p.max}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Sub-tabs: Formula | Derivation */}
      <div style={{ display: "flex", gap: 4, marginBottom: 12 }}>
        {["Formula", "Derivation"].map((t, i) => (
          <button key={i} onClick={() => setTab(i)} style={{
            background: tab === i ? "rgba(255,255,255,0.07)" : "none",
            border: `1px solid ${tab === i ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.06)"}`,
            color: tab === i ? "rgba(255,255,255,0.8)" : "rgba(255,255,255,0.35)",
            borderRadius: 6, padding: "4px 12px", cursor: "pointer", fontSize: 11, fontWeight: 500,
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && (
        <div>
          {/* Loss formula */}
          <div style={{ background: "rgba(0,0,0,0.3)", borderRadius: 8, padding: "16px 18px", borderLeft: `3px solid ${loss.color}88`, marginBottom: 10 }}>
            <T d tex={loss.fullFormula(K, params)} />
          </div>

          {/* Note */}
          <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", lineHeight: 1.7, paddingLeft: 4, marginBottom: 12 }}>
            {loss.note(K, params)}
          </div>

          {/* Output layer */}
          <div style={{ paddingTop: 12, borderTop: "1px solid rgba(255,255,255,0.06)", marginBottom: 12, display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={secLabel}>Output layer</div>
            <T d tex={`\\hat{y} = \\operatorname{softmax}\\!\\left(z^{(L)}\\right) \\in \\Delta^{K-1}, \\quad K = ${K}`} />
            <div style={{ fontSize: 11, color: "rgba(255,255,255,0.28)", fontFamily: "monospace" }}>
              All {K} outputs satisfy <T tex={"\\hat{y}_i \\geq 0"} /> and <T tex={`\\sum_{i=1}^{${K}} \\hat{y}_i = 1`} />.
            </div>
          </div>

          {/* Gradient */}
          <div style={{ paddingTop: 12, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
            <div style={secLabel}>Gradient w.r.t. output logits</div>
            <div style={{ background: "rgba(0,0,0,0.25)", borderRadius: 7, padding: "12px 14px" }}>
              <T d tex={loss.gradFormula(K, params)} />
              <div style={{ marginTop: 8, fontSize: 11, color: "rgba(255,255,255,0.35)", lineHeight: 1.6 }}>
                {loss.gradNote()}
              </div>
            </div>
          </div>
        </div>
      )}

      {tab === 1 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {loss.derivation.map((d, i) => (
            <div key={i} style={{ background: "rgba(0,0,0,0.2)", borderRadius: 8, padding: "12px 14px", borderLeft: `2px solid ${loss.color}55` }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: loss.color, marginBottom: 6 }}>{d.heading}</div>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.55)", lineHeight: 1.75 }}>{d.body(params, K)}</div>
            </div>
          ))}

          {/* Comparison table */}
          <div style={{ marginTop: 6 }}>
            <div style={secLabel}>Quick comparison</div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.07)" }}>
                  {["Loss", "Best for", "Params"].map(h => (
                    <th key={h} style={{ padding: "5px 10px", textAlign: "left", fontSize: 9, letterSpacing: "0.08em", textTransform: "uppercase", color: "rgba(255,255,255,0.25)", fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(LOSSES).map(([key, l]) => (
                  <tr key={key} style={{ background: key === lossKey ? "rgba(255,255,255,0.03)" : "transparent", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                    <td style={{ padding: "7px 10px", color: l.color, fontWeight: key === lossKey ? 700 : 400 }}>{l.label}</td>
                    <td style={{ padding: "7px 10px", color: "rgba(255,255,255,0.45)", fontSize: 10 }}>
                      { key === "ce" ? "Balanced multiclass" : key === "bce" ? "Binary (K=2)" : key === "focal" ? "Imbalanced datasets" : "Noisy / overfit-prone" }
                    </td>
                    <td style={{ padding: "7px 10px", color: "rgba(255,255,255,0.3)", fontFamily: "monospace", fontSize: 9 }}>
                      {l.params.length === 0 ? "none" : l.params.map(p => p.label).join(", ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Param explainer modal ────────────────────────────────────────────────────
function ParamExplainer({ layers, onClose }) {
  const { totalW, totalB, total, breakdown } = computeParams(layers);
  return (
    <div onClick={onClose}
      style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.72)", backdropFilter: "blur(7px)", zIndex: 50, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div onClick={e => e.stopPropagation()}
        style={{ background: "#0d1827", border: "1px solid rgba(99,102,241,0.35)", borderRadius: 16, padding: 28, maxWidth: 660, width: "95vw", maxHeight: "88vh", overflowY: "auto", boxShadow: "0 40px 90px rgba(0,0,0,0.7)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 22 }}>
          <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: "white", letterSpacing: "-0.02em" }}>Parameter count -- how it is computed</h3>
          <button onClick={onClose} style={{ background: "none", border: "none", color: "rgba(255,255,255,0.35)", cursor: "pointer", fontSize: 22, lineHeight: 1 }}>&#215;</button>
        </div>
        <div style={{ background: "rgba(0,0,0,0.3)", borderRadius: 10, padding: "16px 20px", marginBottom: 24, borderLeft: "3px solid rgba(99,102,241,0.5)" }}>
          <div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", fontFamily: "monospace", marginBottom: 12, letterSpacing: "0.08em", textTransform: "uppercase" }}>General formula:</div>
          <T d tex="\text{Params} = \underbrace{\sum_{l=1}^{L} n_l \cdot n_{l-1}}_{\text{weights}} \;+\; \underbrace{\sum_{l=1}^{L} n_l}_{\text{biases}} \;=\; \sum_{l=1}^{L} n_l\,(n_{l-1} + 1)" />
        </div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
              {["Transition", "Tensor", "Formula", "Count"].map(h => (
                <th key={h} style={{ padding: "6px 12px", textAlign: h === "Count" ? "right" : "left", fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(255,255,255,0.28)", fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {breakdown.flatMap(({ layer, nin, nout, W, B }) => [
              <tr key={`w${layer}`} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                <td style={{ padding: "7px 12px", fontFamily: "monospace", fontSize: 11, color: "rgba(255,255,255,0.45)", whiteSpace: "nowrap" }}>
                  <T tex={`l_{${layer - 1}} \\to l_{${layer}}`} />
                </td>
                <td style={{ padding: "7px 12px" }}><T tex={`W^{(${layer})}`} /></td>
                <td style={{ padding: "7px 12px", fontFamily: "monospace", fontSize: 11, color: "rgba(255,255,255,0.38)" }}>
                  <T tex={`${nout} \\times ${nin}`} />
                </td>
                <td style={{ padding: "7px 14px", fontFamily: "monospace", fontSize: 12, fontWeight: 700, color: "#818cf8", textAlign: "right" }}>{W.toLocaleString()}</td>
              </tr>,
              <tr key={`b${layer}`} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)", background: "rgba(0,0,0,0.1)" }}>
                <td style={{ padding: "5px 12px" }} />
                <td style={{ padding: "5px 12px" }}><T tex={`b^{(${layer})}`} /></td>
                <td style={{ padding: "5px 12px", fontFamily: "monospace", fontSize: 11, color: "rgba(255,255,255,0.38)" }}>
                  <T tex={`${nout}`} />
                </td>
                <td style={{ padding: "5px 14px", fontFamily: "monospace", fontSize: 11, fontWeight: 600, color: "#a78bfa", textAlign: "right" }}>{B.toLocaleString()}</td>
              </tr>,
            ])}
          </tbody>
          <tfoot>
            <tr style={{ borderTop: "1px solid rgba(255,255,255,0.1)" }}>
              <td colSpan={3} style={{ padding: "9px 12px", fontFamily: "monospace", fontSize: 11, color: "rgba(255,255,255,0.38)" }}>Weights total</td>
              <td style={{ padding: "9px 14px", fontFamily: "monospace", fontSize: 12, fontWeight: 700, color: "#818cf8", textAlign: "right" }}>{totalW.toLocaleString()}</td>
            </tr>
            <tr>
              <td colSpan={3} style={{ padding: "5px 12px 9px", fontFamily: "monospace", fontSize: 11, color: "rgba(255,255,255,0.38)" }}>Biases total</td>
              <td style={{ padding: "5px 14px 9px", fontFamily: "monospace", fontSize: 12, fontWeight: 700, color: "#a78bfa", textAlign: "right" }}>{totalB.toLocaleString()}</td>
            </tr>
            <tr style={{ borderTop: "2px solid rgba(99,102,241,0.4)", background: "rgba(99,102,241,0.07)" }}>
              <td colSpan={3} style={{ padding: "10px 12px", fontFamily: "monospace", fontSize: 13, fontWeight: 700, color: "white" }}>Total parameters</td>
              <td style={{ padding: "10px 14px", fontFamily: "monospace", fontSize: 15, fontWeight: 700, color: "#6366f1", textAlign: "right" }}>{total.toLocaleString()}</td>
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
  );
}

// ─── Network SVG ──────────────────────────────────────────────────────────────
function NetworkSVG({ layers, selectedLayer, onSelect }) {
  const W = 540, H = 300, R = 14;

  const nodeColor = (li) => {
    if (li === 0) return "#6366f1";
    if (li === layers.length - 1) return "#34d399";
    return ACT[layers[li].activation]?.color ?? "#94a3b8";
  };

  const positions = (li, count) => {
    const x = (W / (layers.length + 1)) * (li + 1);
    const shown = Math.min(count, MAX_NODES_SHOWN);
    return Array.from({ length: shown }, (_, i) => ({
      x, y: (H / (shown + 1)) * (i + 1),
      ellipsis: i === MAX_NODES_SHOWN - 1 && count > MAX_NODES_SHOWN,
    }));
  };

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
      <defs>
        {layers.map((_, li) => (
          <filter key={li} id={`glow${li}`}>
            <feGaussianBlur stdDeviation="3.5" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        ))}
      </defs>

      {/* Edges */}
      {layers.slice(0, -1).map((layer, li) => {
        const from = positions(li, layer.nodes);
        const to   = positions(li + 1, layers[li + 1].nodes);
        const hi   = selectedLayer === li || selectedLayer === li + 1;
        return from.flatMap((f, fi) =>
          to.map((t, ti) => (
            <line key={`e${li}-${fi}-${ti}`}
              x1={f.x + R} y1={f.y} x2={t.x - R} y2={t.y}
              stroke={hi ? "rgba(99,102,241,0.38)" : "rgba(255,255,255,0.045)"}
              strokeWidth={hi ? 1.2 : 0.6}
              style={{ transition: "all 0.25s" }} />
          ))
        );
      })}

      {/* Nodes */}
      {layers.map((layer, li) => {
        const pts = positions(li, layer.nodes);
        const col = nodeColor(li);
        const sel = selectedLayer === li;
        return pts.map((p, ni) => (
          <g key={`n${li}-${ni}`} onClick={() => onSelect(li)} style={{ cursor: "pointer" }}>
            {sel && <circle cx={p.x} cy={p.y} r={R + 7} fill="none" stroke={col} strokeWidth={1} opacity={0.28} />}
            <circle cx={p.x} cy={p.y} r={R}
              fill={sel ? col : `${col}88`}
              filter={sel ? `url(#glow${li})` : undefined}
              style={{ transition: "all 0.2s" }} />
            <text x={p.x} y={p.y + 4} textAnchor="middle" fontSize={8} fontFamily="monospace"
              fill="rgba(255,255,255,0.9)" fontWeight="bold">
              {p.ellipsis ? "..." : li === 0 ? `x${ni + 1}` : `a${ni + 1}`}
            </text>
          </g>
        ));
      })}

      {/* Labels */}
      {layers.map((layer, li) => {
        const x   = (W / (layers.length + 1)) * (li + 1);
        const col = nodeColor(li);
        const sel = selectedLayer === li;
        return (
          <g key={`lbl${li}`} onClick={() => onSelect(li)} style={{ cursor: "pointer" }}>
            <text x={x} y={H - 16} textAnchor="middle" fontSize={9} fontFamily="monospace"
              fill={sel ? col : "rgba(255,255,255,0.3)"} style={{ transition: "fill 0.2s" }}>
              {li === 0 ? "Input" : li === layers.length - 1 ? "Output" : `Hidden ${li}`}
            </text>
            <text x={x} y={H - 4} textAnchor="middle" fontSize={8} fontFamily="monospace"
              fill="rgba(255,255,255,0.18)">n={layer.nodes}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ─── Layer config card ────────────────────────────────────────────────────────
function LayerCard({ layer, index, prevNodes, total, isSelected, isAtMinLayers, onSelect, onNodeDelta, onActivation, onRemove }) {
  const isInput  = index === 0;
  const isOutput = index === total - 1;
  const color    = isInput ? "#6366f1" : isOutput ? "#34d399" : ACT[layer.activation]?.color ?? "#94a3b8";

  return (
    <div onClick={onSelect} style={{
      background: isSelected ? "rgba(99,102,241,0.09)" : "rgba(255,255,255,0.02)",
      border: `1px solid ${isSelected ? "rgba(99,102,241,0.42)" : "rgba(255,255,255,0.06)"}`,
      borderRadius: 10, padding: "11px 13px", marginBottom: 7, cursor: "pointer", transition: "all 0.15s",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
        <div style={{ width: 7, height: 7, borderRadius: "50%", background: color, boxShadow: `0 0 7px ${color}88`, flexShrink: 0 }} />
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: "rgba(255,255,255,0.75)", marginBottom: 6 }}>
            {isInput ? "Input Layer" : isOutput ? "Output Layer" : `Hidden Layer ${index}`}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 5, flexWrap: "wrap" }}>
            <button onClick={e => { e.stopPropagation(); onNodeDelta(-1); }}
              disabled={layer.nodes <= MIN_NODES}
              style={{ ...btnSm, opacity: layer.nodes <= MIN_NODES ? 0.25 : 1, cursor: layer.nodes <= MIN_NODES ? "not-allowed" : "pointer" }}>
              &#8722;
            </button>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", minWidth: 22 }}>
              <span style={{ fontSize: 13, fontFamily: "monospace", fontWeight: 700, color, lineHeight: 1 }}>{layer.nodes}</span>
              {layer.nodes >= MAX_NODES && <span style={{ fontSize: 7, color: `${color}88`, fontFamily: "monospace", marginTop: 1 }}>max</span>}
              {layer.nodes <= MIN_NODES && <span style={{ fontSize: 7, color: "rgba(255,255,255,0.25)", fontFamily: "monospace", marginTop: 1 }}>min</span>}
            </div>
            <button onClick={e => { e.stopPropagation(); onNodeDelta(+1); }}
              disabled={layer.nodes >= MAX_NODES}
              style={{ ...btnSm, opacity: layer.nodes >= MAX_NODES ? 0.25 : 1, cursor: layer.nodes >= MAX_NODES ? "not-allowed" : "pointer" }}>
              +
            </button>
            <span style={{ fontSize: 10, color: "rgba(255,255,255,0.26)" }}>neurons</span>

            {!isInput && (isOutput
              ? <span style={{ fontSize: 10, fontFamily: "monospace", color: "#34d399", background: "#34d39920", border: "1px solid #34d39940", borderRadius: 5, padding: "2px 8px" }}>Softmax</span>
              : <select value={layer.activation}
                  onClick={e => e.stopPropagation()}
                  onChange={e => { e.stopPropagation(); onActivation(e.target.value); }}
                  style={{ background: "#0d1320", border: "1px solid rgba(255,255,255,0.12)", color, borderRadius: 6, padding: "2px 7px", fontSize: 10, fontFamily: "monospace", cursor: "pointer" }}>
                  {ACTIVATIONS.map(a => <option key={a} value={a}>{a}</option>)}
                </select>
            )}

            {!isInput && !isOutput && (
              <button
                onClick={e => { e.stopPropagation(); if (!isAtMinLayers) onRemove(); }}
                title={isAtMinLayers ? "Need at least one hidden layer" : "Remove this layer"}
                style={{ marginLeft: "auto", background: isAtMinLayers ? "rgba(239,68,68,0.05)" : "rgba(239,68,68,0.12)", border: `1px solid ${isAtMinLayers ? "rgba(239,68,68,0.1)" : "rgba(239,68,68,0.28)"}`, color: isAtMinLayers ? "rgba(252,165,165,0.3)" : "#fca5a5", padding: "2px 8px", borderRadius: 5, cursor: isAtMinLayers ? "not-allowed" : "pointer", fontSize: 10 }}>
                &#215;
              </button>
            )}
          </div>
        </div>
      </div>

      {isSelected && (
        <div style={{ marginTop: 8, paddingTop: 8, borderTop: "1px solid rgba(255,255,255,0.06)", fontSize: 11, color: "rgba(255,255,255,0.28)", fontFamily: "monospace", lineHeight: 1.8 }}>
          <T tex={`a^{(${index})} \\in \\mathbb{R}^{${layer.nodes}}`} />
          {index > 0 && <span style={{ marginLeft: 10 }}><T tex={`W^{(${index})} \\in \\mathbb{R}^{${layer.nodes}\\times${prevNodes}}`} /></span>}
        </div>
      )}
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const katexLoaded = useKatexLoader();
  const [layers, setLayers]           = useState(defaultLayers);
  const [sel, setSel]                 = useState(1);
  const [showExplainer, setExplainer] = useState(false);

  const n            = layers.length;
  const outputNodes  = layers[n - 1].nodes;
  const { totalW, totalB, total } = computeParams(layers);
  const isAtMaxLayers = n >= MAX_LAYERS;
  const isAtMinLayers = n <= 3;

  const reset = useCallback(() => { setLayers(defaultLayers()); setSel(1); }, []);

  const addLayer = useCallback(() => {
    if (isAtMaxLayers) return;
    setLayers(prev => {
      const next = [...prev];
      next.splice(prev.length - 1, 0, { nodes: 4, activation: "ReLU" });
      return next;
    });
  }, [isAtMaxLayers]);

  const removeLayer = useCallback((idx) => {
    setLayers(prev => prev.filter((_, i) => i !== idx));
    setSel(s => Math.max(1, Math.min(s, n - 3)));
  }, [n]);

  const updateNodes = useCallback((idx, delta) => {
    setLayers(prev => prev.map((l, i) =>
      i === idx ? { ...l, nodes: Math.max(MIN_NODES, Math.min(MAX_NODES, l.nodes + delta)) } : l
    ));
  }, []);

  const updateAct = useCallback((idx, act) => {
    setLayers(prev => prev.map((l, i) => i === idx ? { ...l, activation: act } : l));
  }, []);

  // Build full composition LaTeX (right-to-left)
  const compTex = (() => {
    let inner = "x";
    for (let i = n - 1; i >= 1; i--) {
      const ac = layers[i].activation;
      const fn = ac === "Softmax"
        ? "\\operatorname{softmax}"
        : `\\sigma_{\\scriptscriptstyle\\text{${ac}}}`;
      inner = `${fn}\\!\\bigl(W^{(${i})}${inner} + b^{(${i})}\\bigr)`;
    }
    return `f(x) = ${inner}`;
  })();

  return (
    <KatexCtx.Provider value={katexLoaded}>
      <div style={{ fontFamily: "'DM Sans', system-ui, sans-serif", background: "#07101c", color: "#e2e8f0", minHeight: "100vh", display: "flex", flexDirection: "column" }}>
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
          * { box-sizing: border-box; }
          ::-webkit-scrollbar { width: 4px; height: 4px; }
          ::-webkit-scrollbar-track { background: #0a1525; }
          ::-webkit-scrollbar-thumb { background: #2d3f5a; border-radius: 4px; }
          .katex { font-size: 1em !important; }
          .katex-display { margin: 0 !important; }
          .katex-display > .katex { text-align: left !important; }
          input[type=range] { height: 4px; }
        `}</style>

        {/* ── Top bar ── */}
        <div style={{ borderBottom: "1px solid rgba(255,255,255,0.06)", padding: "13px 22px", display: "flex", alignItems: "center", justifyContent: "space-between", background: "rgba(0,0,0,0.25)", backdropFilter: "blur(10px)", position: "sticky", top: 0, zIndex: 20 }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 15, fontWeight: 700, letterSpacing: "-0.025em", color: "white" }}>
              Neural Net <span style={{ color: "#6366f1" }}>&#215;</span> Matrix Visualizer
            </h1>
            <div style={{ fontSize: 10, color: "rgba(255,255,255,0.28)", marginTop: 2 }}>
              classification &nbsp;&middot;&nbsp; {n} layers &nbsp;&middot;&nbsp; {total.toLocaleString()} trainable parameters
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {/* Layer counter dots */}
            <div style={{ display: "flex", alignItems: "center", gap: 5, background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 7, padding: "5px 10px" }}>
              {[...Array(MAX_LAYERS)].map((_, i) => (
                <div key={i} style={{ width: 6, height: 6, borderRadius: "50%", background: i < n ? "#6366f1" : "rgba(255,255,255,0.1)", boxShadow: i < n ? "0 0 4px #6366f188" : "none", transition: "all 0.2s" }} />
              ))}
              <span style={{ fontSize: 9, color: "rgba(255,255,255,0.3)", fontFamily: "monospace", marginLeft: 2 }}>{n}/{MAX_LAYERS}</span>
            </div>

            {/* Reset */}
            <button onClick={reset} title="Reset to default architecture"
              style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.12)", color: "rgba(255,255,255,0.45)", padding: "7px 13px", borderRadius: 8, cursor: "pointer", fontSize: 12, fontWeight: 500 }}
              onMouseEnter={e => { e.currentTarget.style.background = "rgba(255,255,255,0.1)"; e.currentTarget.style.color = "rgba(255,255,255,0.75)"; }}
              onMouseLeave={e => { e.currentTarget.style.background = "rgba(255,255,255,0.05)"; e.currentTarget.style.color = "rgba(255,255,255,0.45)"; }}>
              &#8634; Reset
            </button>

            {/* Add layer */}
            <button onClick={addLayer} disabled={isAtMaxLayers}
              title={isAtMaxLayers ? `Maximum ${MAX_LAYERS} layers reached` : "Add a hidden layer"}
              style={{ background: isAtMaxLayers ? "rgba(99,102,241,0.07)" : "rgba(99,102,241,0.18)", border: `1px solid ${isAtMaxLayers ? "rgba(99,102,241,0.18)" : "rgba(99,102,241,0.38)"}`, color: isAtMaxLayers ? "rgba(165,180,252,0.35)" : "#a5b4fc", padding: "7px 15px", borderRadius: 8, cursor: isAtMaxLayers ? "not-allowed" : "pointer", fontSize: 12, fontWeight: 600 }}>
              {isAtMaxLayers ? "Max layers reached" : "+ Add Hidden Layer"}
            </button>
          </div>
        </div>

        {/* ── Body ── */}
        <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", flex: 1, overflow: "hidden", height: "calc(100vh - 57px)" }}>

          {/* Left: config */}
          <div style={{ background: "#0a1525", borderRight: "1px solid rgba(255,255,255,0.06)", overflowY: "auto", padding: 14 }}>
            <div style={secLabel}>Architecture</div>

            {layers.map((layer, li) => (
              <LayerCard
                key={li}
                layer={layer}
                index={li}
                prevNodes={li > 0 ? layers[li - 1].nodes : 0}
                total={n}
                isSelected={sel === li}
                isAtMinLayers={isAtMinLayers}
                onSelect={() => setSel(li)}
                onNodeDelta={d => updateNodes(li, d)}
                onActivation={act => updateAct(li, act)}
                onRemove={() => removeLayer(li)}
              />
            ))}

            {/* Summary */}
            <div style={{ marginTop: 14, background: "rgba(0,0,0,0.28)", borderRadius: 10, padding: "13px 14px", border: "1px solid rgba(255,255,255,0.05)" }}>
              <div style={secLabel}>Summary</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, marginBottom: 12 }}>
                {[
                  { label: "Layers",  val: n,           color: "#6366f1" },
                  { label: "Weights", val: totalW,      color: "#818cf8" },
                  { label: "Biases",  val: totalB,      color: "#a78bfa" },
                  { label: "Classes", val: outputNodes, color: "#34d399" },
                ].map(({ label, val, color }) => (
                  <div key={label} style={{ textAlign: "center", background: "rgba(255,255,255,0.03)", borderRadius: 8, padding: "8px 4px" }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color, fontFamily: "monospace" }}>{val}</div>
                    <div style={{ fontSize: 9, color: "rgba(255,255,255,0.28)", marginTop: 2 }}>{label}</div>
                  </div>
                ))}
              </div>
              <button onClick={() => setExplainer(true)}
                style={{ width: "100%", background: "rgba(99,102,241,0.1)", border: "1px solid rgba(99,102,241,0.25)", color: "#818cf8", borderRadius: 7, padding: "7px 0", cursor: "pointer", fontSize: 11, fontFamily: "monospace" }}>
                how are these computed? &#8594;
              </button>
              <div style={{ marginTop: 14, paddingTop: 14, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                <div style={secLabel}>Function composition</div>
                <div style={{ background: "rgba(0,0,0,0.3)", borderRadius: 7, padding: "12px 14px", overflowX: "auto" }}>
                  <T d tex={compTex} />
                </div>
              </div>
            </div>
          </div>

          {/* Right: graph + math */}
          <div style={{ display: "grid", gridTemplateRows: "auto 1fr", overflow: "hidden" }}>

            {/* SVG graph */}
            <div style={{ padding: "12px 20px", borderBottom: "1px solid rgba(255,255,255,0.06)", background: "rgba(0,0,0,0.13)" }}>
              <div style={secLabel}>Network Graph &middot; click a layer to inspect</div>
              <NetworkSVG layers={layers} selectedLayer={sel} onSelect={setSel} />
            </div>

            {/* Transform cards + loss */}
            <div style={{ overflowY: "auto", padding: "14px 20px" }}>
              <div style={secLabel}>Layer Transformations &middot; expand to show reasoning and weight matrix</div>

              {layers.slice(1).map((layer, li) => (
                <TransformCard
                  key={`tc-${li}`}
                  fromLayer={layers[li]}
                  toLayer={layer}
                  layerIdx={li + 1}
                  isLast={li === n - 2}
                  startOpen={li + 1 === sel}
                />
              ))}

              <LossPanel K={outputNodes} />
            </div>
          </div>
        </div>

        {showExplainer && <ParamExplainer layers={layers} onClose={() => setExplainer(false)} />}
      </div>
    </KatexCtx.Provider>
  );
}
