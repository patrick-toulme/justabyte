"""
Parametric compute-demand model for "We've Built 1% of the AI Compute the World
Needs."

Question: how many AI chips does the world need vs. how many it has, when a
population of people each talk to one orchestrator AI that runs a swarm of
subagents on an open coding model (GLM-5.2-class) -- swept across model size
from 10B to 100T total params and swarm sizes from 1 to 500.

Everything is normalized to H100-equivalents. All weights/serving assume FP8.
Frontier models are assumed MoE: FLOPs (serving + training) scale with ACTIVE
params; memory (chips-per-instance) scales with TOTAL params.

Every constant is sourced inline in the article. Change any number you doubt
and re-run; the gap survives it.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

SECONDS_PER_YEAR = 365.25 * 24 * 3600

# ---------------------------------------------------------------------------
# Reference hardware: H100 (the H100-equivalent unit).
# ---------------------------------------------------------------------------
HBM_BYTES = 80e9            # 80 GB
HBM_BW = 3.35e12           # 3.35 TB/s
FLOPS_FP8_DENSE = 1979e12  # ~1979 TFLOP/s FP8 dense
MFU = 0.35                 # FP8 training MFU @ trillion-param scale (SemiAnalysis)
HBM_UTIL = 0.75            # frac of HBM usable for weights (rest: KV/activ.)
BYTES_PER_PARAM = 1.0      # FP8 serving today (the frontier default in 2026)
# Precision: most frontier serving is FP8 now; Blackwell NVFP4/MXFP4 (~lossless, <1%
# accuracy loss; gpt-oss ships MXFP4, DeepSeek FP8->NVFP4) is rolling out and ~halves
# bytes -> ~2x throughput & capacity. One-time step (can't go below ~4 bits).
SERVE_BYTES_FP4 = 0.5

# ---------------------------------------------------------------------------
# Serving economics (SemiAnalysis InferenceMAX / LMSYS).
# Frontier MoE active/total SHRINKS with size and active params grow SUB-linearly.
# Anchors (total -> active params), log-log interpolated by active_params():
#   DeepSeek-V3 37B@671B (5.5%), Kimi-K2 32B@1T (3.2%), Llama-4 Maverick 17B@400B.
#   Kimi-K3 104B@2.8T lands slightly ABOVE this curve (~82B implied), so the
#   schedule is supply-favorable.
# ---------------------------------------------------------------------------
N_ANCHORS      = [1e10, 1e11, 1e12,  1e13,  1e14]   # 10B,100B,1T,10T,100T
ACTIVE_ANCHORS = [3e9,  12e9, 40e9,  200e9, 1e12]   # active params at each size
# Throughput from SemiAnalysis InferenceMAX (DeepSeek-R1, a REASONING MoE, 37B active)
# at ~42 tok/s/user -- the realistic point for usable reasoning, NOT the max-batch
# ceiling. Measured: H100 266 tok/s/GPU, B200 4,792 (=2,083/H100e). Blended
# Hopper+Blackwell fleet ~1,300 tok/s/H100e. Scales as 1/active_params.
TPS_REF = 1300.0           # aggregate tok/s per H100-equiv at ACTIVE_REF active params
ACTIVE_REF = 37e9

# ---------------------------------------------------------------------------
# Supply (SemiAnalysis-derived via IFP / Epoch). FP8 H100e.
# ---------------------------------------------------------------------------
# World base end-2025 ~12-15M H100e (NVIDIA-only ~10.8M FP8 + hyperscaler ASICs).
# Binding constraint is CoWoS packaging + HBM, not logic dies.
INSTALLED_BASE_H100E = 19.0e6  # world installed base mid-2026, FP8 H100e (Epoch ~15M Jan-2026, rising)
ANNUAL_PROD_H100E = 16.0e6     # ~H100e added/yr (Epoch: capacity ~doubling every 7 months)
# Supply split, FP8 H100e. Epoch: Google is the LARGEST single owner (~25% of global,
# mostly TPU) + Ironwood ramp (~3.2M v7 in 2026, 2.3 H100e/chip) -> 4.8M. NVIDIA ~58%.
# Trainium = 1.4M physical chips all-gens (mostly T2 ~0.65 H100e, some T3 ~1.27) ~1.2M
# H100e. AMD (MI350/MI450, ~$10B/qtr) 1.0M; Others = Maia, MTIA, Huawei Ascend.
SUPPLY_BY_VENDOR = [
    ("NVIDIA",        11.0e6),
    ("Google TPU",     4.8e6),
    ("AWS Trainium",   1.2e6),
    ("AMD",            1.0e6),
    ("Others",         1.0e6),
]
# Not all installed silicon can serve. INSTALLED_BASE is already the MODERN fleet
# (Hopper+; excludes Volta/Pascal/old A100 -- nobody serves frontier models on those).
# Of the modern fleet, only a share is free to serve public inference at any moment
# (rest = training runs, redundancy, cloud/region fragmentation, idle-between-bursts).
USABLE_FRACTION = 0.60
# Power: all-in grid draw per H100-equivalent (GPU + network + cooling + PUE).
# GB200 NVL72 = 120 kW / 72 GPUs ~ 1.2 kW/GPU at the rack; per-H100e is lower as
# perf/watt improves. Use 1.0 kW/H100e (conservative). Context anchors:
POWER_PER_H100E_KW = 1.0
GLOBAL_DATACENTER_GW = 55      # ~485 TWh/yr in 2025 (IEA), ALL datacenters worldwide
US_ELECTRICITY_AVG_GW = 470    # ~4,100 TWh/yr average US electricity consumption

# ---------------------------------------------------------------------------
# Demand population & intensity
# ---------------------------------------------------------------------------
# Output-equivalent sustained generation, 24/7 duty cycle.
WORLD_POP = 8.2e9                                                              # UN/DataReportal 2026
US_POPULATION = 335e6          # 2026; the "one rich country gets there first" ceiling
US_ADULTS = 261e6              # ~78% of US population are adults (Census)
SOFTWARE_ENGINEERS = 28e6      # range 20M (JetBrains) - 37M (SlashData)
KNOWLEDGE_WORKERS = 1.0e9      # Gartner ~1B knowledge workers (analysts/eng/lawyers;
                               # NOT all ~3.4B employed, NOT all ~1.75B white-collar)
# Headline population: HALF of knowledge workers (~500M). Deliberately a PROPORTION,
# used as a legible frame of reference -- not everyone will adopt or afford agentic AI.
# It is a FLOOR, not a ceiling: it counts NONE of the other AI demand (personal finance,
# health, education, science, consumers), each of which is its own swarm story.
KW_ADOPTERS = KNOWLEDGE_WORKERS / 2   # ~500M -- the realistic headline slice
AGENTS_PER_SWE_CODING = 500    # coding swarm per software engineer
AGENTS_PER_EMPLOYEE = 1000     # enterprise: agents per knowledge worker
# Reasoning is the default now; agents GENERATE far more than they visibly output
# (~8x hidden chain-of-thought, Epoch). These are sustained GENERATED tok/s per agent.
TPS_PER_AGENT_FLOOR = 15.0     # background reasoning agent
TPS_PER_AGENT_MID = 40.0       # active reasoning agent (~InferenceMAX 42 tok/s/user)
TPS_PER_AGENT_AGGR = 100.0     # responsive, heavy-reasoning loop

# Realistic-adoption populations: a human talks to ONE orchestrator AI, which spawns
# a SWARM of subagents. The driver is swarm size, not population. These are the
# grounded scenarios (a fraction of one country, a profession, the knowledge economy)
# that the article centers on; "every human" is kept only as the upper-bound ceiling.
POPULATIONS = [
    ("swe",  "All software engineers",  SOFTWARE_ENGINEERS),  # 28M
    ("us20", "1/5 of adult Americans",  US_ADULTS / 5),       # ~52M
    ("us25", "1/4 of adult Americans",  US_ADULTS / 4),       # ~65M
    ("kw",   "Half of knowledge workers", KW_ADOPTERS),       # ~500M (headline slice)
]
SWARM_GRID = np.logspace(0, np.log10(500), 160)   # 1 -> 500 subagents per person
# GLM-5.2: open-weights coding model (Zhipu/Z.ai, MIT, released Jun 2026). 744B total,
# ~40B active; 1M context; ~744 GB FP8 weights -> a box of ~8 datacenter GPUs to serve.
# Frontier-class agentic coding (SWE-bench Pro 62, MCP-Atlas 77) at API prices ~1/6 the
# closed leaders ($2/$6 per Mtok), self-hostable -- this is what makes large swarms
# realistic NOW. Serving cost is set by its ACTIVE params (40B), so it serves like the
# ~1T-total bucket. Used as the default "open coder" the realistic scenarios run on.
GLM_TOTAL  = 744e9
GLM_ACTIVE = 40e9
OPEN_CODER_N = 1e12            # model-size bucket equivalent to GLM-5.2's 40B active

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# Token budget FLATTENS at large N (data wall:
# total quality text ~300T tokens). Anchors (for N_ANCHORS) -> training tokens.
TOK_ANCHORS = [15e12, 25e12, 40e12, 60e12, 80e12]  # 10T/100T are data-bottlenecked
N_FRONTIER_MODELS = 20         # distinct frontier model lines trained worldwide
REFRESH_MONTHS = 6             # retrain cadence (compressing toward 2-3mo)
REFRESH_SECONDS = REFRESH_MONTHS / 12 * SECONDS_PER_YEAR
# RL / post-training compute. RL is INFERENCE-shaped (rollout generation = ~70-90%
# of RL wall-clock, memory-bound decode) and scaling ~10x every 3-5 months. Arc of
# K = RL/pretraining: ~1% pre-2024 -> <10% (o1) -> ~20% (DeepSeek-R1) -> ~1x (Grok 4,
# mid-2025). Use K=1 (frontier parity 2025-26); range 0.1-1x. Sources inline in article.
RL_MULT = 1.0

# Model-size sweep (total params).
N_GRID = np.logspace(10, 14, 200)          # 10B -> 100T
N_MARKERS = [1e10, 1e11, 1e12, 1e13, 1e14]  # 10B,100B,1T,10T,100T


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------
def _loglog_interp(x, xp, fp):
    """Interpolate fp(xp) at x in log-log space, clamped to the anchor range."""
    lx = np.log10(np.asarray(x, dtype=float))
    return 10.0 ** np.interp(lx, np.log10(xp), np.log10(fp))


def active_params(n_total):
    """Active (MoE) params vs total -- sublinear; see ACTIVE_ANCHORS."""
    return _loglog_interp(n_total, N_ANCHORS, ACTIVE_ANCHORS)


def tokens_budget(n_total):
    """Training tokens vs model size -- flattens at large N (data wall)."""
    return _loglog_interp(n_total, N_ANCHORS, TOK_ANCHORS)


def chips_per_instance(n_total):
    """Chips needed just to hold one model instance's weights in HBM."""
    weight_bytes = np.asarray(n_total, dtype=float) * BYTES_PER_PARAM
    return np.ceil(weight_bytes / (HBM_BYTES * HBM_UTIL))


def tps_per_chip(n_total, bytes_per_param=BYTES_PER_PARAM):
    """Aggregate decode tok/s per H100-equiv (bandwidth-bound, high-batch regime),
    calibrated to DeepSeek-R1 FP8; scales inversely with active params and with
    precision bytes (FP4 = 0.5 -> ~2x)."""
    return TPS_REF * ACTIVE_REF / active_params(n_total) * (1.0 / bytes_per_param)


def _agent_count(scenario):
    return {
        "usa":         US_POPULATION * 1,                        # every American, 1 agent
        "human":       WORLD_POP * 1,                            # every human, 1 agent
        "eng100":      SOFTWARE_ENGINEERS * 100,                 # every SWE, 100 agents
        "eng500":      SOFTWARE_ENGINEERS * AGENTS_PER_SWE_CODING,   # SWE coding swarm
        "company1000": KNOWLEDGE_WORKERS * AGENTS_PER_EMPLOYEE,      # 1000 agents/employee
    }[scenario]


def demand_tps(scenario: str):
    agents = _agent_count(scenario)
    return {
        "floor": agents * TPS_PER_AGENT_FLOOR,
        "mid": agents * TPS_PER_AGENT_MID,
        "aggr": agents * TPS_PER_AGENT_AGGR,
    }


def chips_serving(n_total, total_tps):
    """Chips to sustain `total_tps` tokens/sec of an N-param model."""
    return total_tps / tps_per_chip(n_total)


def chips_for_swarm(pop, swarm, n_total=OPEN_CODER_N, tps_per_agent=TPS_PER_AGENT_MID):
    """Realistic frame: `pop` people each talk to one orchestrator AI that runs a
    `swarm` of subagents, on an open coding model (default GLM-5.2-class, 40B active).
    Chips = pop x swarm x tok/s-per-agent / per-chip throughput."""
    return chips_serving(n_total, pop * swarm * tps_per_agent)


def tps_per_chip_active(active, bytes_per_param=BYTES_PER_PARAM):
    """Per-H100e decode throughput for a model with a GIVEN active-param count
    (bypasses the size->active curve, so a named model's real active count is used)."""
    return TPS_REF * ACTIVE_REF / active * (1.0 / bytes_per_param)


def chips_on_glm(pop, swarm, tps_per_agent=TPS_PER_AGENT_MID):
    """Run a scenario on GLM-5.2 specifically (its real 40B active params)."""
    return pop * swarm * tps_per_agent / tps_per_chip_active(GLM_ACTIVE)


# Each demand scenario as a named option to run on GLM-5.2 (label, population, swarm).
GLM_SCENARIOS = [
    ("All software engineers · 50 swarm",     SOFTWARE_ENGINEERS, 50),
    ("1/5 adult Americans · 50 swarm",        US_ADULTS / 5,      50),
    ("1/4 adult Americans · 50 swarm",        US_ADULTS / 4,      50),
    ("Half of knowledge workers · 50 swarm",  KW_ADOPTERS,        50),
    ("Half of knowledge workers · 500 swarm", KW_ADOPTERS,       500),
]


# Premium tier: some users pay ~3x to dedicate ~3x chips to their request (low batch,
# low latency) -- e.g. fast agentic coding. Empirically the premium is STEEPER than 3x
# (InferenceMAX: interactive serving ~30x the chips of max batch), so 3x is conservative.
FAST_CHIP_MULT = 3.0

def serving_with_fast_tier(n_total, total_tps, fast_fraction):
    """Chips to serve when `fast_fraction` of demand runs in the premium (3x) tier."""
    mult = (1.0 - fast_fraction) + fast_fraction * FAST_CHIP_MULT
    return chips_serving(n_total, total_tps) * mult


def chips_for_world(served_size, agents_per_person,
                    tps_per_agent=TPS_PER_AGENT_MID, pop=WORLD_POP):
    """Jevons: if a SMALL `served_size` model matches frontier quality, people run
    MANY agents each. Total chips = pop x agents x tok/s / per-chip throughput."""
    demand = pop * agents_per_person * tps_per_agent
    return chips_serving(served_size, demand)


def power_gw(chips):
    """Continuous grid power (GW) for a fleet of `chips` H100-equivalents, all-in."""
    return chips * POWER_PER_H100E_KW / 1e6


def train_flops_one(n_total):
    return 6.0 * active_params(n_total) * tokens_budget(n_total)


def h100_years_one(n_total):
    return train_flops_one(n_total) / (FLOPS_FP8_DENSE * MFU * SECONDS_PER_YEAR)


def chips_training_fleet(n_total):
    """Sustained chips to retrain N_FRONTIER_MODELS every REFRESH period (pretraining)."""
    flops_per_sec_needed = N_FRONTIER_MODELS * train_flops_one(n_total) / REFRESH_SECONDS
    return flops_per_sec_needed / (FLOPS_FP8_DENSE * MFU)


def chips_rl_fleet(n_total):
    """Sustained chips for RL/post-training, ~ RL_MULT x pretraining. Conservative:
    RL is inference-shaped (memory-bound decode), so true chip cost is likely higher."""
    return RL_MULT * chips_training_fleet(n_total)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (9, 5.5),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def _fmt_params(x, _pos):
    units = [(1e12, "T"), (1e9, "B"), (1e6, "M")]
    for scale, suf in units:
        if x >= scale:
            return f"{x/scale:g}{suf}"
    return f"{x:g}"


def _style_axes(ax, title, ylabel):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Model size (total parameters)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_params))
    ax.set_xticks(N_MARKERS)
    ax.legend(frameon=False, fontsize=9, loc="upper left")


def _supply_lines(ax, years=(0, 3, 5)):
    for y in years:
        level = INSTALLED_BASE_H100E + ANNUAL_PROD_H100E * y
        label = "Installed base today" if y == 0 else f"+ {y} yr production"
        ax.axhline(level, ls="--", lw=1, alpha=0.7,
                   color="#888" if y else "#c0392b",
                   label=label)


def plot_serving(path):
    fig, ax = plt.subplots()
    for scen, color in [("human", "#2980b9"), ("eng100", "#27ae60")]:
        d = demand_tps(scen)
        name = "Every human · 1 agent" if scen == "human" else "Every engineer · 100 agents"
        ax.plot(N_GRID, chips_serving(N_GRID, d["mid"]), color=color, lw=2, label=name)
        ax.fill_between(N_GRID, chips_serving(N_GRID, d["floor"]),
                        chips_serving(N_GRID, d["aggr"]), color=color, alpha=0.12)
    _supply_lines(ax)
    _style_axes(ax, "Chips to SERVE the world an always-on N-param model",
                "H100-equivalent chips needed (log)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_training(path):
    fig, ax = plt.subplots()
    ax.plot(N_GRID, chips_training_fleet(N_GRID), color="#8e44ad", lw=2,
            label=f"Train {N_FRONTIER_MODELS} frontier models / {REFRESH_MONTHS}mo")
    _supply_lines(ax)
    _style_axes(ax, "Chips to TRAIN & refresh the frontier fleet",
                "H100-equivalent chips needed (log)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_total_gap(path):
    fig, ax = plt.subplots()
    d_h = demand_tps("human"); d_e = demand_tps("eng100")
    total_h = chips_serving(N_GRID, d_h["mid"]) + chips_training_fleet(N_GRID)
    total_e = chips_serving(N_GRID, d_e["mid"]) + chips_training_fleet(N_GRID)
    ax.plot(N_GRID, total_h, color="#2980b9", lw=2, label="Humans · 1 agent + training")
    ax.plot(N_GRID, total_e, color="#27ae60", lw=2, label="Engineers · 100 agents + training")
    ax.fill_between(N_GRID, INSTALLED_BASE_H100E, np.maximum(total_h, total_e),
                    where=np.maximum(total_h, total_e) > INSTALLED_BASE_H100E,
                    color="#e74c3c", alpha=0.08)
    _supply_lines(ax)
    ax.axhline(INSTALLED_BASE_H100E * USABLE_FRACTION, ls=":", color="#c0392b", lw=1.3,
               label=f"Usable to serve now (~{USABLE_FRACTION:.0%} of fleet)")
    _style_axes(ax, "The gap: chips needed vs. chips that exist",
                "H100-equivalent chips (log)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_speed_tiers(path):
    fig, ax = plt.subplots()
    d = demand_tps("human")["mid"]
    tiers = [(0.0, "All background batch (15 tok/s)", "#2980b9"),
             (0.2, "20% on premium fast tier", "#e67e22"),
             (1.0, "All premium · 3x chips (fast coding)", "#c0392b")]
    for f, label, color in tiers:
        ax.plot(N_GRID, serving_with_fast_tier(N_GRID, d, f), color=color, lw=2, label=label)
    _supply_lines(ax, years=(0,))
    _style_axes(ax, "The speed premium: 3x chips to compute your request faster",
                "H100-equivalent chips needed (log)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_vendor_supply(path):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    cats = ["All AI silicon\n(every vendor)", "100B / human", "1T / human", "10T / human"]
    colors = ["#76b900", "#4285F4", "#ff9900", "#ED1C24", "#888888"]
    bottom = 0.0
    for (name, val), c in zip(SUPPLY_BY_VENDOR, colors):
        ax.bar(0, val, bottom=bottom, color=c, width=0.6, label=name)
        bottom += val
    total = bottom
    dh = demand_tps("human")["mid"]
    demand_pts = [(1, 1e11), (2, 1e12), (3, 1e13)]
    for i, n in demand_pts:
        v = chips_serving(n, dh)
        ax.bar(i, v, color="#34495e", width=0.6)
        ax.text(i, v * 1.18, _h(v), ha="center", fontsize=8.5, fontweight="bold")
    ax.text(0, total * 1.18, f"~{_h(total)} total", ha="center", fontsize=8.5, fontweight="bold")
    ax.axhline(total, ls="--", color="#888", lw=1)
    ax.set_yscale("log"); ax.set_ylim(1e5, 1e9)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("H100-equivalent chips (log)")
    ax.set_title("They aren't competing: every vendor combined vs. one model per human",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right", title="Supply by vendor")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_agent_explosion(path):
    fig, ax = plt.subplots()
    agents = np.logspace(0, 3.3, 120)  # 1 -> ~2000 agents per person
    for size, label, color in [(1e10, "10B model = 10T quality", "#27ae60"),
                               (1e11, "100B model = 10T quality", "#2980b9"),
                               (1e12, "1T model = 10T quality",  "#8e44ad")]:
        ax.plot(agents, chips_for_world(size, agents), color=color, lw=2, label=label)
    ceiling = chips_serving(1e13, demand_tps("human")["mid"])
    ax.axhline(ceiling, ls="--", color="#e67e22", lw=1.3,
               label=f"'Old ceiling': 10T model, 1 agent each ({_h(ceiling)})")
    ax.axhline(INSTALLED_BASE_H100E, ls="--", color="#c0392b", lw=1.3,
               label=f"World fleet today ({_h(INSTALLED_BASE_H100E)})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Agents per person (concurrent, always-on)")
    ax.set_ylabel("H100-equivalent chips needed (log)")
    ax.set_title("Jevons: a cheaper model just means everyone runs a swarm",
                 fontweight="bold")
    ax.set_xticks([1, 10, 100, 1000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_scenarios(path):
    fig, ax = plt.subplots()
    scen = [("usa",         "Every American · 1 agent",             "#16a085"),
            ("human",       "Every human · 1 agent",                "#2980b9"),
            ("eng500",      "Every SWE · 500 coding agents",        "#e67e22"),
            ("company1000", "Every knowledge worker · 1000 agents", "#c0392b")]
    for key, label, color in scen:
        ax.plot(N_GRID, chips_serving(N_GRID, demand_tps(key)["mid"]),
                color=color, lw=2, label=label)
    _supply_lines(ax, years=(0,))
    ax.axhline(INSTALLED_BASE_H100E * USABLE_FRACTION, ls=":", color="#888", lw=1.2,
               label=f"Usable now (~{USABLE_FRACTION:.0%})")
    _style_axes(ax, "From conservative to aggressive: who runs how many agents",
                "H100-equivalent chips needed (log)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _h(x):
    """Human-readable chip count."""
    if x >= 1e9: return f"{x/1e9:.1f}B"
    if x >= 1e6: return f"{x/1e6:.1f}M"
    if x >= 1e3: return f"{x/1e3:.0f}k"
    return f"{x:.0f}"


def print_table():
    dh = demand_tps("human")["mid"]
    de = demand_tps("eng100")["mid"]
    print(f"\nWorld installed base: {_h(INSTALLED_BASE_H100E)} H100e (FP8) | "
          f"+{_h(ANNUAL_PROD_H100E)}/yr build rate\n")
    hdr = (f"{'N total':>8} {'active':>8} {'chips/inst':>10} {'tok/s/chip':>10} "
           f"{'serve-all':>10} {'x fleet':>8} {'train-fleet':>11}")
    print(hdr); print("-" * len(hdr))
    for n in N_MARKERS:
        serve = chips_serving(n, dh)
        train = chips_training_fleet(n)
        print(f"{_fmt_params(n,0):>8} {_fmt_params(active_params(n),0):>8} "
              f"{chips_per_instance(n):>10,.0f} {tps_per_chip(n):>10,.0f} "
              f"{_h(serve):>10} {serve/INSTALLED_BASE_H100E:>7.1f}x {_h(train):>11}")
    print(f"\n'serve-all' = chips to give every human (8.2B) one always-on "
          f"40 tok/s reasoning agent of that size.")
    print(f"Eng-x100 scenario demand = {de:.2e} tok/s "
          f"({de/dh:.0%} of the human scenario).")
    fast10 = serving_with_fast_tier(1e13, dh, 1.0)
    print(f"Speed premium: all-fast 10T (3x chips) = {_h(fast10)} "
          f"({fast10/INSTALLED_BASE_H100E:.0f}x fleet).")
    tot = sum(v for _, v in SUPPLY_BY_VENDOR)
    print(f"All vendors combined = {_h(tot)} H100e (NVIDIA "
          f"{SUPPLY_BY_VENDOR[0][1]/tot:.0%}); 1T/human needs "
          f"{chips_serving(1e12, dh)/tot:.1f}x that.")
    head = chips_for_swarm(KW_ADOPTERS, 50)
    print(f"\nJevons (small model = frontier quality), pop = half of knowledge workers,")
    print(f"  vs realistic headline {_h(head)} (50-agent swarm on the open model):")
    for size, nm in [(1e11, "100B"), (1e10, " 10B")]:
        for a in (100, 1000):
            c = chips_for_world(size, a, pop=KW_ADOPTERS)
            print(f"  {nm}=frontier x {a:>4} agents/person: {_h(c):>6} "
                  f"({c/INSTALLED_BASE_H100E:>5.0f}x fleet, {c/head:>4.1f}x headline)")
    print(f"\nCorrections (opposite signs): FP4 ~halves demand; usable "
          f"~{USABLE_FRACTION:.0%} of modern fleet ~halves supply.")
    g_full = chips_serving(1e13, dh) / INSTALLED_BASE_H100E
    g_corr = (chips_serving(1e13, dh) * SERVE_BYTES_FP4) / (INSTALLED_BASE_H100E * USABLE_FRACTION)
    print(f"  10T-for-everyone gap: FP8 & full fleet {g_full:.0f}x  ->  "
          f"FP4 & usable-only {g_corr:.0f}x  (barely moves).")
    p10 = power_gw(chips_serving(1e13, dh))
    print(f"\nPower (all-in {POWER_PER_H100E_KW:.1f} kW/H100e): 10T-for-everyone = {p10:.0f} GW "
          f"(~{p10/GLOBAL_DATACENTER_GW:.0f}x all datacenters today, "
          f"~{p10/US_ELECTRICITY_AVG_GW:.0%} of US electricity).")
    print(f"  100T = {power_gw(chips_serving(1e14, dh)):,.0f} GW (~US grid); "
          f"Jevons 100Bx1000 agents = {power_gw(chips_for_world(1e11, 1000)):,.0f} GW.")
    print("\nREALISTIC scenarios (GLM-5.2-class open coder, 40B active, mid 40 tok/s):")
    print(f"  {'population':<26}{'x10':>10}{'x50':>10}{'x100':>10}")
    for key, nm, pop in POPULATIONS:
        cells = "".join(f"{chips_for_swarm(pop, s)/INSTALLED_BASE_H100E:>8.1f}x "
                        for s in (10, 50, 100))
        print(f"  {nm:<26}{cells}")
    hw = chips_for_swarm(KW_ADOPTERS, 50)
    print(f"  -> headline: HALF of knowledge workers x50 swarm = {_h(hw)} "
          f"({hw/INSTALLED_BASE_H100E:.0f}x fleet, {power_gw(hw):,.0f} GW)")

    print(f"\nEvery scenario RUN ON GLM-5.2 ({GLM_TOTAL/1e9:.0f}B total / "
          f"{GLM_ACTIVE/1e9:.0f}B active, {tps_per_chip_active(GLM_ACTIVE):,.0f} tok/s/H100e):")
    print(f"  {'scenario':<36}{'chips':>9}{'x fleet':>10}{'GW':>10}")
    for label, pop, swarm in GLM_SCENARIOS:
        c = chips_on_glm(pop, swarm)
        print(f"  {label:<36}{_h(c):>9}{c/INSTALLED_BASE_H100E:>8.0f}x {power_gw(c):>8,.0f}")

    print("\nCEILING scenarios (10T model, mid 40 tok/s) -- chips & power:")
    for key, nm in [("usa", "Every American x1"), ("human", "Every human x1"),
                    ("eng500", "Every SWE x500"),
                    ("company1000", "Every knowledge worker x1000")]:
        c = chips_serving(1e13, demand_tps(key)["mid"])
        print(f"  {nm:<30}: {_h(c):>6} ({c/INSTALLED_BASE_H100E:>7.1f}x fleet, "
              f"{power_gw(c):>8,.0f} GW)")
    pre = chips_training_fleet(1e13); rl = chips_rl_fleet(1e13); serve = chips_serving(1e13, dh)
    print(f"\nTraining-side @10T ({N_FRONTIER_MODELS} model lines): pretraining {_h(pre)}, "
          f"RL (K={RL_MULT:g}x) {_h(rl)}, total {_h(pre+rl)}")
    print(f"  = {(pre+rl)/INSTALLED_BASE_H100E:.0%} of today's fleet, "
          f"but only {(pre+rl)/serve*100:.1f}% of serving-everyone.")


if __name__ == "__main__":
    import os
    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)
    print_table()
    plot_serving(os.path.join(outdir, "01_serving.png"))
    plot_training(os.path.join(outdir, "02_training.png"))
    plot_total_gap(os.path.join(outdir, "03_gap.png"))
    plot_speed_tiers(os.path.join(outdir, "04_speed_tiers.png"))
    plot_vendor_supply(os.path.join(outdir, "05_vendors.png"))
    plot_agent_explosion(os.path.join(outdir, "06_jevons.png"))
    plot_scenarios(os.path.join(outdir, "07_scenarios.png"))
    print(f"\nWrote figures to {outdir}")
