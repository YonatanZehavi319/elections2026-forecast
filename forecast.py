"""Seat forecast for the election to the 26th Knesset (27 October 2026).

Method:
1. Polls: official filings to the Central Elections Committee, fieldwork ending 7-23 October,
   converted to vote shares. Channel 13 publishes seats only; its seats are converted to shares.
2. Averaging: within each polling firm, polls are weighted by recency (half weight after 5 days)
   and sample size (square root, capped at 2,000); the firms are then combined with equal weight.
3. Uncertainty: 20,000 simulated elections. In each, the firms' weights vary at random and every
   list gets a random error (30% larger for lists running for the first time).
4. Seats: 3.25% threshold, then Bader-Ofer (D'Hondt) with the surplus-vote agreements.
5. Submission: each list's median seats, adjusted to exactly 120 seats with the smallest
   increase in expected error (the competition score is the sum of absolute seat errors).

Run from this folder:  python forecast.py
Requires: numpy, pandas. Output: forecast.csv
"""
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
REFERENCE_DATE = pd.Timestamp("2026-10-23")   # last day of fieldwork before the poll ban
WINDOW_START = pd.Timestamp("2026-10-07")
HALF_LIFE_DAYS = 5          # a poll this many days older counts half as much
SAMPLE_CAP = 2000           # larger samples get no extra weight
DIRICHLET_ALPHA = 1.0       # spread of the random firm weights in the simulation
OTHER_SHARE = 2.0           # % of valid votes for small lists not tracked
THRESHOLD = 3.25
N_SIM = 20000
SEED = 26
PARTY_SD_COEF = 0.30        # random error of a list = coef * sqrt(its share in %)
NEW_LIST_MULT = 1.3         # extra error for lists running for the first time

TRACKED = ["מחל", "דרך", "רק", "אמת", "ודם", "ב", "ט", "שס", "ג", "ל", "עם", "די", "ך", "זך"]
NEW_LISTS = {"דרך", "רק", "די", "ך", "זך"}


# ---------------------------------------------------------------- 1. polls -> vote shares
def load_polls():
    """One row per national poll in the window: firm, end date, sample size, seats_<list>, pct_<list>."""
    meta = pd.read_csv("data/polls_meta.csv", parse_dates=["field_end"])
    res = pd.read_csv("data/polls_results.csv")
    seats = res.pivot_table(index="file", columns="letters", values="seats", aggfunc="first")
    pct = res.pivot_table(index="file", columns="letters", values="pct", aggfunc="first")
    meta = meta[(meta["national"] == "yes") & meta["file"].isin(seats.index)
                & (meta["field_end"] >= WINDOW_START)]
    rows = []
    for _, m in meta.iterrows():
        row = {"file": m["file"], "pollster": m["pollster"], "end": m["field_end"], "n": m["respondents"]}
        for p in TRACKED:
            row[f"seats_{p}"] = seats.loc[m["file"]].get(p)
            row[f"pct_{p}"] = pct.loc[m["file"]].get(p) if m["file"] in pct.index else np.nan
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def poll_shares(polls):
    """Vote share (%) of each tracked list in each poll."""
    seat_cols = [f"seats_{p}" for p in TRACKED]
    pct_cols = [f"pct_{p}" for p in TRACKED]
    has_pct = polls.apply(
        lambda r: all(pd.notna(r[f"pct_{p}"]) for p in TRACKED if r[f"seats_{p}"] > 0), axis=1)

    shares = pd.DataFrame(np.nan, index=polls.index, columns=TRACKED)
    # Polls with percentages: rescale the tracked lists to (100 - OTHER_SHARE).
    for i in polls.index[has_pct]:
        v = polls.loc[i, pct_cols].astype(float).to_numpy()
        v = np.where(np.isnan(v), 0.5, v)            # lists reported only as "below 1%"
        shares.loc[i] = v / v.sum() * (100 - OTHER_SHARE)
    # Seat-only polls (Channel 13): seats / 120 of the vote above the threshold.
    below_avg = (shares[has_pct].where(polls.loc[has_pct, seat_cols].to_numpy() == 0)
                 .sum(axis=1).mean())
    for i in polls.index[~has_pct]:
        above = 100 - OTHER_SHARE - below_avg
        for p in TRACKED:
            s = polls.loc[i, f"seats_{p}"]
            shares.loc[i, p] = s / 120 * above if s > 0 else polls.loc[i, f"pct_{p}"]
    return shares


# ---------------------------------------------------------------- 2. averaging
def firm_averages(polls, shares):
    """Each firm's weighted average share per list (recency x sample size)."""
    age = (REFERENCE_DATE - polls["end"]).dt.days.clip(lower=0)
    w = 0.5 ** (age / HALF_LIFE_DAYS) * np.sqrt(polls["n"].clip(upper=SAMPLE_CAP) / 500)
    out = {}
    for firm, idx in polls.groupby("pollster").groups.items():
        s = shares.loc[idx].to_numpy()
        ww = w.loc[idx].to_numpy()[:, None]
        mask = ~np.isnan(s)
        avg = np.nansum(s * ww, axis=0) / np.maximum((ww * mask).sum(axis=0), 1e-12)
        avg[mask.sum(axis=0) == 0] = np.nan
        out[firm] = avg
    return pd.DataFrame(out, index=TRACKED).T


# ---------------------------------------------------------------- 3. simulation
def simulate(firms, rng):
    F = firms.to_numpy()
    mask = ~np.isnan(F)
    W = rng.dirichlet(np.full(len(F), DIRICHLET_ALPHA), size=N_SIM)   # random firm weights
    means = (W @ np.nan_to_num(F)) / (W @ mask)
    central = np.nansum(F, axis=0) / mask.sum(axis=0)                   # equal firm weights
    sd = PARTY_SD_COEF * np.sqrt(np.clip(means, 0.1, None))
    sd *= np.array([NEW_LIST_MULT if p in NEW_LISTS else 1.0 for p in TRACKED])
    sim = np.clip(means + rng.normal(0, 1, means.shape) * sd, 0, None)
    sim = sim / sim.sum(axis=1, keepdims=True) * (100 - OTHER_SHARE)
    return sim, central


# ---------------------------------------------------------------- 4. seats
def dhondt(votes, seats_total, max_div=60):
    """D'Hondt (Bader-Ofer) for each row of votes; seats_total is a scalar or a per-row array."""
    n, k = votes.shape
    seats_total = np.broadcast_to(np.asarray(seats_total), (n,))
    q = (votes[:, :, None] / np.arange(1, max_div + 1)).reshape(n, -1)
    q_sorted = -np.sort(-q, axis=1)
    idx = np.clip(seats_total - 1, 0, q.shape[1] - 1)
    cutoff = np.where(seats_total > 0, q_sorted[np.arange(n), idx], np.inf)
    return (q.reshape(n, k, max_div) >= cutoff[:, None, None]).sum(axis=2)


def surplus_pairs():
    lists = pd.read_csv("data/lists_2026.csv").dropna(subset=["surplus_partner"])
    pairs = {tuple(sorted((a, b))) for a, b in zip(lists["letters"], lists["surplus_partner"])}
    return [(a, b) for a, b in pairs if a in TRACKED and b in TRACKED]


def allocate(sim):
    """Seats per tracked list for every simulated result."""
    qualify = sim >= THRESHOLD
    votes = np.where(qualify, sim, 0.0)
    slots = votes.copy()
    merged = []
    for a, b in surplus_pairs():                  # partners share remainders if both pass
        ia, ib = TRACKED.index(a), TRACKED.index(b)
        both = qualify[:, ia] & qualify[:, ib]
        slots[:, ia] = np.where(both, votes[:, ia] + votes[:, ib], votes[:, ia])
        slots[:, ib] = np.where(both, 0.0, votes[:, ib])
        merged.append((ia, ib, both))
    seats = np.zeros_like(sim, dtype=int)
    for start in range(0, len(sim), 5000):
        seats[start:start + 5000] = dhondt(slots[start:start + 5000], 120)
    for ia, ib, both in merged:                   # split each pair's seats between the partners
        split = dhondt(votes[:, [ia, ib]], seats[:, ia].copy())
        seats[:, ia] = np.where(both, split[:, 0], seats[:, ia])
        seats[:, ib] = np.where(both, split[:, 1], seats[:, ib])
    return seats


# ---------------------------------------------------------------- 5. submission
def best_submission(seats, total=120):
    """Per-list medians, then +/-1 steps with the smallest expected-error increase until `total`."""
    x = np.median(seats, axis=0).round().astype(int)

    def marginal(i, step):
        s = seats[:, i]
        return np.mean(np.abs(s - (x[i] + step))) - np.mean(np.abs(s - x[i]))

    while x.sum() != total:
        step = 1 if x.sum() < total else -1
        costs = [marginal(i, step) if x[i] + step >= 0 else np.inf for i in range(len(x))]
        x[int(np.argmin(costs))] += step
    return x


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)
    polls = load_polls()
    firms = firm_averages(polls, poll_shares(polls))
    sim, central = simulate(firms, rng)
    seats = allocate(sim)
    submit = best_submission(seats)

    names = pd.read_csv("data/lists_2026.csv").set_index("letters")["name"]
    table = pd.DataFrame({
        "name": [names[p] for p in TRACKED],
        "vote_share": central.round(2),
        "p_pass": (seats > 0).mean(axis=0).round(2),
        "seats_p10": np.percentile(seats, 10, axis=0),
        "seats_median": np.median(seats, axis=0),
        "seats_p90": np.percentile(seats, 90, axis=0),
        "submit": submit,
    }, index=pd.Index(TRACKED, name="letters"))
    print(f"{len(polls)} polls from {polls['pollster'].nunique()} firms "
          f"({polls['end'].min():%d %b}–{polls['end'].max():%d %b})")
    print(table.drop(columns="name").to_string())
    print(f"Total seats submitted: {submit.sum()}")
    table.to_csv("forecast.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
