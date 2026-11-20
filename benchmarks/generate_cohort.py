"""Synthetic birth-cohort-like data: many features, many small effects, a weak outcome (E2).

CohortSystem(p, n_causes, signal_r2, seed) builds one system; .sample(n, rng, regime) draws data.

Structure (loosely modelled on CHILD; everything is synthetic):
  tiers      measured variables are placed in 4 time-ordered tiers: genetics 15%, prenatal 25%,
             early life 30%, microbiome-like 30%. The outcome Y (e.g. lung function) is measured
             after all of them, so it has no measured children: its Markov blanket = its parents.
  latents    n_latent unmeasured factors, each in a tier, each driving a block of about p/10
             measured variables (correlated clusters, like taxa or exposures with shared drivers).
             Two latents are also unmeasured causes of Y, so their measured children are PROXIES:
             predictive of Y without causing it.
  features   each measured variable gets Poisson(1.5) measured parents from earlier in the order,
             plus its latent block; mechanisms linear (60%) or a random nonlinearity; its explained
             variance is U(0.2, 0.6). 30% of variables are binary (thresholded before their
             children are generated, as exposures are).
  outcome    n_causes direct measured causes spread over all tiers, effect sizes exponential(1)
             (a few moderate, most small) plus two pairwise interactions; the two latent causes
             add the same total weight as 4 average measured causes (times latent_y_weight; 0 = no
             real proxies, so every non-cause is noise with respect to Y). Y = sqrt(signal_r2) * signal
             + sqrt(1 - signal_r2) * noise, so even the full signal (incl. latents) explains
             signal_r2 of the variance.

Regimes for .sample():
  iid           the training distribution
  proxy_break   the latent values feeding measured variables are redrawn independently of the ones
                feeding Y: proxies keep their marginal distribution but lose their link to Y
                (a marker that means something else at another site). Y's mechanism is unchanged.
  cause_shift   Y's measured causes are shifted by +1 SD (new exposure levels); mechanisms intact.
All variables are standardized with means/SDs from a large reference sample of the iid regime, so
shifted test data really are shifted.
"""
import numpy as np

from benchmarks.generate import _rand_fn

TIER_SHARE = [0.15, 0.25, 0.30, 0.30]
TIER_NAMES = ["genetics", "prenatal", "early_life", "microbiome"]


class CohortSystem:
    def __init__(self, p=200, n_causes=15, signal_r2=0.3, seed=0, n_latent=6, binary_frac=0.3, latent_y_weight=1.0):
        rng = np.random.default_rng(seed)
        self.p, self.signal_r2 = p, signal_r2
        sizes = np.floor(np.array(TIER_SHARE) * p).astype(int)
        sizes[-1] = p - sizes[:-1].sum()
        self.tier = np.repeat(np.arange(4), sizes)  # variables are already in time order
        # latents: each sits before its block; block = random variables in its tier or later
        self.lat_tier = rng.integers(0, 4, n_latent)
        self.lat_block = []
        for t in self.lat_tier:
            pool = np.flatnonzero(self.tier >= t)
            self.lat_block.append(rng.choice(pool, size=min(len(pool), max(3, p // 10)), replace=False))
        self.y_latents = rng.choice(n_latent, 2, replace=False)
        # measured parents and mechanisms
        self.parents, self.fns, self.lat_w, self.r2 = [], [], [], []
        lat_of = [[] for _ in range(p)]
        for k, block in enumerate(self.lat_block):
            for j in block:
                lat_of[j].append(k)
        for j in range(p):
            k = min(rng.poisson(1.5), j)
            pa = np.sort(rng.choice(j, size=k, replace=False)) if k else np.array([], int)
            self.parents.append(pa)
            if rng.random() < 0.6:
                w = rng.uniform(0.5, 1.5, len(pa)) * rng.choice([-1, 1], len(pa))
                self.fns.append([(lambda x, a=a: a * x) for a in w])
            else:
                self.fns.append([_rand_fn(rng) for _ in pa])
            self.lat_w.append({l: rng.uniform(0.5, 1.5) * rng.choice([-1, 1]) for l in lat_of[j]})
            self.r2.append(rng.uniform(0.2, 0.6))
        self.binary = set(rng.choice(p, size=int(binary_frac * p), replace=False).tolist())
        # outcome
        self.y_parents = np.sort(rng.choice(p, size=n_causes, replace=False))
        w = rng.exponential(1.0, n_causes) * rng.choice([-1, 1], n_causes)
        self.y_w = w / np.abs(w).sum()
        pairs = rng.choice(n_causes, size=(2, 2), replace=False)
        self.y_inter = [(self.y_parents[a], self.y_parents[b], 0.15 * rng.choice([-1, 1])) for a, b in pairs]
        self.y_lat_w = rng.choice([-1, 1], 2) * (4 / n_causes) / 2 * latent_y_weight
        # reference standardization and binary thresholds from a large iid sample
        self.thresh, self.mu, self.sd = {}, None, None
        ref = self._raw(20_000, np.random.default_rng(seed + 99_991), "iid", set_thresholds=True)
        self.mu, self.sd = ref[0].mean(0), ref[0].std(0) + 1e-12
        self.y_mu, self.y_sd = ref[1].mean(), ref[1].std() + 1e-12

    def _raw(self, n, rng, regime, set_thresholds=False):
        p = self.p
        L = rng.normal(size=(n, len(self.lat_tier)))  # latent values driving Y
        L_feat = rng.normal(size=L.shape) if regime == "proxy_break" else L  # ... and driving features
        X = np.zeros((n, p))
        ys = set(self.y_parents.tolist())
        for j in range(p):
            f = np.zeros(n)
            for fn, q in zip(self.fns[j], self.parents[j]):
                f += fn(X[:, q])
            for l, w in self.lat_w[j].items():
                f += w * L_feat[:, l]
            sd = f.std()
            sig = (f - f.mean()) / sd if sd > 1e-12 else np.zeros(n)
            r2 = self.r2[j] if sd > 1e-12 else 0.0
            x = np.sqrt(r2) * sig + np.sqrt(1 - r2) * rng.normal(size=n)
            if j in self.binary:
                if set_thresholds:
                    self.thresh[j] = np.median(x)
                x = (x > self.thresh[j]).astype(float)
            if regime == "cause_shift" and j in ys and self.sd is not None:
                x = x + self.sd[j]  # +1 reference SD, after any thresholding (binary: 0/1 -> shifted)
            X[:, j] = x
        Z = (X - self.mu) / self.sd if self.mu is not None else X
        s = Z[:, self.y_parents] @ self.y_w
        for a, b, w in self.y_inter:
            s = s + w * Z[:, a] * Z[:, b]
        s = s + L[:, self.y_latents] @ self.y_lat_w
        if self.mu is None:  # first (reference) pass: fix the signal scale
            self.s_mu, self.s_sd = s.mean(), s.std() + 1e-12
        s = (s - self.s_mu) / self.s_sd
        y = np.sqrt(self.signal_r2) * s + np.sqrt(1 - self.signal_r2) * rng.normal(size=n)
        return X, y

    def sample(self, n, rng, regime="iid"):
        X, y = self._raw(n, rng, regime)
        return (X - self.mu) / self.sd, (y - self.y_mu) / self.y_sd

    def proxies(self):
        """Measured variables driven by a latent cause of Y (predictive of Y, not causes)."""
        out = set()
        for l in self.y_latents:
            out |= set(self.lat_block[l].tolist())
        return sorted(out - set(self.y_parents.tolist()))


if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression
    for p in (50, 200, 500):
        s = CohortSystem(p=p, seed=0)
        rng = np.random.default_rng(1)
        X, y = s.sample(5000, rng)
        Xt, yt = s.sample(5000, rng)
        r2 = lambda cols: LinearRegression().fit(X[:, cols], y).score(Xt[:, cols], yt)
        print(f"p={p}: oracle-parents linear R2 {r2(list(s.y_parents)):.3f}, all-features {r2(list(range(p))):.3f}, "
              f"{len(s.proxies())} proxies, proxies-only {r2(s.proxies()):.3f}")
