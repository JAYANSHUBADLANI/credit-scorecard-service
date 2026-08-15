# Progress

Running status for this project. Updated as phases complete.

## Status: all four phases complete, with one caveat

The caveat is Phase 2. The Dockerfile and docker-compose.yml are written and statically
checked, but they have never been built or run, because there is no Docker daemon in the
environment this was built in. Everything else was executed and every number quoted in the
README comes from a real run.

## Phase by phase

### Phase 1, the scorecard and the scoring API: done

- Refit the simple weight of evidence and logistic scorecard on `application_train.csv`,
  reusing the established methodology rather than building anything new. 15 characteristics
  retained from 20 candidates.
- Holdout Gini 0.4927, KS 0.3687, AUC 0.7463 on 92,254 held out applications.
- FastAPI `/score`, `/health` and `/model` endpoints.
- Input validation is strict: no silent coercion of strings to numbers, unknown fields
  rejected rather than ignored, explicit ranges on every field, two cross field rules.
- Verified that the served score matches offline batch scoring on 200 held out applications:
  maximum difference 0.005 points, which is the response rounding, and zero band mismatches.

### Phase 2, containerisation: written, not verified

- `Dockerfile` (one image, four roles) and `docker-compose.yml` (trainer, api, stream,
  monitor, dashboard) are written, with dependency ordering, a healthcheck and a non-root
  user.
- `make compose-check` runs `scripts/validate_compose.py`, which checks what can be checked
  without a daemon: dependency targets exist, healthcheck conditions have a healthcheck to
  wait on, bind mount sources exist, command modules exist, ports do not collide, the
  Dockerfile copies everything the commands need, dependencies are pinned, and the raw CSV is
  excluded from build context. It currently passes.
- **Not done and cannot be done here: building the image and running the stack.** See the
  README section "Containerisation, and what is not verified".

### Phase 3, the stream, the monitoring layer and alerting: done

- Simulated scoring stream posting real held out applications to the live API over HTTP, in
  two documented regimes, stable then deliberately shifted.
- Rolling score PSI, band PSI and per characteristic CSI against the training time reference,
  computed on a schedule over fixed size windows.
- Debounced alerting with cooldown and an attribution tier, all tested.
- Drift strength was calibrated rather than guessed: an initial strength of 3.0 produced a PSI
  above 2.0, which is a broken feed rather than a drift scenario, so it was reduced to 1.5.

### Phase 4, dashboard, tests, entrypoint, docs: done

- Streamlit dashboard: stability trends against thresholds, characteristic attribution,
  approval rate and mean predicted PD against their training baselines, the alert table with
  its audit trail, and the monitor run history.
- 124 tests covering the transformation, the binning, the endpoint, the drift computation, the
  window mechanics, the debounce, the store and the dashboard.
- Single entrypoint: `make demo`.
- Business write up in `docs/business_case.md`.

## Things worth knowing that came out of building it

- **The window size needed evidence, not a guess.** `make noise` measures the stability index
  on a population that has not moved. At a 100 request window, 34% of windows cross the alert
  threshold on pure sampling noise. At 2000, the noise floor is roughly 0.02. That measurement
  is what sets `window_size` and `min_window_size`.
- **One drift event fired sixteen alerts before the attribution tier was added.** A population
  shift moves every correlated characteristic at once. That was the alert fatigue problem
  appearing inside my own system, and it is why characteristic breaches are folded into the
  population alert unless they occur on their own.
- **Information value was slightly inflated by empty bins.** The missing bin is always present
  so bin indices stay stable between training and serving, and the smoothing constant gave
  those empty bins a small information value contribution. Now excluded. The effect on this
  dataset was about 0.0001 and changed no feature selection, but it is wrong at small sample
  sizes.
- **Adding a column to the alerts table did nothing to an existing database.** `CREATE TABLE
  IF NOT EXISTS` does not alter a table that already exists, so the first end to end run after
  the attribution change died partway through the monitoring pass, having already posted
  40,000 requests. The store now applies additive migrations at startup, and
  `tests/test_store.py` reproduces the original failure.
- **The dashboard would have crashed on load.** `st.warning(icon="!")` is not a valid emoji
  and raises. Serving HTTP 200 proved nothing, because Streamlit returns the page shell before
  it runs the script. The dashboard tests use Streamlit's own harness, which runs the script
  properly, and they also caught a cached loader keyed on no arguments that returned the wrong
  store's data.

## Still open

1. **The container path has never been verified.** `docker compose up --build`, then check
   `http://localhost:8000/health` and `http://localhost:8501`. If it needs fixes, they will most
   likely be in the pinned wheels resolving on linux/amd64, or bind mount permissions on Linux.
2. **No remote exists yet.** This has stayed local so far; it goes up on GitHub once the above
   is checked.

`.gitignore` commits all of `reports/`, about 100 KB, because those files are the evidence
behind every number in the README, and excluding them would make the claims uncheckable without
a 158 MB download and an eight minute run. Only the console log is excluded. The 158 MB raw CSV,
the 24 MB SQLite store, and the 50 KB fitted artifact are all excluded.

## Known limitations, stated up front

These are in the README too, deliberately, because an interviewer will find them:

- Local Docker Compose is not production infrastructure, and here it is not even verified.
- The scoring stream is a documented simulation built from real held out data, not real
  traffic, and the drift in it was deliberately injected by me.
- The alert thresholds are the conventional 0.10 and 0.25, a stated assumption rather than a
  calibration against real incident data. The noise floor beneath them is measured.
- `SK_ID_CURR` order is a proxy for application sequence. The dataset has no date column, so
  the "later slice" is later by id, not verifiably later in time. It is genuinely held out
  from fitting either way.
- Nothing here monitors whether the model is still right, only whether its inputs and outputs
  have shifted. Outcome monitoring needs realised defaults, which this extract does not have.
