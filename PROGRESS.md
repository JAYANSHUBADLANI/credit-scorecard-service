# Progress

Running status for this project. Updated as phases complete.

## Status: all four phases complete, plus a cloud deployment

The Phase 2 caveat is closed. The container path was written and statically checked for a long
time without ever being built, because there was no Docker daemon available. That has now been
built, run, and deployed. The API is live on Google Cloud Run at
https://credit-scorecard-api-403429711696.us-central1.run.app, and `/docs` is the useful entry
point. Every number quoted in the README comes from a real run.

## Phase by phase

### Phase 1, the scorecard and the scoring API: done

- Refit the simple weight of evidence and logistic scorecard on `application_train.csv`,
  reusing the established methodology rather than building anything new. 14 characteristics
  retained from 19 candidates, `CODE_GENDER` excluded as a prohibited basis rather than
  selected out.
- Holdout Gini 0.4899, KS 0.3674, AUC 0.7449 on 92,254 held out applications.
- FastAPI `/score`, `/health` and `/model` endpoints.
- Input validation is strict: no silent coercion of strings to numbers, unknown fields
  rejected rather than ignored, explicit ranges on every field, two cross field rules.
- Verified that the served score matches offline batch scoring on 200 held out applications:
  maximum difference 0.005 points, which is the response rounding, and zero band mismatches.

### Phase 2, containerisation: done and verified

- `Dockerfile` (one image, four roles) and `docker-compose.yml` (trainer, api, stream,
  monitor, dashboard) are written, with dependency ordering, a healthcheck and a non-root
  user.
- `make compose-check` runs `scripts/validate_compose.py`, which checks the structural things
  cheaply: dependency targets exist, healthcheck conditions have a healthcheck to wait on, bind
  mount sources exist, command modules exist, ports do not collide, the Dockerfile copies
  everything the commands need, dependencies are pinned, and the raw CSV is excluded from build
  context. It passes.
- **Built and run.** `docker compose up trainer` fitted the card inside the container on all
  215,257 rows and exited 0, reproducing the holdout figures of the fit current at the time,
  AUC 0.7463 and Gini 0.4927, before `CODE_GENDER` was removed. `api` came up
  healthy and returned 592.64 in band `approve` for a complete application. `dashboard` served
  HTTP 200.

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
- 141 tests covering the transformation, the binning, the endpoint, the drift computation, the
  window mechanics, the debounce, the store and the dashboard.
- Single entrypoint: `make demo`.
- Business write up in `docs/business_case.md`.

### Phase 5, cloud deployment: api done, the rest open

- The API is deployed to Google Cloud Run in `us-central1`, 512Mi, capped at 2 instances, and
  is publicly reachable. Verified against the live URL, not a local run: `/health` 200 with
  matching model metadata, `/score` 200 returning the same 592.64 and band `approve` the local
  container returned, a malformed `/score` 422 with field by field errors, `/docs` 200, and `/`
  404 since no route is defined there. Median round trip 0.365s over 10 warm requests, which is
  mostly the distance to `us-central1`.
- `Dockerfile.cloudrun` is a second image that bakes the trained artifact in, because Cloud Run
  has no bind mounts to supply it the way compose does. `Dockerfile.cloudrun.dockerignore` is
  the per Dockerfile ignore file that lets `models/` through. The local Dockerfile is unchanged,
  so local development still trains fresh on every run.
- Verified the deployed shape locally first by running that image with no volume mounts at all,
  `requests_scored: 0` confirming a fresh instance and `/score` returning the identical 592.64.
- The first build was arm64, this laptop's native architecture, and Cloud Run rejected it with
  an explicit amd64 requirement. Rebuilt with `--platform=linux/amd64`, which is what is
  deployed.
- Only `api` is deployed. `dashboard` and `monitor` are not, see "Still open".

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

1. **The monitor needs rearchitecting before it can be deployed.** It is a long running loop,
   and a loop on Cloud Run means paying for an always on minimum instance. The right shape is a
   Cloud Run Job on a Cloud Scheduler trigger, invoked per run rather than looping forever. That
   is a change to how the monitor is entered, not a deployment command, which is why it is not
   done yet.
2. **The dashboard is not deployed.** Unlike the monitor this one is straightforward, a second
   Cloud Run service pointed at the same image with the Streamlit command. The open question is
   whether it is worth it given the point below about the database.
3. **The SQLite store does not persist on Cloud Run.** No bind mount exists there, so
   `requests_scored` and the drift history reset when an instance is recycled. Scoring itself is
   unaffected. Moving the store to Cloud SQL or Firestore is what would fix it properly, and it
   is a precondition for the dashboard being useful when deployed.
4. **No CI/CD.** The build, push and deploy were run by hand. Cloud Build triggered on a push to
   `main` is the obvious next step.
5. **No Cloud Monitoring dashboard.** Latency and error rates have only been observed through
   curl timings, not through Cloud Monitoring.
6. **Secret Manager is not needed and was not added.** I checked before building anything for
   it: the service holds no credentials, no API keys and no database password, since the config
   is thresholds and feature lists and the store is a local SQLite file. Adding Secret Manager
   here would be infrastructure for a problem that does not exist. It becomes relevant if the
   store moves to Cloud SQL or if authentication is put in front of `/score`.
7. **`/score` is unauthenticated and `/docs` is public.** A deliberate choice for a public
   portfolio demo over a public Kaggle dataset, and the wrong choice for anything real.

`.gitignore` commits all of `reports/`, about 100 KB, because those files are the evidence
behind every number in the README, and excluding them would make the claims uncheckable without
a 158 MB download and an eight minute run. Only the console log is excluded. The 158 MB raw CSV,
the 24 MB SQLite store, and the 50 KB fitted artifact are all excluded.

## Known limitations, stated up front

These are in the README too, deliberately, because an interviewer will find them:

- The deployment is one container on Cloud Run, not production infrastructure. It is genuinely
  built, deployed and serving, but with the model baked into the image, a non durable SQLite
  store, no authentication and no CI/CD, and with the monitor and dashboard not deployed at all.
- The scoring stream is a documented simulation built from real held out data, not real
  traffic, and the drift in it was deliberately injected by me.
- The alert thresholds are the conventional 0.10 and 0.25, a stated assumption rather than a
  calibration against real incident data. The noise floor beneath them is measured.
- `SK_ID_CURR` order is a proxy for application sequence. The dataset has no date column, so
  the "later slice" is later by id, not verifiably later in time. It is genuinely held out
  from fitting either way.
- Nothing here monitors whether the model is still right, only whether its inputs and outputs
  have shifted. Outcome monitoring needs realised defaults, which this extract does not have.
