# Credit scorecard serving and drift monitoring

I built a scoring API for an application credit scorecard, containerised it, deployed it to
Google Cloud Run where it is
[live now](https://credit-scorecard-api-403429711696.us-central1.run.app/docs), and then built a
drift monitoring system on top of it that keeps running: a stream of scoring requests, rolling
population and characteristic stability indices against the training time reference, threshold
alerting with a debounce so it does not fire on noise, and a dashboard showing all of it.

The model here is deliberately not the interesting part. It is a refit of the same simple
weight of evidence and logistic scorecard methodology I used on the application scorecard this
accompanies, kept unchanged on purpose. What I wanted to build was the part that comes after a
model is finished: serving it, watching it, and deciding when to interrupt somebody. Every
other project in my portfolio is analysis in a notebook or a scripted pipeline. Nothing was
ever served, and nothing kept running.

The thing I most want to be judged on is the alerting design, and specifically what it
declines to send.

## What is real and what is simulated

Stating this first because it determines how everything below should be read.

**Real.** The data is the Kaggle Home Credit Default Risk `application_train.csv`, 307,511
applications. The model is fitted on 215,257 of them and evaluated on 92,254 that take no part
in fitting. Every application scored in the run below is a genuine, unmodified row from that
held out slice. No feature value anywhere in this project is synthetic. Every request goes
over HTTP to the live API, which does the scoring and writes its own log. Every number in this
README comes from running `make demo`, and the raw outputs are in `reports/`.

**Simulated.** The traffic. Specifically, the arrival order and the timing. There is no
production system here and no real users. The stream decides which real applications arrive
when, and in the later portion it deliberately reweights that choice to introduce a known
population shift. That shift is mine. I chose it, calibrated it, and it is the reason the
monitor has something to detect.

**A proxy, not a fact.** `application_train.csv` has no date column, so there is no true time
ordering. I split on `SK_ID_CURR` ascending and treat the later ids as later applications.
That is a documented assumption about sequence, not a verified origination date. What is true
regardless is that no held out row takes any part in fitting.

I could not use naturally occurring drift because there is none to use: the held out slice is
a later portion of the same static extract, so its distribution is close to the training slice
by construction. A monitoring system that was never once observed to detect anything proves
nothing, so I injected a shift I could measure the response to. That makes this evidence that
the mechanism works. It is not evidence about how this population behaves in the field.

## The system

```
                    application_train.csv  (307,511 rows, no date column)
                                 |
                    split on SK_ID_CURR ascending
                       /                        \
              training slice                 held out slice
              215,257 rows                    92,254 rows
                    |                               |
              [ make train ]                        |
                    |                               |
        models/scorecard.joblib                     |
        WOE bins + logistic + band cutoffs          |
        + training time reference distributions     |
                    |                               |
                    v                               v
            +---------------+   HTTP POST /score   +------------------+
            |  FastAPI      | <------------------- |  stream replay   |
            |  scoring API  |                      |  (simulated)     |
            +---------------+                      +------------------+
                    |
              writes every scored request, with the bin
              each characteristic fell into
                    |
                    v
            +---------------------+       +----------------------+
            |  SQLite store       | <---> |  monitor (scheduled) |
            |  scoring log        |       |  PSI / CSI / bands   |
            |  drift metrics      |       |  debounce + cooldown |
            |  alerts             |       +----------------------+
            +---------------------+
                    |
                    v
            +---------------------+
            |  Streamlit          |
            |  dashboard          |
            +---------------------+
```

The reference distributions are captured at fit time and frozen into the artifact. This is the
single most important design decision in the project. A baseline recomputed periodically from
recent traffic drifts along with the traffic and never raises anything, which is the most
common way drift monitoring is built wrong.

## Quick start

Local, and this is the path with measured output behind it:

```bash
make install && make demo
```

That fits the card, starts the API, posts the simulated stream to it over HTTP, runs the
monitor over every complete window, writes the history to `reports/`, and prints a summary. It
took 143 seconds on my machine, see `reports/end_to_end_summary.json` for the exact figure.

Then to look at it:

```bash
make dashboard
```

The containerised path, which stands up all five services in dependency order:

```bash
docker compose up --build
```

The API has been deployed to Google Cloud Run at the URL below. It is currently returning 503
and the deployed image predates the removal of `CODE_GENDER`, so treat it as a record of the
deployment rather than a working demo. See "Containerisation and cloud deployment" below:

**https://credit-scorecard-api-403429711696.us-central1.run.app/docs**

## Getting the data

The raw CSV is not in this repository. It is roughly 158 MB, which does not belong in git.

```bash
kaggle competitions download -c home-credit-default-risk -f application_train.csv -p data/raw
```

Unzip it so that `data/raw/application_train.csv` exists. Any copy of the Kaggle file works.
Only `application_train.csv` is needed, not the other tables.

## The scorecard

A refit of established methodology, not new modelling. Quantile prebins merged until every bin
holds at least 3% of the population and the bad rate moves in one direction, missing values
kept as their own bin rather than imputed, characteristics kept above an information value
floor of 0.02 and dropped when correlated above 0.75, logistic regression on the weight of
evidence columns, scaled to 600 points at 50:1 odds with 20 points per doubling.

14 characteristics retained from 19 candidates. Measured on the 92,254 held out applications:

| Metric | Training | Held out |
|---|---|---|
| AUC | 0.7387 | 0.7449 |
| Gini | 0.4774 | 0.4899 |
| KS | 0.3545 | 0.3674 |
| Bad rate | 8.13% | 7.95% |

Bands are set as percentiles of the training score distribution and then frozen as absolute
cutoffs, so the API applies fixed numbers rather than recomputing quantiles per request. On
the held out slice:

| Band | Score | Share | Bad rate |
|---|---|---|---|
| Decline | below 532.45 | 9.8% | 25.3% |
| Refer | 532.45 to 553.26 | 20.0% | 12.6% |
| Approve | 553.26 and above | 70.2% | 4.2% |

Held out AUC slightly above training is normal for a regularised model on a different slice
and is not something I have tuned toward.

### Characteristics deliberately excluded

`CODE_GENDER` is not in the model, and it is not in the request contract either.

Gender is a prohibited basis under ECOA and Regulation B. A card that allocates points on it
cannot be signed off however well it ranks, so the question was never whether the lift
justified it. I measured the cost anyway, because "it was not worth much" is a weaker claim
than a number: held out AUC falls from 0.7463 to 0.7449, fourteen ten thousandths, and the
characteristic ranked eleventh of twenty by information value. It was never load bearing.

Excluding it at the boundary rather than merely dropping it from `features.categorical` is
deliberate. A field the service still accepts is one refit away from being scored on again,
and a caller who sends it should be told it was refused rather than left to assume it counted.
So `CODE_GENDER` is absent from `RAW_INPUTS`, absent from the pydantic request model, and a
request carrying it comes back 422 naming the field.

What this does not do is make the card fair. Gender remains recoverable from the
characteristics that stayed, and proxy discrimination through correlated inputs is a real
effect that removing one column does not touch. Measuring it needs outcome data by protected
class, which this extract does not carry. Excluding the direct use is the necessary step here,
not the sufficient one.

## The scoring API

`POST /score` takes raw application fields, `GET /health` reports what is loaded, `GET /model`
reports what is deployed, which is the first question in any model review.

The derived characteristics, the ratios and the ages, are computed **server side** by the same
`src/features.py` that training calls. Training and serving skew in derived features is a
common cause of a model that validates well and then behaves oddly in production, and one
implementation is the cheapest defence against it. There is a test asserting that a record
transformed alone matches the same record transformed inside a batch.

Validation is strict, and the reason is credit specific. A scorecard maps every input into a
bin, and every bin has a weight of evidence including the missing bin, so a malformed input
never fails loudly on its own. An age of 4,000 or a string where a number belongs would
quietly land in some bin and come back as a confident looking score. So the boundary refuses:

| Request | Response |
|---|---|
| `"EXT_SOURCE_2": "0.35"` | 422, `EXT_SOURCE_2: Input should be a valid number` |
| `"EXT_SOURCE_2": 1.4` | 422, `Input should be less than or equal to 1` |
| `"DAYS_BIRTH": 9461` | 422, `Input should be less than or equal to -6570` |
| `"NAME_EDUCATION_TYPE": "PhD"` | 422, lists the five fitted levels |
| `"EXT_SOURCE_TWO": 0.4` | 422, `Extra inputs are not permitted` |
| `AMT_ANNUITY > AMT_CREDIT` | 422, cross field rule |
| `"EXT_SOURCE_1": null` | 200, missing is a legitimate value with its own fitted bin |

Three specifics worth calling out. Strict mode means `"0.35"` is refused rather than coerced,
because a string arriving where a float belongs means the caller has a bug and returning a
score would hide it. `extra="forbid"` means a misspelled field is refused rather than ignored,
which is the worst case: a plausible score computed without a characteristic the caller
believed they had sent. And a rejected request is never written to the scoring log, so caller
side bugs cannot contaminate the drift baseline and send someone hunting for a population
change that never happened.

The API also checks the published request contract against the fitted artifact at startup, and
the two directions of a disagreement are treated differently on purpose. A level the model was
fitted on that the contract does not accept is fatal: the service would answer 422 to an
application the card was built to score, and nothing in the rejection would say why. A level
the contract accepts that the fit cannot resolve is not fatal, it is reported on stderr. Every
such level scores through the catch all bin, which is exactly what the binning already does to
any level below the 1% population floor, so no response changes.

That asymmetry was a fix, not the original design. The first version demanded exact equality
in both directions, which meant a rare level absent from a refit's training slice took the
whole service down at boot. On this data that is not hypothetical: `Maternity leave` and
`Unknown` rest on two rows each in 215,257, and under a random split the API failed to start
on 8% of refits. It was refusing to serve over a distinction the card cannot draw, since six
of the declared levels share the catch all in a normal fit.

Measured over the 40,000 request run: mean latency 2.53 ms, p95 2.65 ms, 285.5 requests per
second single threaded, 0 rejections. See `reports/end_to_end_summary.json` for the full
latency distribution and `reports/stream_manifest.json` for the throughput figure.

## Containerisation and cloud deployment

An earlier version of this README said the image had never been built and the stack had never
been run. That is no longer true. The stack has been built and run locally, and the API is
deployed and serving on Google Cloud Run:

**https://credit-scorecard-api-403429711696.us-central1.run.app/docs**

That link is the interactive API documentation, which is the useful thing to open in a browser.
The bare root path returns 404 on purpose, since no route is defined there.

### What runs locally

`docker compose up --build` stands up the whole thing. What I actually observed running it, on
Docker 29.7.2 against an arm64 host:

- `trainer` fitted the card inside the container on the full 215,257 rows and exited 0,
  reproducing the host's figures exactly: 14 characteristics, holdout AUC 0.7449, Gini 0.4899,
  KS 0.3674, cutoffs 532.45 and 553.26. The `reports/model_performance.json` the container
  wrote differed from the host's in its `trained_at` timestamp and in nothing else, so the
  containerised training path is not a different pipeline that happens to run.
- `api` reached `healthy` and reported 14 characteristics.
- A `/score` request with a complete application returned 591.09 in band `approve`, the same
  score to the cent the host returns for that application.
- The same request with `CODE_GENDER` added came back 422, so the exclusion holds through the
  served path rather than only in the tests.
- `monitor` and `dashboard` reached `healthy` as well, which is the point of giving each role
  its own healthcheck. Both used to inherit the image's probe against the API port, which
  neither of them serves, and so reported unhealthy for their whole lives while working
  perfectly. The monitor's own probe reported a wake up nine seconds old across 26 recorded
  runs, which is the check doing its job rather than a stub returning zero.
- `dashboard` served HTTP 200.

That was an arm64 build, this laptop's native architecture. Whether the pinned wheels resolve on
linux/amd64 is a separate question, and it is answered separately below, because Cloud Run
required an explicit `--platform=linux/amd64` rebuild before it would accept the image.

The structure is a single image serving four roles, chosen by command rather than built four
times, dependencies installed before source is copied so editing a module does not invalidate
the slow layer, a non-root user, and a compose file with five services in dependency order.
Each role states its own healthcheck or disables the inherited one: the image declares a check
against the API port because most roles built from it serve that port, and the two that do not
would otherwise report unhealthy for their whole life while working perfectly. The monitor's is
a staleness check on its own run log rather than a port probe, since the failure worth catching
there is a loop that quietly stopped waking up. `scripts/validate_compose.py` now fails on any
service that leaves the question unanswered. `trainer` runs once and must exit cleanly before `api` starts, so the API can
never come up on a missing artifact. `stream`, `monitor` and `dashboard` wait for the API to
report healthy. Data, the artifact and the SQLite file are bind mounted rather than baked in.

`make compose-check` still exists and still passes. It is a static check that catches the
structural mistakes cheaply, a `depends_on` target that does not exist, a command pointing at a
renamed module, a missing bind mount source, a duplicated port. It is no longer the only
evidence, but it is faster than a build when something structural changes.

### Why there are two Dockerfiles

`Dockerfile` is the local one. It deliberately leaves the model out, because `.dockerignore`
excludes `models/` and compose trains a fresh artifact into a bind mount on every run. For
local development that is the right shape.

Cloud Run has no bind mounts and no host filesystem to mount from, so an image deployed there
has to be self contained. `Dockerfile.cloudrun` is the same image plus `COPY models/ ./models/`,
paired with `Dockerfile.cloudrun.dockerignore`, which is Docker's per Dockerfile ignore file
convention and lets `models/` through while keeping every other exclusion. Two Dockerfiles for
two genuinely different deployment shapes, rather than one Dockerfile awkwardly serving both and
breaking local development in the process.

I checked the deployed shape before pushing it anywhere, by running the Cloud Run image locally
with no volume mounts at all. `/health` reported `requests_scored: 0`, confirming a genuinely
fresh instance rather than one reading a leftover database, and `/score` returned the same
592.64. So the baked in artifact is the right one, not merely present.

### Getting it onto Cloud Run

The first build was arm64, which is this laptop's native architecture, and Cloud Run rejected it
outright with an explicit error requiring amd64. The rebuild used `--platform=linux/amd64` and
that is the image that is deployed. Worth keeping in mind if you rebuild on Apple silicon,
because nothing else in the toolchain warns you.

```bash
docker build --platform=linux/amd64 -f Dockerfile.cloudrun \
  -t us-central1-docker.pkg.dev/PROJECT/credit-scorecard-service/api:1.0.0 .
docker push us-central1-docker.pkg.dev/PROJECT/credit-scorecard-service/api:1.0.0
gcloud run deploy credit-scorecard-api \
  --image=us-central1-docker.pkg.dev/PROJECT/credit-scorecard-service/api:1.0.0 \
  --region=us-central1 --memory=512Mi --max-instances=2 --allow-unauthenticated
```

`--max-instances=2` is deliberate. The service is genuinely open to the internet, so the cap is
what bounds the cost of that.

### What the live service actually does

Checked against the deployed URL, not against a local run:

| Request | Result |
| --- | --- |
| `GET /health` | 200, model version 1.0.0, 15 features, matching `trained_at` |
| `POST /score`, complete application | 200, score 592.64, PD 0.025159, band `approve` |
| `POST /score`, missing fields | 422, field by field validation errors |
| `GET /docs` | 200 |
| `GET /` | 404, no route defined |

The score returned by the deployed service is identical to the one the local container returned,
which is the point of baking the artifact in rather than refitting per environment.

Round trip latency from my machine is a median of 0.365s over 10 warm requests. That figure is
mostly distance to `us-central1` and says very little about the service, which handles these
requests in single digit milliseconds locally. I am quoting it because it is what I measured,
not because it is a meaningful throughput number.

### What is deployed and what is not

Only `api` is on Cloud Run. `dashboard` and `monitor` are not, and I would rather say why than
leave it looking like an oversight. The dashboard would be a straightforward second Cloud Run
service. The monitor is not, because it is a long running loop, and a perpetual loop on Cloud
Run means paying for an always on minimum instance. The honest shape for it is a Cloud Run Job
on a Cloud Scheduler trigger rather than a loop that never ends, which is a rewrite of how the
monitor is invoked rather than a deployment command. I have not done that yet.

Two further things are true of the deployed copy as of this writing, and both matter more than
the fact that it exists. It was built and pushed before `CODE_GENDER` was removed, so the image
on Cloud Run still serves the earlier fifteen characteristic card with gender among them, and
disagrees with this repository until it is rebuilt and redeployed. And the service is currently
answering 503 rather than serving at all: six requests over a minute every one of them failed,
which is longer than a cold start. I have not diagnosed it. I am not going to keep describing a
service as live while it returns 503, so the link below is left in place and labelled instead of
quietly removed.

### Known limitations of this deployment

- **The SQLite store does not survive a restart.** Cloud Run's filesystem is writable for the
  life of an instance but not durable across restarts or scale events, and there is no bind
  mount to point at the way local development has. Scoring is correct on every request, but
  `requests_scored` and the drift history reset whenever Cloud Run recycles the container. A
  production version would put that store in Cloud SQL or Firestore. This is the main reason the
  monitoring half of this project is best seen by running it locally.
- **`/docs` is public.** This is a considered choice rather than an oversight. The project is a
  demonstration over a public Kaggle dataset and there is nothing confidential in the schema. A
  real deployment would gate or disable it, along with putting authentication in front of
  `/score`.
- **There is no CI/CD.** The build and deploy above were run by hand. Wiring Cloud Build to
  deploy on a push to `main` is the obvious next step and is not done.

## The simulated scoring stream

The stream is an ordinary HTTP client. It posts held out applications to `/score` exactly as
any caller would, so the scoring log is written by the API itself rather than through a back
door into the database. It runs in two documented regimes.

**Regime 1, the first 16,000 requests.** Drawn from the held out slice with equal probability.
This is the control. The population genuinely has not shifted, so the indices should stay near
zero, and any alert here is a false positive. This half matters as much as the other: it is
what shows the monitor is not simply firing on noise.

**Regime 2, the remaining 24,000.** The same pool of real applications, drawn with unequal
probability, weighted toward lower external bureau scores and younger applicants, ramping up
over the first 30% of the regime and then holding. The effect is a portfolio quietly taking on
a riskier mix, which is the realistic version of this failure: not a broken feed, but an
origination channel gradually changing who it sends. Onset then plateau, because a real
channel change persists once it happens.

The strength was calibrated, not guessed. My first attempt used a strength of 3.0, which
produced a score PSI above 2.0. That is not a drift scenario, it is a broken feed, so I
measured the response curve and reduced it to 1.5, which peaks around 0.5 to 0.6. Severe, and
the sort of thing a risk function would want to hear about, but not absurd.

`reports/stream_manifest.json` records the exact parameters of every run. The seed is fixed,
so the run reproduces.

## The monitoring layer

Three signals per window, answering three different questions.

- **`psi_score`**, population stability on the score distribution against the training score
  deciles. Has the population changed.
- **`psi_band`**, the same calculation over the three decision bands. Has what the model
  decides changed. Deliberately coarse, because this is the version that maps onto approval
  rate and expected loss.
- **`csi`** per characteristic, against the training bin distribution. Which input moved. This
  is the attribution step.

I deliberately do not compute PSI on predicted probability. Score is a monotone transform of
log odds, so binning by score decile and by probability decile assign identical records to
identical bins. It would be the same quantity reported twice.

Windows are claimed by scoring log id, not by wall clock time, so every window is exactly
2,000 requests and the sampling variance of the index is constant. Time based windows vary in
size with traffic, and a quiet overnight window then produces a large index for no reason
other than having fewer records in it. A monitor that alerts every night at 3am gets muted. A
partial window waits rather than being scored, and the monitor records that it waited, so a
stopped monitor and a quiet one are distinguishable.

**Why the window is 2,000.** This is measured, not assumed. `make noise` draws repeatedly from
the training reference distribution itself, so the population is stable by construction and
every index produced is pure sampling noise. On the worst behaved characteristic, over 500
trials each:

| Window size | Worst p95 | Worst observed | Share crossing 0.10 | Share crossing 0.25 |
|---|---|---|---|---|
| 100 | 0.4868 | 0.8432 | 96.6% | 34.0% |
| 200 | 0.1593 | 0.3741 | 44.4% | 0.8% |
| 500 | 0.0634 | 0.0970 | 0.0% | 0.0% |
| 1000 | 0.0303 | 0.0479 | 0.0% | 0.0% |
| 2000 | 0.0147 | 0.0224 | 0.0% | 0.0% |
| 5000 | 0.0060 | 0.0105 | 0.0% | 0.0% |

At 100 requests a third of windows cross the alert threshold on a population that has not
moved at all. At 2,000 the noise floor sits an order of magnitude below the warn line, so a
reading above 0.10 means something happened. The 500 minimum is the smallest window where
noise never reached the warn line. A threshold without a known noise floor underneath it is
not a threshold, it is a number.

The 0.10 and 0.25 cut points themselves are the conventional scorecard thresholds. They are a
stated convention, not a calibration against observed incidents, because there are no real
incidents here. See `docs/business_case.md`.

## Alerting and debounce

Three rules, all of which exist to protect one thing: that an alert still means something in
month nine.

**Sustained breach.** An alert requires three consecutive breaching windows. A spike that
reverts never fires. The rule is consecutive rather than an average over a trailing period,
because a mean would let one extreme window drag two quiet ones over the line, which is the
exact failure being guarded against. Three breaches in four windows with a gap does not fire,
and there is a test for that specific case.

**Cooldown.** After firing, the same metric is suppressed for five windows. Drift does not
repair itself, so without this a sustained shift re fires every window, which is alert fatigue
by another route. The breach stays visible in the metric history and on the dashboard
throughout. What is suppressed is the repeat notification, not the observation.

**Attribution rather than duplication.** When the population shifts, every correlated
characteristic breaches at once. Before I added this tier, the run below fired sixteen alerts
for one event, which is the fatigue problem appearing inside my own system. A characteristic
breach is now folded into the population alert as attribution while a population metric is also
breaching. When the population metrics are quiet and one characteristic moves alone, it alerts
in its own right, because that is the broken feed case and it is genuinely separate.

No debounce state is stored. The consecutive count and the cooldown are derived from the
metric and alert history on each evaluation, so a restarted monitor reaches the same conclusion
as one that has been up for a week.

## What the run produced

From `make demo`: 40,000 requests, 0 rejected, 20 windows of 2,000, 4 alerts.

| Window | Regime | Score PSI | Band PSI | Approval rate | Mean PD | Status |
|---|---|---|---|---|---|---|
| 1 | stable | 0.0062 | 0.0003 | 69.3% | 0.082 | ok |
| 2 | stable | 0.0017 | 0.0001 | 69.7% | 0.082 | ok |
| 3 | stable | 0.0077 | 0.0008 | 69.2% | 0.082 | ok |
| 4 | stable | 0.0060 | 0.0019 | 69.8% | 0.081 | ok |
| 5 | stable | 0.0052 | 0.0003 | 70.0% | 0.083 | ok |
| 6 | stable | 0.0025 | 0.0011 | 70.0% | 0.080 | ok |
| 7 | stable | 0.0043 | 0.0010 | 69.3% | 0.080 | ok |
| 8 | stable | 0.0074 | 0.0008 | 68.8% | 0.082 | ok |
| 9 | ramping | 0.0138 | 0.0070 | 66.3% | 0.086 | ok |
| 10 | ramping | 0.0977 | 0.0769 | 57.8% | 0.104 | ok |
| 11 | ramping | 0.3025 | 0.2594 | 47.1% | 0.127 | **breach** |
| 12 | ramping | 0.5250 | 0.4317 | 39.9% | 0.143 | **breach** |
| 13 | plateau | 0.6414 | 0.5547 | 36.9% | 0.154 | **ALERT FIRED** |
| 14 | plateau | 0.5945 | 0.4898 | 38.8% | 0.150 | breach, cooldown |
| 15 | plateau | 0.5719 | 0.4857 | 39.1% | 0.148 | breach, cooldown |
| 16 | plateau | 0.6144 | 0.4793 | 38.9% | 0.148 | breach, cooldown |
| 17 | plateau | 0.6024 | 0.4983 | 38.0% | 0.150 | breach, cooldown |
| 18 | plateau | 0.5508 | 0.4346 | 40.4% | 0.147 | **ALERT REFIRED** |
| 19 | plateau | 0.5708 | 0.4554 | 39.4% | 0.148 | breach, cooldown |
| 20 | plateau | 0.6363 | 0.5464 | 36.8% | 0.151 | breach, cooldown |

This is the full lifecycle in one run:

- **Eight stable windows, zero false positives.** Score PSI between 0.0017 and 0.0077, against
  a warn threshold of 0.10. 16,000 requests through the monitor without a single spurious
  alert.
- **Detection is gradual, as designed.** The index moves at window 9, is clearly elevated at
  10, and first breaches at 11.
- **The debounce cost two windows and worked.** Breach began at window 11. The alert fired at
  window 13, on the third consecutive breach, naming windows 11, 12 and 13 in its audit trail.
- **The cooldown suppressed four repeats.** Windows 14 through 17 were still in breach and
  still recorded as such. No notification. The alert refired at window 18, exactly five windows
  after the first, and reported an 8 window consecutive run.
- **Approval rate fell from 69% to 37%** and mean predicted PD nearly doubled from 0.082 to
  0.151. This is the business consequence the monitoring exists to surface.

The attribution, ranked from the final window:

| Characteristic | CSI | Status | Injected? |
|---|---|---|---|
| EXT_SOURCE_2 | 0.4702 | alert | yes |
| AGE_YEARS | 0.4074 | alert | yes |
| EXT_SOURCE_3 | 0.3751 | alert | yes |
| EMPLOYED_YEARS | 0.1991 | warn | no, correlated with age |
| NAME_INCOME_TYPE | 0.1255 | warn | no, correlated with age |
| EXT_SOURCE_1 | 0.1049 | warn | no, correlated with the other bureau scores |
| ID_PUBLISH_YEARS | 0.0724 | ok | no |
| ... | | | |
| REGION_POPULATION_RELATIVE | 0.0229 | ok | no |
| GOODS_CREDIT_RATIO | 0.0136 | ok | no |
| NAME_EDUCATION_TYPE | 0.0134 | ok | no |

The three characteristics I injected the shift into rank first, second and third. The next
three are the ones genuinely correlated with them, which is correct behaviour rather than
noise. Characteristics I did not touch stayed flat, the whole bottom half of the card under
0.06 while the injected three sit above 0.37. An analyst handed this alert would be pointed at
the bureau scores and the age mix immediately.

Every alert record carries the metric, window, value, threshold, the consecutive run, the
specific window ids behind it and the attributed characteristics:

```
[2026-08-28T07:08:31.754+00:00] window 13 | psi_score | 0.6414 vs 0.25 | 3 consecutive
   breach windows: [11, 12, 13]
   attributed: ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'AGE_YEARS']
```

## The dashboard

`make dashboard`, on port 8501. Stability trends plotted against their thresholds with alert
windows marked, characteristic attribution ranked and selectable, approval rate and mean
predicted PD against their training baselines, the alert table with its audit trail, and the
monitor run history including the wake ups that found nothing to do.

The simulation warning is the first thing on the page.

## Tests

141 tests, `make test`, roughly 11 seconds. `make install-dev` first, since the test
runner is not in `requirements.txt`: nothing in the runtime image invokes pytest, and a
dependency that ships unused still has to be patched.

| File | Covers |
|---|---|
| `test_alerting.py` | debounce, cooldown, gap resets, the attribution tier, audit trail |
| `test_monitor.py` | window claiming, partial windows, flush, the debounce through the store |
| `test_drift.py` | the index against a hand computed value, empty bins, attribution |
| `test_api.py` | endpoint, 13 parameterised validation cases, rejected requests not logged |
| `test_features.py` | derivations, the sentinel, single record equals batch record |
| `test_binning.py` | monotonicity, missing bins, unseen categories, bins agree with weights |
| `test_store.py` | round trips, window claiming, schema migration |
| `test_contract.py` | the startup contract check, in both directions |
| `test_config.py` | how the project root is settled, and what happens when it cannot be |
| `test_dashboard.py` | the dashboard script runs, empty and populated |

Four of these exist because they caught something rather than confirmed something: the
schema migration test reproduces an actual failure where adding a column to the alerts table
did nothing to an existing database, the dashboard tests caught an invalid emoji argument that
would have crashed the page on load and a cache key that returned the wrong store's data, and
`test_contract.py` pins the asymmetry described under The scoring API, after the original
symmetric check turned a rare training level into a boot failure.

## Repository layout

```
src/
  config.py      typed access to config/config.yaml, no thresholds hard coded in logic
  features.py    raw fields to model features, the one shared transformation path
  binning.py     monotonic WOE binning, plus the bin assignment the CSI needs
  scorecard.py   logistic on WOE, points scaling, the frozen serving artifact
  train.py       fit, evaluate, freeze the artifact and the reference distributions
  schemas.py     the request contract and its validation rules
  scoring.py     the scoring path, transport agnostic
  api.py         FastAPI app
  store.py       SQLite store, with an additive schema migration
  drift.py       PSI, CSI, band drift
  alerting.py    debounce, cooldown, the attribution tier
  monitor.py     the scheduled loop
  stream.py      the simulated traffic generator
dashboard/app.py
scripts/
  run_end_to_end.py     the documented entrypoint behind every number here
  window_size_noise.py  the measured justification for the window size
  validate_compose.py   structural checks on the compose setup, no daemon needed
docs/business_case.md
Dockerfile                        local image, model trained into a bind mount
Dockerfile.cloudrun               Cloud Run image, model baked in, self contained
Dockerfile.cloudrun.dockerignore  per Dockerfile ignore file, lets models/ through
docker-compose.yml                the five service local stack
```

## Honest assessment

What this demonstrates: a model served behind a real API with validation strict enough to be
useful, a monitoring system that runs on a schedule against a frozen training baseline and
keeps working, and an alerting design where the interesting decisions are about restraint.
The window size is justified by measurement rather than convention. The attribution points at
the right characteristics. I found and fixed four real defects during the build and wrote
tests for each, the last of them a startup check that was strict in the wrong direction and
would have taken the service down over a category level the card cannot even represent.

Where it is weak, and I would rather say this than have it drawn out of me:

- **The deployment is a single container, not production infrastructure.** The API is genuinely
  built, deployed and serving on Cloud Run, which is more than a Dockerfile nobody ran, but it is
  one service with an in image model, a non durable SQLite store, no authentication, no CI/CD and
  no autoscaling story beyond a two instance cap. The monitor and dashboard are not deployed at
  all. Standing something up is the easy half of this.
- **The scoring stream is a documented simulation.** Real applications, real model, real HTTP,
  constructed arrival order, and a drift I injected myself. It proves the mechanism responds.
  It says nothing about how this population behaves in reality.
- **The alert thresholds are stated assumptions.** 0.10 and 0.25 are convention. They are not
  calibrated against real incident data, because there is none. The noise floor beneath them
  is measured, which is a different and weaker claim than calibration.
- **`SK_ID_CURR` order is a proxy for time.** The dataset has no date column. The held out
  slice is genuinely held out from fitting, but "later" is an assumption.
- **Nothing here monitors whether the model is still right.** Stability indices watch inputs
  and outputs. A card can be perfectly stable and quietly stop ranking risk. Catching that
  needs back testing against realised defaults, which this extract cannot support.
- **SQLite and a single threaded API are demonstration scale.** 286 requests per second is
  fine for this and nowhere near a real origination platform.
- **One model, no challenger.** A mature monitoring setup benchmarks the champion against
  something.
- **Excluding gender is not the same as demonstrating fairness.** `CODE_GENDER` is out of the
  card and out of the request contract, which is the necessary step. It is not the sufficient
  one. Gender stays recoverable from the characteristics that remain, and nothing here measures
  disparate impact, because that needs outcome data by protected class and this extract has
  none. Nor does the monitoring layer watch for it: a stability index says a distribution
  moved, never that a decision was unfair.

## License

MIT, see `LICENSE`.
