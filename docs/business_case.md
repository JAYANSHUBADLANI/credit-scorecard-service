# Why a monitoring system rather than a one time check

## Who this is for

The reader I had in mind while building this is a model risk or validation function at a
consumer lender, responsible for the ongoing oversight of an application scorecard that is
already live and already making decisions. Not the person who built the card. The person who
has to answer, on a quarterly basis and in front of an audit committee, whether it is still
working.

That framing matters because it changes what the deliverable is. A modeller's question is
"does this card rank order risk". A validation function's question is "how would you know if it
stopped, and how quickly". Those are different questions, and the second one is answered by a
system rather than by an analysis.

## The regulatory expectation, stated plainly

Ongoing monitoring is not an optional extra that mature teams add. In the canonical US model
risk guidance, SR 11-7, validation rests on three elements: an evaluation of conceptual
soundness, ongoing monitoring, and outcomes analysis. Ongoing monitoring is one of the three,
not a footnote to the first. The stated purpose is to confirm that the model is implemented
appropriately and continues to perform as intended, and specifically to detect whether changes
in products, exposures, activities, clients or market conditions mean the model needs
adjustment, redevelopment or replacement. UK and European supervisory expectations are drafted
differently but land in the same place.

The practical consequence is that a population stability index computed once, at model build
time, in a notebook, does not discharge the expectation at all. It describes the data the
model was fitted on. The obligation is continuous, and it bites hardest at exactly the point
nobody is looking any more: eighteen months in, when the acquisition mix has quietly changed
and the person who built the card has moved teams.

That is the gap this project addresses. Not "can I compute a PSI", which is nine lines of
numpy, but "is there a thing that keeps computing it, decides when the answer matters, and
leaves a record".

## What the system is actually watching for

Three failure modes, and the design maps one signal onto each.

**The population changes.** The lender opens a new acquisition channel, or a broker starts
sending a different mix, and the applicants arriving no longer look like the ones the card was
fitted on. The card is not broken, but it is being used outside the population it was
calibrated for, and the band cutoffs no longer mean what the credit policy intended. Score PSI
catches this.

**What the model decides changes.** Whatever the cause, the approval rate has moved. This is
the one a business owner reacts to fastest, because it shows up in volume and in expected
loss before it shows up anywhere else. Band PSI catches this, and it is deliberately coarse:
three buckets that map onto the actual decisions, not ten statistical ones.

**A single input breaks.** A bureau feed goes stale and starts returning nulls, or an upstream
system changes a unit. The characteristic shifts hard, but the score may barely move, because
fourteen other characteristics dilute it. This is the failure most likely to go unnoticed, and
it is why the characteristic level indices alert in their own right when the population level
metrics are quiet.

## Why the thresholds are what they are, and what they are not

The 0.10 and 0.25 cut points are the conventional scorecard thresholds, in wide use and set
out in Siddiqi's *Credit Risk Scorecards* among others: below 0.10 no meaningful shift, 0.10
to 0.25 a moderate shift worth investigating, above 0.25 a significant shift.

**They are a stated convention, not a calibration.** I have not tuned them against observed
incidents on this portfolio, because there are no observed incidents on this portfolio: the
traffic is a documented simulation. If this were a real deployment, the honest version of this
section would report the false positive rate over the first two quarters and move the numbers
accordingly. Anyone reviewing this project should treat the thresholds as a defensible
starting point that would be re calibrated in service, and should ask that question.

What I did do is establish the noise floor beneath them, which is a separate matter and is
measured rather than assumed. `scripts/window_size_noise.py` draws repeatedly from the
training reference distribution itself, so the population is stable by construction and every
index it produces is pure sampling noise. On the worst behaved characteristic:

| Window size | Worst p95 | Worst observed | Share crossing 0.10 | Share crossing 0.25 |
|---|---|---|---|---|
| 100 | 0.4868 | 0.8432 | 96.6% | 34.0% |
| 200 | 0.1593 | 0.3741 | 44.4% | 0.8% |
| 500 | 0.0634 | 0.0970 | 0.0% | 0.0% |
| 1000 | 0.0303 | 0.0479 | 0.0% | 0.0% |
| 2000 | 0.0147 | 0.0224 | 0.0% | 0.0% |
| 5000 | 0.0060 | 0.0105 | 0.0% | 0.0% |

This is the justification for a 2000 request window and a 500 request minimum. At 100 requests
a third of windows would breach the alert threshold on a population that had not moved at all,
and the monitor would be worse than useless: it would be actively misleading. At 2000 the
noise floor sits an order of magnitude below the warn line, so a reading above 0.10 means
something happened. The minimum of 500 is the smallest window where noise never reached the
warn line in 500 trials.

The general point is that a threshold without a known noise floor underneath it is not a
threshold, it is a number. This is the part of the design I would most want to be asked about,
and it took ten minutes to measure.

## The alerting design, and why it is mostly about restraint

Three rules, and all three exist to protect one thing: that an alert from this system still
means something in month nine.

**Sustained breach, not any breach.** An alert requires the metric to be above its threshold
for three consecutive windows. A spike that reverts never fires. This costs detection latency,
one to two windows, and buys the property that the alert is worth reading. The rule is
consecutive rather than an average over a trailing period, because a mean would let one
extreme window drag two quiet ones over the line, which is the exact failure being guarded
against.

**Cooldown after firing.** Drift does not repair itself. Without a cooldown, a genuine
sustained shift re fires every window for as long as it lasts, which is alert fatigue arriving
by a different route. The condition stays visible in the metric history and on the dashboard
throughout, so nothing is hidden. What is suppressed is the repeat notification.

**Attribution rather than duplication.** When a population shifts, every correlated
characteristic breaches at once. In the run documented in the README, one shift put fifteen
characteristics over the threshold simultaneously. Firing sixteen alerts for one event is the
fatigue problem restated, so a characteristic breach is folded into the population alert as
attribution while a population metric is also breaching. When the population metrics are
quiet, a characteristic alerts on its own, because that is the broken feed case and it is
genuinely a separate incident.

The common thread is that the hard part of monitoring design is not detection. Detection is a
threshold. The hard part is deciding what not to send, because a monitoring system that gets
muted has failed completely, and it fails silently.

## What a reviewer should do when one fires

The alert record carries the metric, the window, the value, the threshold, the consecutive
windows behind it and the characteristics in breach. That is enough to start, and the intended
sequence is:

1. Read the attribution. If one characteristic dominates, suspect the feed before suspecting
   the population. A single input moving alone is far more often an engineering fault than a
   change in who is applying.
2. If the shift is spread across characteristics, it is a population change, and the question
   moves to acquisition: new channel, new campaign, new geography.
3. Check whether the band mix moved with it. A population shift that does not change the
   decision mix is worth understanding but is not urgent. One that moves the approval rate is
   a credit policy matter immediately, because expected loss has already changed.
4. Only then consider the model. Stability indices say nothing about discrimination. A card
   can be perfectly stable and quietly stop ranking risk, and catching that requires outcomes
   analysis against realised defaults, which needs the outcome window to have elapsed.

That last point is the honest limit of this whole system and it is worth stating clearly.
Everything here monitors inputs and outputs. None of it monitors whether the model is still
right, because being right is only observable once the loans have had time to go bad. Drift
monitoring is the early warning that runs in the meantime, and it is a leading indicator, not
a substitute for back testing against realised performance.

## What this would need before it was real

Stated plainly, because these are the questions an interviewer should ask:

- **The thresholds are uncalibrated.** Covered above. They are convention, and a real
  deployment would move them once it had a false positive history.
- **The traffic is simulated.** Real applications, real model, constructed arrival order, with
  a deliberately injected shift in the later portion. It demonstrates that the mechanism works.
  It says nothing about how this population behaves in the field.
- **There is no outcome monitoring.** No back testing against realised default, because the
  Kaggle extract has no post origination performance to test against.
- **There is no challenger.** A mature monitoring setup benchmarks the champion against
  something. This has one model.
- **The infrastructure is local.** Docker Compose on one machine, SQLite as the store. That is
  enough to demonstrate the design and nowhere near enough to run it.
