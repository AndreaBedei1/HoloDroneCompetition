# Cover letter — draft

> **Status: DRAFT.** Every factual statement below is supported by the
> manuscript. Items marked `TODO` are administrative details that cannot be
> derived from the repository and must be supplied by the corresponding author
> before submission. Do not submit with a `TODO` left in place.

---

`TODO: date`

To the Editors
*Robotics and Autonomous Systems*
Elsevier

**Re: Submission of "Marine Race Arena: A Reproducible Benchmark for
Onboard-Only Autonomous Underwater Gate Racing, Multi-Vehicle Evaluation and
Learned Control"**

Dear Editors,

We are pleased to submit the above manuscript for consideration as a regular
research paper in *Robotics and Autonomous Systems*.

**The problem.** Autonomous racing has become a standard instrument for
comparing perception, planning and control on ground and aerial platforms,
because a racing benchmark fixes the task, the course, the scoring rules and the
output format, and thereby makes methods from different groups comparable.
Underwater robotics has no equivalent instrument. Marine simulators have matured
considerably and supply hydrodynamics, sensor models and rendering of good
quality, but they supply the physical substrate rather than the evaluation
contract that sits above it. In practice this means two underwater controllers
reported on the same simulator can remain incomparable, because nothing prevents
one of them from consuming information the other refused, or from being scored
by logic it partly controls.

**Our contribution.** We present Marine Race Arena, a configurable benchmark and
evaluation framework for onboard-only autonomous underwater gate racing. Its
organizing principle is a strict, framework-enforced separation between autonomy
and evaluation: a controller reads only participant-local time, allow-listed
onboard sensing, physically received acoustic beacon packets and optional
teammate messages, while an independent referee uses privileged simulator state
to validate ordered gate crossings, charge penalties, terminate runs and compute
scores that never return to the control loop. On that contract we build an
onboard gate-perception and target-association front end, two interpretable
rule-based reference controllers, a cooperative fleet layer with a distributed
leader–follower policy over an acoustic-inspired channel, and a
learning-integration path that exposes the same observation and action interface
to reinforcement-learning agents.

**What the experiments show.** We report three separate bodies of evidence: 78
real-simulator runs across three heterogeneous circuits, two graded current
profiles, homogeneous fleets and three-vehicle coordination; a current-free
demonstration in which the reference controller completes all three circuits in
9 of 9 runs with no collision, out-of-bounds or wrong-direction event; and a
474-episode matched comparison of six controllers under identical seeds and
geometries. Three findings recur. Clean-track success is a poor predictor of
robustness — the same controllers fall from perfect completion to 2 and 3 of 5
under a moderate current on the same circuit. In a heterogeneous convoy,
coordination rather than individual speed is the binding constraint — a one-gate
leader–follower margin removes all 127.3 mean gate and world collisions of the
uncoordinated case. And learned policies are not simply worse than rules but
differently shaped — they complete 90.6–100 % of short generated geometries and
are significantly faster there than the rule baseline, yet reach only 20–40 % on
full circuits, with 90.6 % of those failures being lost gate sequences rather
than collisions.

**Fit with the journal.** The manuscript sits squarely within the scope of
*Robotics and Autonomous Systems*: it concerns autonomous systems operating with
partial onboard information under environmental disturbance, addresses
perception, control and multi-robot coordination within one experimental
framework, and contributes an evaluation methodology usable by other groups
rather than a single algorithm. The journal's readership includes precisely the
communities — marine robotics, multi-robot systems and robot learning — that
need a shared way to compare underwater autonomy methods.

**Novelty and honest scope.** We do not claim to be the first underwater
benchmark, nor a better simulator than any system we compare against; the
manuscript is explicit that the underwater simulators we build on exceed our
work on hydrodynamic fidelity, sensor variety and training throughput. The
distinction we do claim is narrower: among the systems we review, ours combines
an underwater racing protocol over ordered gates with an evaluator that is
structurally independent of the participant and an observation contract that
excludes privileged state from control, and extends both to team-level
evaluation. We also report negative results in full, including an exploratory
pose-aware perception component that fails under strong gate obliquity and a
learned policy that failed a pre-registered readiness gate on four of nineteen
criteria and then completed 0 of 30 holdout trials. We considered these
essential to include: a benchmark paper that reported only successes would
undermine the very property it argues for.

**Open source and reproducibility.** The complete implementation, the track
configurations, the controllers, the evaluation harness, the learning pipeline
and the aggregated result artifacts are publicly available at
<https://github.com/AndreaBedei1/HoloDroneCompetition>. Every experiment carries
a manifest recording the source commit or model SHA-256, the simulator version,
the track file hash, the adapter actually used, the fallback status, the current
profile and measured current vector, the control timestep and the seed. The
result tables and figures are regenerated from those artifacts by committed
scripts, and the manuscript documents the commands.

**Declarations.** This manuscript is original, has not been published elsewhere
and is not under consideration by another journal. `TODO: confirm whether any
portion has appeared in a conference proceedings, and if so state the venue and
describe the extension explicitly — the journal version adds the onboard
perception section, the referee formalization and boundary measurement, the
learning extension and its evaluation, and the current-free completion study.`
All authors have approved the submission. The authors declare no competing
financial interests or personal relationships that could have influenced this
work.

`TODO: suggested reviewers, if the journal requests them.`
`TODO: any prior interaction with the editorial office to reference.`

Thank you for considering our submission. We would be glad to answer any
questions.

Sincerely,

`TODO: corresponding author name`
`TODO: affiliation and postal address`
`TODO: email`
on behalf of Andrea Bedei, Lorenzo Bacchiani, Giovanni Pau and Roberto Girau
