# Schematic figures to generate with NanoBanana

Five conceptual figures need real graphic design rather than an auto-generated
diagram. Each entry below is a complete, self-contained generation prompt: it
states the target filename, the scientific message, the exact layout, the
permitted and forbidden arrows, the colour groups, the verbatim text, the aspect
ratio and the exact LaTeX insertion point.

**Shared house style for all five.** Flat vector illustration. Thin uniform
strokes (1.5–2 px at final size). Rounded corners on boxes. Generous white
space. Publication style, not marketing style: no drop shadows, no gradients on
boxes, no 3-D bevels, no glow, no photographic texture. One clean sans-serif
throughout (Inter, Source Sans or Helvetica). White or transparent background,
no outer border frame.

**Shared palette.**

| Role | Hex |
| --- | --- |
| Autonomy / controller group | `#2F6DB0` teal-blue |
| Evaluation / referee group | `#E07B39` warm amber |
| Learning / policy group | `#7A9E3F` olive |
| Perception group | `#9A6CB4` violet |
| Ink, text, outlines | `#12263A` |
| Panel fill | `#F2F5F9` |
| Neutral separator | `#9AA7B4` |
| Forbidden / prohibition | `#C0392B` |

**Delivery.** Vector PDF (preferred) or SVG. Place in
`article_journal/figures/generated/`. Each figure replaces the TikZ placeholder
named in its "Insertion point" section; the placeholder currently compiles, so
the manuscript is not blocked on these.

---

## SCHEMA 1 — Overall architecture and the information boundary

* **Filename:** `schema1_architecture.pdf`
* **Replaces:** `figures/architecture_diagram.tex` (currently a TikZ placeholder)
* **Figure number in the current build:** Fig. 2
* **Aspect ratio:** 16:7, landscape, full text width

### Purpose

This is the paper's central figure. It must make it **visually impossible** to
believe that referee state reaches the controller.

### Scientific message

A single simulator state is projected into two disjoint views. The controller
view drives the vehicle; the referee view drives the score. Information flows one
way only.

### Exact layout

Two horizontal lanes separated by a **thick dashed grey horizontal divider** that
runs the full width of the figure, with a padlock glyph at its centre and the
label **"information boundary"** immediately below the padlock.

**Far left, spanning the full height and crossing the divider:** a tall rounded
panel filled `#F2F5F9`, headed **"HoloOcean simulator"** with a smaller
subheading **"full state $x_t$"**. Inside it, three stacked small labels:
"vehicle dynamics", "gate geometry", "current field". This is the only element
allowed to touch both lanes, because it is the common source.

**UPPER LANE — autonomy, teal `#2F6DB0`.** Left to right, four boxes connected
by solid teal arrows:

1. **"Onboard sensors"** — icon row: camera, gauge, compass, wave. Sub-label:
   "camera, depth, IMU, DVL".
2. **"Official observation"** — sub-label: "local time, allow-listed sensors,
   received beacon packets, teammate inbox". Small teal tag on this box reading
   **"$\Phi_i(x_t)$"**.
3. **"Controller"** — sub-label: "any policy: rule, hybrid, learned".
4. **"Command"** — sub-label: "surge / sway / heave / yaw, clamped to
   $[-1,1]^4$".

From **"Command"**, a thick teal arrow curves down-left and re-enters the
simulator panel, labelled **"actuation"**. This closes the loop.

**LOWER LANE — evaluation, amber `#E07B39`.** Left to right, four boxes connected
by solid amber arrows:

1. **"Privileged simulator state"** — sub-label: "global pose, exact gate
   geometry, obstacles, true current". Small amber tag reading **"$\Psi(x_t)$"**.
2. **"Independent referee"** — sub-label: "ordered crossing test, event
   arbitration".
3. **"Penalties and termination"** — sub-label: "collision, out of bounds,
   stuck, missed gate".
4. **"Official result"** — sub-label: "progress, penalized time, ranking, event
   log".

### Arrows — draw exactly these and nothing else

| # | From | To | Colour | Style |
| --- | --- | --- | --- | --- |
| A | simulator panel | Onboard sensors | teal | solid |
| B | Onboard sensors | Official observation | teal | solid |
| C | Official observation | Controller | teal | solid |
| D | Controller | Command | teal | solid |
| E | Command | simulator panel | teal | solid, thick, curved |
| F | simulator panel | Privileged simulator state | amber | solid |
| G | Privileged simulator state | Independent referee | amber | solid |
| H | Independent referee | Penalties and termination | amber | solid |
| I | Penalties and termination | Official result | amber | solid |
| J | Command | Independent referee | grey | thin dashed, labelled "commands are observed, not returned" |

### Forbidden — must not appear

* Any arrow from any lower-lane box to any upper-lane box.
* Any arrow from "Official result" anywhere except off the right edge.
* Any arrow from "Privileged simulator state" into the upper lane.
* Any box straddling the divider except the leftmost simulator panel.
* Any shared tint or shared container merging the teal and amber groups.

Optionally add **one** thin `#C0392B` arrow from "Independent referee" pointing
up toward "Controller", struck through with a red X and labelled "never" —
include it only if it reads instantly as a prohibition; if ambiguous, omit it.

### Verbatim text

"HoloOcean simulator", "full state", "vehicle dynamics", "gate geometry",
"current field", "Onboard sensors", "camera, depth, IMU, DVL", "Official
observation", "local time, allow-listed sensors, received beacon packets,
teammate inbox", "Controller", "any policy: rule, hybrid, learned", "Command",
"surge / sway / heave / yaw", "actuation", "information boundary", "Privileged
simulator state", "global pose, exact gate geometry, obstacles, true current",
"Independent referee", "ordered crossing test, event arbitration", "Penalties
and termination", "collision, out of bounds, stuck, missed gate", "Official
result", "progress, penalized time, ranking, event log", "commands are observed,
not returned".

### Caption to use

> Runtime architecture and the information boundary. A single simulator state is
> projected into two disjoint views: the official observation $\Phi_i(x_t)$ that
> reaches the controller, and the privileged state $\Psi(x_t)$ that reaches the
> referee. The controller closes its loop through actuation; the referee
> observes issued commands but returns nothing. No referee-derived quantity ---
> expected gate index, completed-gate count, status or accrued penalty --- is
> reachable from the code path that constructs the observation.

### Insertion point

Already `\input{figures/architecture_diagram}` in
`sections/03_benchmark_model.tex`, immediately after the
`\subsection{Runtime Architecture}` heading. Replace the file body with an
`\includegraphics[width=\linewidth]{figures/generated/schema1_architecture.pdf}`
inside the existing `figure` environment; keep the label
`fig:runtime_architecture`.

---

## SCHEMA 2 — Onboard gate perception pipeline

* **Filename:** `schema2_perception.pdf`
* **Replaces:** `figures/perception_pipeline.tex`
* **Figure number in the current build:** Fig. 4
* **Aspect ratio:** 16:6, landscape, full text width

### Scientific message

Camera pixels alone produce an ambiguous list of candidate gates; the
controller's own acoustic bearing resolves which one is the target. The
resolution happens inside the participant's autonomy, not inside the framework.

### Exact layout

Two input streams entering from the left, converging on a decision block, one
output on the right.

**Upper stream — visual, violet `#9A6CB4`.** Five boxes, left to right:

1. **"FrontCamera"** — sub-label "RGB $640\times480$, $90^\circ$ FOV". Show a
   small thumbnail of a blue underwater frame with two faint gate rectangles.
2. **"Grayscale + Canny"** — sub-label "thresholds $(30, 80)$". Thumbnail: white
   edge outlines on black.
3. **"Closed contours"** — sub-label "bounding boxes". Thumbnail: two rectangles
   outlined.
4. **"Geometric filter"** — sub-label "size $\ge 5\%$, aspect $[0.45, 2.40]$,
   perimeter ratio $[0.50, 2.20]$, area $\ge 0.008$".
5. **"Candidate list"** — sub-label "up to 8, each with $(c_x, c_y, \alpha,
   \gamma)$". Draw as a small stack of three cards, the top two labelled
   "cand 1", "cand 2".

**Lower stream — acoustic, teal `#2F6DB0`.** Two boxes:

1. **"Received beacon packets"** — sub-label "all in-range transmitters,
   unlabelled". Draw as three cards labelled "B01", "B02", "B03" with a small
   question-mark glyph over the stack.
2. **"Expected beacon"** — sub-label "controller's own tracker state; bearing
   $\beta$, range $\tilde\rho$". Highlight the card "B01" in teal and grey out
   "B02", "B03".

**Centre-right — the decision block, drawn larger than the others**, headed
**"Acoustic–visual association"**, containing four stacked test rows, each with a
small pass/fail glyph on its left:

* "field of view: $|\beta| \le 70^\circ$"
* "implied bearing: $\hat\beta = -\arctan(c_x)$"
* "conflict rejection: drop all with mismatch $> 32^\circ$"
* "near field: $\tilde\rho < 1.6$ m requires $\alpha \ge 0.05$"

**Right — output.** One box **"Visual target"** with sub-label "$(c_x, c_y)$
normalized image error", and directly beneath it a second, greyed box **"or
none"** with sub-label "proceed on acoustic and inertial evidence".

**Bottom strip, spanning the width**, a thin panel filled `#F2F5F9` with a small
padlock and the text **"never used: simulator gate pose, global vehicle pose,
referee target index"**.

### Arrows

Solid violet along the upper stream; solid teal along the lower stream; both
converge into the association block; two arrows leave it, a solid one to "Visual
target" and a thin dashed grey one to "or none".

### Forbidden

* No arrow into the association block from anywhere except the two streams.
* No arrow from the bottom prohibition strip into any box.
* Do not draw a "ground truth" or "simulator" box anywhere in this figure.

### Caption to use

> The onboard gate-perception front end. The camera path yields an unlabelled
> list of candidate gates; the controller's own expected-beacon packet supplies
> the bearing that selects among them, and rejects candidates inconsistent with
> it. When no candidate survives, the front end reports that no visual target is
> available and the controller proceeds on acoustic and inertial evidence. No
> simulator gate pose, global vehicle pose or referee target enters any stage.

### Insertion point

Already `\input{figures/perception_pipeline}` in `sections/05_perception.tex`,
immediately after the section heading. Keep the label `fig:perception_pipeline`.

---

## SCHEMA 3 — LocalCourseTracker and the Center-then-Commit passage

* **Filename:** `schema3_tracker.pdf`
* **Replaces:** `figures/center_commit_states.tex`
* **Figure number in the current build:** Fig. 6
* **Aspect ratio:** 16:8, landscape, full text width

### Scientific message

Both reference controllers share one state machine and one advancement
criterion; they differ only in whether the visual servo remains active during
the passage itself.

### Exact layout

Two coupled halves, side by side, sharing a horizontal band at the bottom.

**Left half — the state machine.** Six rounded boxes in a vertical column, top
to bottom: **SEARCH**, **APPROACH**, **VISUAL\_ALIGN**, **COMMIT**,
**VERIFY\_EXIT**, **ADVANCE**, then a seventh below, **FINISHED**, drawn with a
double outline. Forward arrows down the column, each labelled with its guard:

* SEARCH → APPROACH: "expected packet received"
* APPROACH → VISUAL\_ALIGN: "$|\beta|$ small"
* VISUAL\_ALIGN → COMMIT: "centred frames persist"
* COMMIT → VERIFY\_EXIT: "forward DVL displacement"
* VERIFY\_EXIT → ADVANCE: "exit clearance evidence"
* ADVANCE → FINISHED: "last beacon confirmed"

Two return arrows routed outside the column: COMMIT → APPROACH on the right,
labelled "visual loss"; ADVANCE → SEARCH on the left, labelled "next gate".

Shade **COMMIT** and **VERIFY\_EXIT** with a light amber wash and attach a small
bracket to their right labelled **"passage phase"**.

**Right half — the passage, drawn as a plan view.** A gate aperture in
cross-section, seen from above, with a vehicle approaching from the left along a
path. Draw **two** paths through the same aperture:

* A teal path labelled **"Continuous Servo"**, which continues to bend toward
  the observed image centre inside the aperture, with three small camera glyphs
  along it and a wobble near the aperture plane.
* An amber path labelled **"Center-then-Commit"**, which bends until a marked
  point labelled **"lock"** just before the aperture and then runs straight
  through, with the camera glyphs after the lock point struck through and a
  small note **"visual correction suspended; depth and attitude held"**.

**Bottom band, spanning both halves**, filled `#F2F5F9`: three small labels in a
row — "same observation", "same tracker", "same advancement criterion" — with a
short vertical tick joining each to the half above.

### Forbidden

* Do not draw a gate position, gate normal or coordinate frame supplied by the
  simulator; the plan view is an explanatory schematic, not a controller input.
* Do not imply the tracker receives referee confirmation: no amber arrow may
  enter the left half.

### Caption to use

> The shared `LocalCourseTracker` state machine (left) and the two passage
> strategies it drives (right). Both reference controllers use the same
> observation, the same state machine and the same three-modality advancement
> criterion; they differ only inside the shaded passage phase. Continuous Servo
> keeps applying image-centroid corrections through the aperture, where the
> detected contour becomes partial; Center-then-Commit suspends visual
> correction after the lock and holds depth and attitude instead.

### Insertion point

Already `\input{figures/center_commit_states}` at the end of
`sections/06_controllers.tex`. Keep the label used by that file.

---

## SCHEMA 4 — Learning pipeline and curriculum

* **Filename:** `schema4_learning.pdf`
* **New figure**, to be inserted in `sections/09_learning.tex`
* **Aspect ratio:** 16:9, landscape, full text width

### Scientific message

The learned policy is trained through the same environment, observation and
action contract used to evaluate every other controller, and the curriculum
advances on measured generalization rather than on elapsed steps.

### Exact layout

Two stacked bands.

**UPPER BAND — the training loop, olive `#7A9E3F` with teal observation
elements.** A closed cycle drawn left to right and returning:

1. **"HoloOcean + arena + referee"** — sub-label "same components as evaluation".
2. → **"Official observation"** (teal outline) → **"Observation encoder"** —
   sub-label "35 bounded features, sequence-length independent". Attach a small
   callout listing the six groups: "acoustic 7", "visual 5", "depth 4",
   "IMU/DVL 6", "previous action 4", "local derivatives 9".
3. → **"PPO policy"** — sub-label "MLP $256\times256$, Gaussian actor--critic".
4. → **"Action"** (teal outline) — sub-label "surge / sway / heave / yaw,
   $[-1,1]^4$".
5. → back into box 1, closing the loop.

Below box 1, a separate amber box **"Reward"** — sub-label "bounded signed
deltas; crossing bonus; collision charged per entry" — with an arrow up into
"PPO policy". Attach a small amber note to it: **"training only; never part of
the observation"**.

**LOWER BAND — the curriculum, drawn as a left-to-right ladder** of four rungs,
each a wide flat box with its length mixture shown as a small stacked bar:

* **"$S_0$ stabilization"** — mixture bar 50 / 35 / 15 / 0
* **"$S_1$ chaining"** — 30 / 35 / 25 / 10
* **"$S_2$ balanced"** — 20 / 30 / 30 / 20
* **"$S_3$ long sequence"** — 10 / 20 / 35 / 35

Between consecutive rungs, an upward arrow labelled **"promotion on unseen
validation seeds"**. Above the ladder, a legend for the four bucket colours:
"focus 2 gates", "short 3–5", "medium 6–12", "long 13–24".

To the right of $S_3$, a final box with a double outline: **"Freeze + hash"** →
**"Pre-registered readiness gate"** → **"Evaluate"**. Attach a small note to the
readiness gate: **"19 criteria registered before evaluation"**.

Below the ladder, a thin grey strip: **"procedural course family; the three
official circuits are never used for training"**.

### Forbidden

* No arrow from the reward box into the observation encoder or into the
  observation box.
* No arrow from the official circuits strip into any training rung.
* Do not draw an expert or rule controller inside the rollout loop.

### Caption to use

> The learning pipeline. A policy is trained through the same arena, adapter and
> referee used for evaluation, consuming the same official observation and
> emitting the same four-channel command. The reward may inspect privileged
> state during training but is never part of the observation. The curriculum
> controls the mixture of episode lengths and promotes on measured
> generalization over unseen validation seeds rather than on elapsed
> transitions; the three official circuits are never used for training.

### Insertion point

`sections/09_learning.tex`, immediately after the
`\subsection{Training Pipeline and Curriculum}` heading and before the
"Algorithm and architecture" paragraph. Add
`\input{figures/schema4_learning}` there and give the figure the label
`fig:learning_pipeline`, then cite it from that paragraph.

---

## SCHEMA 5 — Fleet architecture, local autonomy and team scoring

* **Filename:** `schema5_fleet.pdf`
* **New figure**, to be inserted in `sections/08_fleet.tex`
* **Aspect ratio:** 16:9, landscape, full text width
* **Note:** this schema also stands in for the fleet photograph that could not
  be captured (see `FIGURE_SOURCE_AUDIT.md`, row E). No claim depends on a
  photograph.

### Scientific message

Three vehicles run one course as one team. Each keeps its own autonomy and its
own referee state; they exchange only their own estimates over a lossy acoustic
link; the team score is a conjunction, so no vehicle can be left behind.

### Exact layout

Three horizontal regions.

**LEFT — the convoy.** Three BlueROV2-style vehicles in a diagonal line,
labelled **V1 (leader)**, **V2**, **V3**, moving toward a small gate at the
upper right. Draw a dashed curved path through two gate frames. Above V1 write
**"Continuous Servo"**; above V2 and V3 write **"Center-then-Commit"**. Between
consecutive vehicles draw a double-headed teal arc with a small waveform glyph,
labelled **"acoustic link: range-dependent delay, packet loss"**.

**CENTRE — per-vehicle stacks.** Three identical vertical stacks, one per
vehicle, each in teal:

* "own onboard sensing"
* "own LocalCourseTracker"
* "own base controller"
* "leader–follower wrapper"

Attach to each wrapper a small callout showing the only three fields
transmitted: **"local\_beacon\_index"**, **"local\_lap"**, **"local\_status"**.
Between the V2 and V3 stacks and their predecessor, draw a short teal arrow
labelled **"yield while $\hat g_{\mathrm{pred}} - \hat g < \Delta g$"**.

**RIGHT — the referee side, amber.** A single panel headed **"Independent
referee"** containing three small identical row-blocks labelled "V1 state", "V2
state", "V3 state", each with the fields "gates", "penalties", "status". Below
them, separated by a rule, a wider block headed **"team\_summary"** containing:

* "team gates $= \sum_i G_i$"
* "finished $= \bigwedge_i$ finished$_i$"
* "team time $= \max_i T_i^{\text{finish}} - \min_i T_i^{\text{release}}$"
* "penalized $= T_{\text{team}} + \sum_i P_i + P_{\text{ivc}}$"

Beside the referee panel, a small amber badge reading **"inter-vehicle proximity
detected here, never exposed to controllers"**.

**Bottom strip**, `#F2F5F9`: **"per-vehicle rows are diagnostics; only
team\_summary is the official result"**.

### Arrows

Teal arrows from each vehicle up into its own stack; teal arcs between adjacent
vehicles for the acoustic link; amber arrows from an implicit simulator edge
into each referee row-block; amber arrows from the three row-blocks down into
`team_summary`.

### Forbidden

* No arrow from the referee panel to any vehicle or any per-vehicle stack.
* No arrow between two per-vehicle stacks except through the acoustic link arc.
* Do not draw a shared world model, shared map or shared position table on the
  controller side.

### Caption to use

> Fleet architecture. Three vehicles run one circuit as a single team. Each keeps
> its own onboard sensing, its own local course estimate and its own controller;
> coordination uses only a static predecessor assignment and the three
> self-reported fields exchanged over an acoustic-inspired channel with
> range-dependent delay and packet loss. The referee maintains one scoring state
> per vehicle and aggregates them into the team summary, which is the only
> official fleet result; inter-vehicle proximity is detected on the referee side
> and is never exposed to a controller.

### Insertion point

`sections/08_fleet.tex`, immediately after the
`\subsection{Distributed Leader--Follower Coordination}` heading. Add
`\input{figures/schema5_fleet}` and label it `fig:fleet_architecture`, then cite
it from the "Module design" paragraph of that subsection.

---

## Optional, low priority

**Deferred HoloOcean captures.** Two images would be nice to have and support no
claim; both require launching the simulator, which was not possible while the
manuscript was prepared because a PPO training campaign held every available
engine instance (see `FIGURE_SOURCE_AUDIT.md`).

1. A dedicated in-aperture frame during a Center-then-Commit passage, to
   replace panel (c) of Fig. 5 with a purpose-shot image.
2. A photo-real render of three BlueROV2 vehicles in convoy on Horseshoe Bay, to
   accompany Schema 5.

Capture only when no training is running, with `--adapter holoocean --headless`,
and record the seed, track, controller and commit alongside the image.
