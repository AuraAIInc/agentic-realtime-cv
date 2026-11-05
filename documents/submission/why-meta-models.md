# Why Meta's Models Are Essential to Our Solution

Final report: https://github.com/AuraAIInc/agentic-realtime-cv/blob/main/documents/submission/agentic-realtime-cv-report.md

Team: Muyang Lu, Terry Li · Meta Global AI Developer Hackathon 2026

## The short answer

Our system lets a cheap camera watch everything and still be trusted. It works by spending intelligence only where doubt exists. The small detector on the Raspberry Pi handles about 99% of frames on its own and raises a typed doubt on the rest. Those doubts need somewhere to go: a model that can look at the right pixels, decide what evidence it needs, answer in a form code can apply, and do it for cents, all day.

Meta's Muse Spark 1.3 is that model. Without it, the doubts have nowhere to go. The cheap model stays cheap and wrong, or every frame goes to AI and the system stops being cheap. Meta's wider model family supplies the parts around that loop: Muse SAM 3.1 for the ground truth that proves the gain, Muse Wonder for the design, and Muse Glimmer for keeping footage on the device.

## The problem these models solve

Computer vision today is split into two halves that fail in opposite ways.

- **The small detector is cheap but makes mistakes.** A Raspberry Pi 5 CPU runs only YOLO nano, at about three frames per second. It confuses our two near-identical white cats, misses a cat asleep in its bed, and once tracked an exercise ball as a cat for 26 minutes.
- **A large vision model understands, but cannot see everything.** Labeling every frame of an 8-minute window with AI took over an hour and $24.65. That is roughly $1.2 million a year per camera, and it still missed the sleeping cat.

A bigger detector is not the answer in this setting, because it needs GPU training and hardware a home device does not have. The answer is to connect the two halves, and that connection depends on what the model at the other end can do.

## Five things the solution needs from a model

**1. Seeing the difference between look-alikes, and saying why.**

Qb is a cool-white short-hair with a smooth outline and white paws. Qbo is a cream long-hair with a golden-brown face mask, brown paws and a dark plume tail. At 800×600 resolution from a wide-angle ceiling camera, the small detector often cannot tell them apart. The tracker's own rules make it slightly worse: on their own, they lowered identity accuracy from 0.934 to 0.920.

Muse Spark's vision reasoning compares a crop against each cat's written profile and names the cues it actually sees. In one real case it wrote: "cool white, smooth sharp outline, no fluffy halo, compact body, round head = Qb; nearness caused a cross-steal; the detector was right." Because a verdict must cite visible cues, every correction can be checked by a person. After review, identity accuracy reached 0.993.

**2. Deciding which evidence to look at.**

Sending every frame is what makes AI unaffordable. Sending one fixed frame is often the wrong frame. Our agent needs to choose: read the tracker's history first, fetch a few full frames, then crop into the disputed boxes, and stop as soon as the evidence settles the case.

Muse Spark does this through multi-turn tool calling, with images returned inside tool results, so it sees only what it asked for. In the demo, one call fetched four frames, then four crops, then answered four related doubts at once. A typical review takes four turns and eight tool calls. This is the difference between "send everything" and "look where it matters".

**3. Answering with an operation, not an essay.**

A free-text answer cannot safely rewrite a record. Each doubt in our system has a strict answer contract: one verdict from a fixed list, at most one qualifier such as which cat it really is, and a short rationale. When related doubts are merged, the contract is built at runtime from the union of their allowed verdicts.

Muse Spark's strict structured output lets every answer be applied directly to the tracked rows, with each changed row recording which doubt changed it. In the demo run, 12 of 12 verdicts were parsed and applied. One verdict renamed a whole visit of 163 rows; another kept two cats' visits correctly apart.

**4. Costing cents, so it can run all day.**

The system only makes sense if a doubt costs little. Muse Spark's everyday pricing, combined with cached input for repeated context such as system prompts and cat profiles, keeps reviews cheap: 52% of all input tokens across the build were served from cache. The demo window's 12 doubts cost $0.39 to review, about 63 times cheaper than having AI label every frame. Across three replayed windows (65 minutes of footage), review cost $3.77 in total, against about $205 to label every frame.

**5. Writing memory a person can read.**

"Unusual" only means something against "normal". Our long-term memory stores, in plain language, what each cat usually does in each spot and at each hour, organized in space (cells, zones, the whole view) and in time (5-minute slots, hours, days, weeks). Muse Spark writes that memory:

- it labels each trusted visit's activity from key frames ("sleeping, confidence 0.92, curled up tight in the small green bed");
- it judges whether a new observation genuinely contradicts what memory already holds;
- it compresses children into parents under word caps, keeping only the strongest patterns.

The result is memory a person can read and audit, and a short brief that rides along with the next doubt as context.

## The rest of the family completes the solution

**Muse SAM 3.1 makes the results trustworthy.** Claims of improvement need ground truth. SAM 3.1's segmentation-based boxes, with Muse Spark naming each cat, labeled every frame of the demo window. That pipeline scored F1 0.794 against 0.612 for a single vision model, with full coverage. Every number in our results is checkable against that ground truth. The same labels can retrain the small detector, so the edge model improves over time.

**Muse Wonder made the design right before any code existed.** The doubt taxonomy, the verdict contracts and the memory architecture were argued out with the deepest-reasoning tier first, where a wrong design would cost the most.

**Muse Glimmer keeps the solution private and local.** The project targets local and private deployment, where video should not leave the home. Muse Glimmer is the on-device path, and it is the basis of our next step: giving a person's own AI agent these "eyes" without sending footage to the cloud.

**One family, matched to each job.** Using one model family for design, ground truth, the running agent and the edge keeps prompts, schemas, pricing and behavior consistent across the project. We did not stitch together unrelated systems. Each Meta model does the job it is best at.

## What the solution achieves because of these models

Measured against hand-validated ground truth on an 8-minute window with both cats on camera:

- false alarms fell by **93%** (107 to 8);
- precision rose from **0.895 to 0.992**;
- identity accuracy rose from **0.934 to 0.993**;
- **99%** of what the camera saw ended up with the right cat's name;
- the AI was called on only **about 1% of frames**, 12 doubts reviewed in 5 calls;
- the review cost **$0.39**, about **63 times cheaper** than AI on every frame.

In the hardest two minutes, where the tracker had filed the sleeping cat under the wrong name, precision rose from 0.53 to 0.97, and false alarms fell from 65 to 4.

None of these gains come from the small detector or the tracker's rules. They come from the Muse Spark verdicts.

## Responsibility and privacy

Using Meta's models responsibly shaped several decisions:

- **Standard pricing only.** The Contributor tier would have been cheaper, but it grants permission to train on footage, and customers' home cameras are not ours to give away.
- **Only doubts leave the device.** About 1% of frames raise a doubt, and only then is the agent called.
- **Memory informs, never overrides.** The agent is told that memory "may help, may be stale", and the image always decides.
- **Every decision is traceable.** Each agent run keeps a full trace of its tool calls, the images it saw, its tokens, cost and time.

## Why this matters beyond our cats

The same gap, between cheap constant attention and expensive understanding, appears wherever a small model watches for rare moments on limited hardware: look-alike lab mice, tiny drones and birds, vehicles in fog, production lines and home robots. Meta's models make the bridge between those two halves practical. Muse Spark answers the hard moments with evidence and contracts. Muse SAM 3.1 proves the gain. Muse Glimmer keeps it local. Cats are where we proved it, not where it ends.
