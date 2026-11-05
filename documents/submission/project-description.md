# Agentic Realtime Computer Vision

**AI gives the camera a brain. The camera gives AI eyes.**

Final report: https://github.com/AuraAIInc/agentic-realtime-cv/blob/main/documents/submission/agentic-realtime-cv-report.md

Team: Muyang Lu, Terry Li · Meta Global AI Developer Hackathon 2026 · Built on Meta Muse Spark

## In one sentence

A $100 Raspberry Pi camera watches our two cats every second of the day with a tiny detector, recognizes when it is unsure and says why, sends only those moments (about 1 in 100 frames) to Meta's Muse Spark agent, writes each verified answer back into its own record, and builds a plain-language memory of what is normal for each cat, in each place, at each hour.

## The problem

We live with two cats, Qb and Qbo. On a workday we see them for two or three hours, and the rest of their day is invisible. We want answers to simple questions: when did each cat eat, when did he sleep and for how long, when did he play, and how does today compare with his normal week? These questions matter because cats are very good at hiding illness. Veterinary guidance treats changes in routine as an early warning sign, and owners' memories of routine are often wrong when compared with video.

A camera should be able to answer these questions. In practice, computer vision today is split into two halves that fail in opposite ways.

- **A small detector on cheap hardware sees every frame but understands little.** A Raspberry Pi 5 CPU can only run a nano-sized YOLO model, at about three frames per second. That model is almost free to run all day, but it makes mistakes. It confuses our two near-identical white cats, and it misses a cat asleep in its bed. Once, it decided that an exercise ball was Qbo and tracked it as a cat for 26 minutes, with complete confidence and almost no movement.
- **A large vision AI model understands a frame but cannot afford to see them all.** A home camera at two frames per second produces 172,800 frames a day. We measured what it costs to have AI detect and identify every frame of an 8-minute window: over an hour of processing and $24.65. That comes to roughly $1.2 million a year per camera, and the AI still missed the sleeping cat.

Bigger models alone will not close this gap. The gap comes from scale, not from intelligence: no improvement in model quality makes it affordable to send every frame of every camera to a large model. Training a much larger detector is not an option on this hardware either, because it needs a large labeled dataset, GPU training and compute that a small home device does not have.

So the real problem is this: **how do you let a cheap, imperfect model watch everything, and still trust what it records?**

## Our scope

We deliberately took the small model as a given. This project is a high-performance workflow for compute-limited environments: local deployment on a Raspberry Pi, and private deployment where video should not leave the home. We do not try to replace the small detector with a bigger one. Instead, we make its output trustworthy by adding intelligence only where it is needed. This is the problem every compute-limited and privacy-bound deployment actually faces.

## The core idea: spend intelligence only where doubt exists

Most of a cat's day is easy to watch. Cats spend most of their time lying or sitting still, and only a small share of moments are genuinely hard to interpret. If the cheap model can recognize the hard moments itself and explain what is hard about them, an expensive model only needs to see those few moments.

Three things have to flow between the two halves, and this project builds all three:

- **Attention flows up.** The cheap model finds the moments that deserve expensive thought, and says why.
- **Knowledge flows down.** The agent's verdicts correct the record now, and become labeled data for improving the small model later.
- **Memory spans both.** A plain-language history makes "unusual" mean unusual for this cat, in this room, at this hour.

## What we built

The system is a loop with five parts. Here is how one real doubt from our demo travels through it.

**1. The camera watches every frame.** A YOLO nano detector runs on the Raspberry Pi all day. Beside that always-on loop, the camera exposes a small server of tools, so an AI agent can capture a sharper frame, re-run detection on a region or change the frame rate when it needs to look harder. Every override expires within 24 hours, and with no requests the camera behaves exactly as before.

**2. The tracker names its own doubt.** Two correctors turn raw boxes into tracks and then into visits. The hackathon contribution is that both correctors now report their uncertainty at the exact decision point where it arises. There are fifteen typed doubt codes, plus a routine spot-check. Examples:

- "I relabeled this cat based on distance alone, with no box overlap."
- "This box jumped more than 300 pixels, so I split the visit."
- "I filled in a long run of guessed frames with no real detection."
- "The top two cats scored almost the same for this visit."
- "This visit may continue the same cat's previous visit."
- "The tracker's frame labels disagree with the visit's overall vote."

A low-confidence box does not trigger a review by itself. Doubts fire on conflicts in the tracker's own decisions. This is why only about 1% of frames escalate, and why each question arrives already explained. In our demo, one doubt fired at 16:35:21: the tracker had relabeled the cat in the bed from Qb to Qbo, only because Qbo's last box was 92 pixels away.

**3. The doubt becomes a plain-English question.** Raw tracker numbers mean nothing to a language model, so each doubt code has its own question builder. The question states the conflict, what to validate, the alternatives, the exact frames and boxes to inspect, and both cats' written appearance profiles. It describes the small detector honestly as "a hint, not evidence", and attaches a short memory brief marked "may help, may be stale". Each code also has a strict answer schema: one verdict from a fixed list, at most one qualifier, and a short rationale. Related doubts that occur close together are merged into one agent call, so the agent can reason across cases and fetch each frame only once. In the demo, 12 doubts were reviewed in just 5 calls.

**4. The Muse Spark agent investigates and answers.** The supervisor agent runs on Meta's Muse Spark 1.3. It does not receive a pile of images. It decides what evidence it needs and fetches it with tools:

- tools that ask the camera to look now, sharper or with another model;
- tools that fetch stored frames, cropped to a box;
- tools that read the tracker's structured history before any image is needed;
- local tools for each cat's written profile, owner reference photos and contact sheets.

The agent also loads short "skills" on demand. These are playbooks for specific tasks, such as an identity check that climbs a cheapest-evidence-first ladder: metadata, then the crop against the written profiles, then a sharper view only if needed, then owner photos, then camera reference crops, and finally a human, rather than forcing a guess. A verdict that cites no visible cue is not allowed. For the bed-cat doubt, the agent pulled four frames, zoomed into each cat, and answered: the bed cat has "cool white fur, a smooth sharp outline and no fluffy halo, so it is Qb", and the nearness had "caused a cross-steal; the detector was right."

**5. Verdicts rewrite the record, and memory learns.** Each verdict is applied to the tracked rows, and every changed row records which doubt changed it and why. Any correction can be traced back to the agent's full trace, including its tool calls, the images it saw, its tokens, cost and time. The verified visit then enters memory.

## Memory that knows what normal looks like

Detectors see frames and trackers see paths, but neither knows that Qb usually naps in the green bed in the afternoon. Most AI agent frameworks treat memory as conversation history. Video needs memory organized by where and when, and protected from noise. We built two memories with different jobs.

- **Short-term memory** is a readable day log. It keeps everything from today: every visit, weak or strong, every doubt, every verdict and its rationale. A real entry reads: "16:34–16:40: Qb slept curled up tight in the small green bed, paws tucked in. Sleeping, confidence 0.92, 772 sightings." At the end of the day, it is merged into long-term memory.
- **Long-term memory** keeps only trusted knowledge, from visits where both the detector and the activity label reach 0.7 confidence. It is organized as two pyramids. In space, 48 small cells roll up into 12 zones and then one whole-view summary. In time, 5-minute slots roll up into hours, days and weeks. Five-minute slots are kept for a week, hours for a month, days for six months, and weekly habits indefinitely.

Muse Spark writes this memory in plain language. A vision call labels each visit with an activity such as sleeping, walking, eating or playing, along with the cat's appearance and the background. A second call checks whether a new observation genuinely contradicts what memory already knows. A third compresses children into a parent summary under a word cap. Code, not the model, enforces the rules that keep memory honest:

- a pattern needs at least three observations before it becomes stable;
- one-off observations wait in a small buffer;
- a stable fact flips only after three consecutive contradictions;
- word caps force each summary to keep only its strongest patterns.

When a new doubt arrives, memory supplies a short brief tailored to the doubt type. For an identity doubt, it says whether this cat is hard to see here. For a visit doubt, it says which cat is usually here at this hour. Memory informs the agent, but it never overrides the image.

## Results, measured against ground truth

We measured the system against hand-validated ground truth on an 8-minute window with both cats on camera (1,319 frames), at standard detection thresholds.

- **False alarms fell by 93%**, from 107 to 8.
- **Precision rose from 0.895 to 0.992.**
- **Identity accuracy rose from 0.934 to 0.993.** This is the share of boxes in the right place that also carry the right cat's name.
- **99% of what the camera saw ended up with the right name**, compared with 93% from the detector alone.
- **The agent was called on about 1% of frames**: 12 doubts in 8 minutes, reviewed in 5 calls.
- **The review cost $0.39**, about 63 times cheaper than having AI label every frame of the same window ($24.65).

The agent, not the tracker's rules, does the work. The tracker's rules alone left overall accuracy flat and slightly lowered identity accuracy. The agent's 12 verdicts produced the gain.

The gain is largest exactly where the tracker raised doubts. In the hardest two minutes, when the tracker had filed the sleeping Qb under the wrong name, precision rose from 0.53 to 0.97 and false alarms fell from 65 to 4. Where the detector was already right and no doubt was raised, the review changed nothing. That is the design working: intelligence is spent where doubt exists, and nowhere else.

Across three replayed windows (65 minutes of footage, 10,970 frames), doubts stayed at about 1% of frames, and the total review cost was $3.77, against about $205 to have AI label every frame.

We also report what the system cannot do. Recall stays limited, because the small detector boxes only about 40% of the cats in the ground truth at all, and a reviewer can only relabel boxes that exist. Of what the camera did see, 99% ends up correctly named. Improving recall is a detector problem, outside this project's scope, and it is exactly what our roadmap's "teach the Pi" step addresses.

## Why Muse Spark makes this possible

The loop only works if the model at the other end of a doubt can do five things at once:

- tell two look-alike cats apart from pixels, and quote the cues it sees;
- decide which evidence to fetch instead of being sent everything, using multi-turn tool calling with images returned inside tool results;
- return an operation that code can apply to rows, not an essay, using strict structured output, including answer schemas built at runtime for merged calls;
- cost cents per doubt, helped by cached input pricing (52% of all input tokens were cached across the build);
- write memory that a person can read and that stays stable.

Muse Spark 1.3 does all five. Across the build, it ran 124 agent sessions and 1,058 tool calls, and every session returned an answer. The rest of the Muse family covered the stages around the loop:

- **Muse SAM 3.1** produced the segmentation-based ground truth that makes every number above checkable.
- **Muse Wonder** shaped the design: the doubt taxonomy, the verdict contracts and the memory architecture.
- **Muse Glimmer** is the path to running locally and privately on the device.
- **Muse Code** was our development environment for the pipeline, the agent, the memory system, the tests and the documentation.

We priced everything at Standard rates, because the cheaper Contributor tier grants permission to train on the footage, and customers' home cameras are not ours to give away.

## Beyond cats

Cats are where we proved the idea, not where it ends. Any small model watching for rare moments has the same gap between cheap, constant attention and expensive understanding:

- **Look-alike lab mice**, whose identities swap when they huddle, as Qb and Qbo's did.
- **Tiny, fast objects** such as drones, birds or balls, where a few pixels can be a decoy.
- **Vehicles in fog or at night**, where tracks break or swap and the guessed parts must be checked.
- **Production lines**, where a small model checks every item on site and only the odd few need a second opinion.
- **Pet cameras, robots and wildlife monitoring**, all running on limited hardware for long periods.

## What's next

- **Give your own agent these eyes.** Run Muse Glimmer locally, so a personal AI agent can ask the camera questions and hear when a routine changes, without footage leaving the home.
- **Teach the Pi.** Turn the agent's verdicts into labeled data and retrain the same small detector, so fewer doubts reach the agent over time.
- **Every camera, all year.** Measure full days and more camera views, then apply the same loop to labs, factory lines and roads.

## Links

- Final report, with architecture, real prompts, tool and skill design, memory examples, full results, costs and limits: https://github.com/AuraAIInc/agentic-realtime-cv/blob/main/documents/submission/agentic-realtime-cv-report.md
- How Muse is used across the project: https://github.com/AuraAIInc/agentic-realtime-cv/blob/main/documents/muse-usage.md
- Code: https://github.com/AuraAIInc/agentic-realtime-cv
