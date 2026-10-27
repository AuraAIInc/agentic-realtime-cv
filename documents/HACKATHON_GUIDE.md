# Meta Global AI Developer Hackathon 2026: Team Guide

Everything we know about the event, plus what it means for our project (the Aura cat
agent in this folder). Sources: the event page, the Official Rules, and the Participant
Guide on cerebralvalley.ai (as of Oct 26, 2026).

- Event page: <https://cerebralvalley.ai/e/meta-ai-developer-hackathon>
- Submit: <https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit>
- Meta page: <https://dev.meta.ai/events/global-hackathon>

---

## Action items (do these first)

| # | What | Owner | By |
| --- | --- | --- | --- |
| 1 | Track decided: **Indie Developer**, team of 2 (Terry, Muyang). Name the Representative who submits. Project plan: [PROJECT_PLAN.md](PROJECT_PLAN.md). | | Oct 27 |
| 2 | **Swap the agent's LLM from OpenAI to Meta models.** `mcp_agent` runs on `gpt-6.1-sol` today. A project that does not use Meta models fails Stage 1 (pass/fail). | | Oct 29 |
| 3 | Create a **separate public repo** for the submission. `aura_api_server` is private and must stay private. Copy only the hackathon code, with no secrets, internal IPs (e.g. `10.0.0.63` in `mcp_agent/README.md`), or proprietary model weights. | | Oct 30 |
| 4 | Tag the pre-hackathon state (e.g. `git tag hackathon-baseline`) and write down what existed before Oct 27. The tracker, correctors, `tracking_mcp` and `mcp_agent` all predate the event (commits Oct 23-25). | | Oct 27 |
| 5 | Every team member claims their own **$150 Meta Model API credits** (form in the Participant Guide). Needs a credit card on the account; set a Spend Limit on the Billing page. | each | ASAP, hard stop Nov 4 9 PM ET |
| 6 | Request partner credits early (they are capped): Baseten (2,000 people), Modal (first 1,000), Fireworks (per team, shortlist). | | Oct 27 |
| 7 | Join the Discord, watch the kickoff, check `#announcements` daily. | all | Oct 26 |

---

## Key dates

All official times are ET. We are on Pacific time, so subtract 3 hours (this stays true
after DST ends on Nov 1).

| Date | ET | PT | What |
| --- | --- | --- | --- |
| Mon Oct 26 | 12:00 PM | 9:00 AM | Kickoff livestream (rules, prizes, judging, Meta speakers) |
| Tue Oct 27 | 12:00-1:00 PM | 9:00-10:00 AM | Agents 101: Build a Multimodal AI Agent (Suraj Subramanian) |
| Wed Oct 28 | 12:00-1:00 PM | 9:00-10:00 AM | Agents 102: Best Practices for Verifiable Agents (Yash Khare) |
| Fri Oct 30 | 12:00-1:00 PM | 9:00-10:00 AM | Fireworks: Inference to Training with Muse Open Models (Sinan Ozdemir) |
| Weekdays Oct 27 - Nov 4 | 10:00-11:00 AM | 7:00-8:00 AM | Meta office hours, text only, Discord `#office-hours` |
| Tue Nov 3 | | | Last full day for submission questions and troubleshooting |
| **Wed Nov 4** | **9:00 PM** | **6:00 PM** | **Submission deadline. Late submissions are not considered.** Also the last day to claim API credits. |
| Thu Nov 5 - Sun Nov 8 | | | Judging |
| Sun Nov 8 | EOD | | Cutoff for X engagement (Most Viral prize) |
| Mon Nov 9 | 12:00 PM | 9:00 AM | Closing ceremony: top 5 per track announced, live final round, winners, community vote opens |
| Mon Nov 16 | | | Community vote winner announced on Discord |

Building period is **Oct 27 - Nov 4**.

---

## Rules that matter for us

- **Must use Meta models.** Allowed: Muse Spark series, Muse Wonder, Muse Image, Muse Voice
  Transcribe and SAM 3.1 via the Meta Model API; open-weight Muse Glimmer and Muse Spark
  Open; the Muse Code coding agent.
- **Repo must be public.** See action item 3.
- **New work only.** "What you present at this Hackathon must have been built during the
  event." Startups may build on an existing product if they clearly identify what is new;
  that allowance is written for startups, and **we are in the Indie track**, so it applies
  to us more strictly (question for the organizers, see PROJECT_PLAN.md §7). The demo must only highlight features built during the
  hackathon. Failing to make this clear means **immediate disqualification** at the
  judges' discretion. Plan: a "Built during the hackathon" section in the README, a
  baseline tag, a commit history that starts at the baseline, and a demo video that says
  out loud what is new.
- **Team size:** max 4 (solo allowed). One submission per team, entered by the
  Representative, who lists every team member.
- **One track per project.**
- **Age 18+**; sanctions/country exclusions apply; no Meta or Cerebral Valley employees.
- **Banned:** anything violating legal, ethical or platform policies, or using code, data or
  assets we don't have rights to.

### Anti-projects (generic or banned)

The organizers list these as things to avoid: AI medical advice (mental health advisor,
skin blemish detector...), basic RAG apps, **simple image analyzers**, simple education
chatbots, job application screeners, nutrition coaches, personality analyzers, sports
analyzers/coaches, anything political or war-related.

What this means for Aura:

- We must **not look like a "simple image analyzer"**. The pitch has to be the agent: it
  reasons over live camera, stored images and tracker state, takes actions (capture,
  high-res, schedule), and answers real questions about the cats over time.
- **Stay away from health or diagnosis claims** ("your cat looks sick", "eating
  disorder"). That reads as AI medical advice. Frame it as behavior and activity
  observation and alerting, with a human deciding.

---

## Judging

**Stage 1 (pass/fail):** does the project reasonably fit a problem statement, and did it
reasonably use the Meta Model API and/or Meta open-weight models to build it during the
event?

**Stage 2 (each criterion scored 0-10, weighted):**

| Criterion | Weight | What judges ask |
| --- | --- | --- |
| Impact | 30% | Real-world potential, who benefits, could people actually use it, fits a problem statement |
| Use of Meta AI tooling | 25% | Creative use, beyond a basic integration, capabilities "that surprised even us" |
| Demo | 25% | Working and impressive, holds up live, the video is "genuinely cool to watch" |
| Technical depth & execution | 20% | Sound, refined engineering, real craft |

**Stage 3:** top 5 per track go to a live final round; judges watch the demo videos live
and rank them on the same rubric.

Judges are not named. Expect a large first-pass screen (likely thousands of entries), so
the **video and README must make the point within the first 30 seconds**.

### Problem statements (pick one, say which)

- **Build for Your World:** bring Meta's models into your domain, solve a problem you know
  firsthand, free up time and mental space, make something possible that wasn't before.
  *Our natural fit: a multi-cat household, knowing who did what and when without watching
  footage.*
- **Push the Frontier:** software, robotics, manufacturing, energy, materials; give a machine
  a new capability, accelerate a scientific experiment, understand and control a physical
  system.
  *Possible angle: an agent that perceives and acts on a physical sensing system (camera on a
  Pi, schedule control).*

---

## Submission checklist

Deadline **Nov 4, 9:00 PM ET / 6:00 PM PT**. Aim to submit by Nov 4 noon PT.

- [ ] Public GitHub repo, accessible logged-out, with a README covering the problem, the
      architecture, which Meta models are used and where, how to run it, and **what was
      built during the hackathon vs. before**
- [ ] **3-minute demo video**, link publicly accessible. It shows only what we built
      during the event. It is a primary judging artifact along with the code and the live
      product.
- [ ] Live product / demo link that holds up (judges may try it)
- [ ] All team members listed on the submission page
- [ ] Track and problem statement chosen
- [ ] X post link submitted (if going for Most Viral)
- [ ] No secrets, keys, internal hosts or customer data in the repo or video

---

## Prizes

Prizes are cash in USD and are paid only to the top 5 in each track.

| Place | Indie ($400K) | Startup ($550K) |
| --- | --- | --- |
| 1st | $150K | $250K |
| 2nd | $100K | $150K |
| 3rd | $75K | $75K |
| 4th | $50K | $50K |
| 5th | $25K | $25K |

**Bonus prizes ($50K total, separate from track prizes):**

| Prize | Amount | How | Fit for us |
| --- | --- | --- | --- |
| Most Viral Project | $20K | Highest engagement (likes + reposts + comments) on an X post of the demo, until Nov 8 EOD; submit the post link | **Strong.** Cat videos travel well. Post early on Nov 4/5. |
| Best Build on Meta Open Weight Models | $10K | Built on Muse Glimmer or Muse Spark Open, on our own hardware or partner inference; partner verifies | Possible if we serve Muse Spark Open (Baseten/Modal) for part of the pipeline |
| Community Vote | $10K | Most votes in the public gallery, one per verified account | Mobilize our users/community |
| Best Agentic Use-Case | $5K | Autonomy, task effectiveness, practical usefulness | **Strong.** This is what `mcp_agent` is. |
| Best Creative Build | $5K | Most original use of Muse Image or Muse Wonder | Optional (e.g. a generated daily "cat diary" comic) |

Payout: through Cerebral Valley via Tremendous, within 30 days of verification; W-9 or
W-8BEN required; winners pay their own taxes and transfer fees. A startup prize goes to the
startup via the Representative.

---

## Resources and credits

**Meta Model API ($150 per person)**
- Each participant claims individually via the credit claim form linked in the
  Participant Guide; credits arrive in batches, so check the Billing page.
- A credit card is required; a temporary pre-authorization hold is released in a few days.
- Set a Spend Limit notification so we don't go over $150.
- Docs: <https://dev.meta.ai/docs/> (Overview, Quickstart, Model list, Cookbook, API
  fundamentals, Muse Code recipes, OSS model cookbook)

**Baseten ($100 + 2M TPM, up to 2,000 people, per individual, reviewed by use case)**
- Use model id **`meta-models/muse-spark-open-1026`** for the hackathon (the docs say
  `meta-models/muse-spark-open`, which is not the hackathon one).
- Docs cover chat completions, reasoning output, and vision + audio input for Muse Spark
  Open.

**Modal ($100, first 1,000 participants):** Endpoints for Meta open-weight models, plus
serverless GPUs, sandboxes and training tools.

**Fireworks ($25 per team, shortlist):** redemption details are emailed if selected.

Partner credits are not guaranteed.

---

## Legal / IP notes

- We keep ownership, but submitting grants Meta, Cerebral Valley and partners a
  **perpetual, irrevocable, royalty-free license** to use, modify and publicize the
  submission (marketing, live demos, etc.).
- **No confidentiality.** Anything shared during the event (Discord, office hours, the
  submission) can be used by the organizers. Keep proprietary pieces (training data,
  weights, the private backend) out of the public repo and the video.
- Sessions and our screen shares or chat messages can be recorded and reused.
- Open-source components are fine if their licenses are followed and our work builds on
  them.

---

## Open questions to raise in `#questions` / office hours

- We are in the Indie track. Can we build on our own pre-existing code (tracker, MCP
  servers, predating Oct 27) if we mark it as existing infrastructure and present only
  the new work?
- We have a registered company but no product users. Is Indie the right track?
- Can the live product depend on hardware (camera on a Pi) that judges can't access, as
  long as there is a hosted demo or recorded data replay?
- Does using Meta models for the agent but our own YOLO for detection count as reasonable
  use for Stage 1? (Most likely yes; confirm.)
