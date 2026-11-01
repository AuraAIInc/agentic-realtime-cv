# Model API chaining probe: 400s on image tool-output turns

Date: 2026-10-31. Endpoint: Meta Model API (`https://api.meta.ai/v1`),
model `muse-spark-1.3`, via the OpenAI SDK's Responses interface
(`client.responses.create`), as `agent/supervisor_agent/agent.py` uses it.

## Symptom

An interactive agent query needing two tool rounds crashed the CLI:

```
openai.BadRequestError: Error code: 400 - referenced response not found or expired
```

The failing call was round 3 of one `MCPAgent.run`: rounds 1-2
(`camera__get_status`, `tracking__get_live_status`, then
`camera__capture_frame`) succeeded; the follow-up request whose input was the
`capture_frame` result (a `function_call_output` carrying `input_text` + a
base64 `input_image`) plus `previous_response_id` was rejected. Text-only
multi-round chains had always worked, so the first suspect was
image-bearing chained input.

## Probe method

Throwaway scripts (not committed) drove `responses.create` directly with a
single dummy `echo` function tool, then with 34 dummy tools, walking this
shape while varying one factor at a time:

1. chain root: `input=<text>`, optional `instructions`, optional `tools`
2. tool-call turn: `input=<text>`, `previous_response_id=<R1>`
3. tool-output turn: `input=[function_call_output(...)]`,
   `previous_response_id=<R2>` — text-only or text+image

Images were real retrieved frames (512px/15KB and 800x600/67KB base64 JPEGs,
`detail: "auto"`), the same shape `tool_adapter.to_function_output` builds.

## Results

Image-output chained turns, grouped by where `instructions` was sent:

| Root turn | Later turns | Image turn | Outcome |
|---|---|---|---|
| no instructions | no instructions | chained | OK (4/4) |
| no instructions | no instructions | chained + instructions | OK |
| no instructions | instructions on call turn | chained, no instructions | OK |
| instructions | no instructions | chained, no instructions | FAIL (5/5 at the time) |
| instructions | instructions | chained, no instructions | FAIL |
| instructions (+calls, drained) | no instructions | chained, no instructions | OK (3/3, later runs) |

Failure messages observed (both 400, same request family):

- `No tool output found for function call call_...` — even though the output
  for that call is present in the request.
- `referenced response not found or expired` — the original CLI crash.

## Failing example

Minimal chain that 400s intermittently (6/10 runs with root `instructions`;
0/7 without). Request 3 is byte-identical in shape to what the agent sends
after `camera__capture_frame`:

```python
# 1. chain root WITH instructions -> OK (id=resp_6ae6...)
client.responses.create(model="muse-spark-1.3",
    instructions="You must always call the echo tool; never answer directly.",
    input="Reply with exactly: root",
    tools=[{"type": "function", "name": "echo", ...}],
    max_output_tokens=2048)

# 2. chained tool-call turn, no instructions -> OK (function_call call_0d62...)
client.responses.create(model="muse-spark-1.3",
    input="Call echo with text 'a'.",
    tools=[...], previous_response_id="resp_6ae6...", max_output_tokens=2048)

# 3. chained image tool-output turn, no instructions -> 400
client.responses.create(model="muse-spark-1.3",
    input=[{"type": "function_call_output", "call_id": "call_0d62...",
            "output": [{"type": "input_text", "text": '{"tool":"echo"}'},
                       {"type": "input_image",
                        "image_url": "data:image/jpeg;base64,...",  # 15-67 KB
                        "detail": "auto"}]}],
    tools=[...], previous_response_id="resp_9f31...", max_output_tokens=2048)
# >>> openai.BadRequestError: Error code: 400 -
#     No tool output found for function call call_0d62...
```

Drop `instructions` from request 1 (keeping everything else identical) and
request 3 passes. Text-only output in request 3 also passes with
`instructions` present — the failure needs the image part plus the poisoned
root.

A final replica of the crash shape (34 tools, `instructions` everywhere,
2 calls in R1, text outputs, another call, then the image output) **passed**,
and the model described the image.

## Conclusion

Root-turn `instructions` is *necessary but not sufficient* for the failure:
all 6 failures had it, all 7 runs without it passed, but 4 runs with it also
passed. There is a second, hidden variable — most likely gateway-side
flakiness (e.g. response-store replication lag when the image turn re-reads
the chain), since identical shapes passed and failed on different runs. The
two error strings are presumed to be the same server-side failure surfacing
at different layers.

## Fix applied

`MCPAgent.run` no longer sends the `instructions` parameter at all. The
instruction text is prepended to the turn-1 input instead, where it persists
via chaining like any other context. `tools` is still sent every turn
(proven harmless in every probe). Rationale at the time: every observed
failure carried root-turn `instructions`; zero failures occurred without it.

## End-to-end retest (fix NOT sufficient)

Three live runs of "check the camera view to see if there is any cat"
against camera_sim + tracking MCP, with the fix active (unit-pinned: no
request carries `instructions`):

- run 1: crashed with the ORIGINAL error (`referenced response not found or
  expired`) on the first image round (5 tool calls done, incl. 1 capture).
- run 2: crashed with the same error deeper in (10 tool calls done).
- run 3: full pass (4 turns, 6 calls, images described, answered).

So root-turn `instructions` is **not** necessary either — the probe
correlation does not transfer to the live flow. The live requests differ
from the probes in payload size (~45k input tokens: 34 large tool schemas
plus long instructions-as-text), chain content, and per-turn latency (tens
of seconds of model reasoning per round). Suspects now: gateway-side
flakiness around large chained image turns (e.g. store replication lag or a
short response TTL surfacing as "not found or expired"), possibly
size-dependent. Investigation continues; the `instructions` fold stays in
place (harmless, still the best-supported mitigation) but is not the fix.

## Side findings (same endpoint)

- Only `tool_choice: "auto"` is accepted; `"required"`/`"none"`/named
  choices return 400. The agent never sets it — no change needed.
- Bare `input_text`/`input_image` items at the top level of `input` are
  rejected (`` `input[0]` did not match any supported type``). Images must
  ride inside `function_call_output.output` (works) — which is the only
  shape the agent sends.
- The model emits a large `reasoning` block first: tiny
  `max_output_tokens` values return reasoning-only responses with no message
  or function call. The agent sets no token cap.
- Chained requests MUST answer every pending function call of the referenced
  response, otherwise the same `No tool output found` 400 is (correctly)
  returned. The agent always answers all calls each round.
