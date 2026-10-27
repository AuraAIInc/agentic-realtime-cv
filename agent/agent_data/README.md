# agent_data

Data for the RCAT task agent (`supervisor_agent/tasks.py`). Edit these files by hand; no code changes needed.

| Path | What | Who writes it |
|---|---|---|
| `config/cats.yaml` | One profile per cat: tracker ids per view (Qb = 0 / 25, Qbo = 2 / 26), summary, low-res, camera and high-res cues, look-alikes | you |
| `config/labels.yaml` | Activity labels (exactly one per crop) and context labels (any number, can be zone rules) | you, or the agent after you confirm (`--labels`) |
| `config/zones.yaml` | Named places, occluders and entries per view, in view pixels | you, or the agent after you confirm |
| `config/agent.yaml` | Per-task skill, tool list, image budget; trajectory thresholds; reference weights | you |
| `references/<cat>/profile/` | Owner photos; `*_crop.jpg` is the cat only (<= 512 px) and is what the model sees | you |
| `references/<cat>/scene/` | Live-camera frames (view 3, 800x600) + `*_crop.jpg` (<= 384 px) | you / confirmed results |
| `references/<cat>/extra/` | Kept, never used for matching (e.g. the painted portrait) | you |
| `references/<cat>/index.json` | One record per image: source, quality, pose, lighting, view, box, trust; ranked without opening images | script / you |
| `logs/<task>.jsonl`, `logs/review_queue.jsonl` | Every task result; anything sent to a human | agent (git-ignored) |
| `label_suggestions/<name>/` | Crops the agent could not label, with its suggested name, for you to review | agent (git-ignored) |

Scene frames are detector-labelled (`confirmed_by: detector`); `agent_check` is a visual
check, not a human confirmation. Set `use_for_matching: false` in index.json to drop an image.
