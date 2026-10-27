# memory

What the agent can pull when answering or deciding (Project Plan, Part 2: Memory).

Already exists elsewhere: short-term tracker state (`tracking_mcp`), long-term MySQL periods and
stats, stored images (image server), and static knowledge in `../agent_data/` (cat profiles,
reference photos, labels, zones).

New here: **summarized text memory**, written by the large model and retrievable by time,
cat or text. Open: summary interval (per period, hourly, daily), storage and retrieval.
