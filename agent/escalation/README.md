# escalation

Hard scenes go to a large model with tools; its verified labels become training data
(Project Plan, Part 3: Escalation and continuous learning).

Already exists: the escalation tasks in `../supervisor_agent/tasks.py` (identity check,
trajectory physics check, activity labels; rules first, then the model; image budgets;
`send_to_human`).

New here: the **trigger** that decides automatically from the live stream what counts as a
hard scene (gap length, jump size, identity flips, low confidence, unseen objects), and the
path from escalation labels to the retraining dataset.
