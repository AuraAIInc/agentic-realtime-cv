# existing_work_before_hackathon

Code that existed before the hackathon started (Oct 27, 2026), kept separate so it is clear
what was built during the event. It is the production ingest path on the server; the
hackathon pipeline in [`agent/`](../agent/) carries its own MySQL-free copy of the
correctors (`agent/tracking/`) and escalation vocabulary (`agent/escalation/`).

| Path | What it is | Still used by |
|---|---|---|
| [`detection_ingest_server.py`](detection_ingest_server.py) | WebSocket ingest on `:8000`: runs both correctors per camera message and writes `revised_detection` / `period` rows to MySQL; starts the tracking MCP server | the live deployment |
| [`model/track_corrector.py`](model/track_corrector.py), [`model/period_corrector.py`](model/period_corrector.py) | Original TrackCorrector / PeriodCorrector (MySQL-backed), with escalation hooks ([`model/README.md`](model/README.md)) | ingest server |
| [`model/db.py`](model/db.py) | `MysqlDB` connection helper (`MYSQL_*` in `.env`) | ingest server, `agent/tracking/db_stream.py` (`--replay-db`), `image_server/fetch_hour_images.py` |
| [`model/visualization.py`](model/visualization.py) | Hourly detection chart | `image_server/fetch_hour_images.py` |
| [`tracking_mcp/`](tracking_mcp/README.md) | MCP server (`:10002`) over the live tracker state and the MySQL history | supervisor agent (`tracking__*` tools) |
| [`run_tracking_viz.py`](run_tracking_viz.py) | Frame-by-frame corrector visualization from MySQL (read-only) | manual debugging |

```bash
pip install -r existing_work_before_hackathon/requirements.txt
cd existing_work_before_hackathon && python detection_ingest_server.py   # or ./run_ingest_server.sh
python -m pytest model/test_escalation.py tracking_mcp -q                  # no MySQL needed
```

Pre-hackathon helpers that nothing in this repo used (S3/SES/EC2 utilities, the daily
statistics job, the email template) were removed from this repo.
