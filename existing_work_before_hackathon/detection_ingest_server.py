import asyncio
import websockets
import json
import datetime
from typing import Dict
from model.track_corrector import TrackCorrector
from model.period_corrector import PeriodCorrector
from model.db import MysqlDB
from model.my_logging import logger
from tracking_mcp.history import History
from tracking_mcp.live_state import LiveState

track_corrector_pool:Dict[int, TrackCorrector] = {}
period_corrector_pool:Dict[int, PeriodCorrector] = {}
db = MysqlDB()
# Read-only view of the pools for the tracking MCP server (tracking_mcp)
live_state = LiveState(track_corrector_pool, period_corrector_pool)

track_buffer_size = 1

def process_image(detections):

    # {"detections": [{"cat_id": "0", "box": [90, 335, 145, 366], "confidence": 0.6832}],
    #  "is_grey": 0,
    #  "view_id": "0",
    #  "device_id": "0sfdsfgsdfgsdf",
    #  "create_time": "2024-09-25-00-08-21-619643"}

    min_confidence = 0.5

    create_time = detections["create_time"]

    img_detected = {int(d["cat_id"]): d["box"] + [d["cat_id"], d["confidence"]] for d in detections["detections"] if d["confidence"] > min_confidence}
    logger.debug(f"track_corrector.process_images input: {img_detected}")
    if detections["view_id"] not in track_corrector_pool:
        track_corrector_pool[detections["view_id"]] = TrackCorrector(view_id=detections["view_id"])
        track_corrector_pool[detections["view_id"]].set_bulk_insert_size(track_buffer_size)
    track_corrector = track_corrector_pool[detections["view_id"]]
    if detections["view_id"] not in period_corrector_pool:
        period_corrector_pool[detections["view_id"]] = PeriodCorrector(view_id=detections["view_id"])
    period_corrector = period_corrector_pool[detections["view_id"]]
    for is_append, revised_detection in track_corrector.process_images(
        create_time,
        img_detected,
        model_id=detections["model_id"],
        is_grey=detections["is_grey"],
        view_id=detections["view_id"],
        device_id=detections["device_id"],
        save_db=True,
        plot_progress=False,
    ):
        if is_append:
            for corrected_period in period_corrector.process_detection(revised_detection, view_id=detections["view_id"]):
                logger.info(f"Detected period: {corrected_period}")
                live_state.record_period(detections["view_id"], corrected_period)
        else:
            period_corrector.unprocess_detection(revised_detection)


async def handle_websocket(websocket, path=None):  # websockets >= 14 passes only the connection
    message_queue = asyncio.Queue()

    async def receive_messages():
        try:
            while True:
                message = await websocket.recv()
                await message_queue.put(message)
        except websockets.ConnectionClosed:
            logger.info(f"Connection closed: {websocket.close_code} - {websocket.close_reason}")

    async def process_messages(min_confidence=0.5):
        last_received_time = datetime.datetime.now()
        while True:
            message = await message_queue.get()
            detection = json.loads(message)
            view_id = detection["view_id"]
            live_state.record_frame(detection, datetime.datetime.now(), min_confidence)
            if view_id not in track_corrector_pool:
                track_corrector_pool[view_id] = TrackCorrector(view_id=view_id)
                track_corrector_pool[view_id].set_bulk_insert_size(track_buffer_size)
            track_corrector = track_corrector_pool[view_id]
            if view_id not in period_corrector_pool:
                period_corrector_pool[view_id] = PeriodCorrector(view_id=view_id)
            period_corrector = period_corrector_pool[view_id]
            received_time = datetime.datetime.now()
            if received_time.date() != last_received_time.date():
                logger.info("New day, reset track_corrector and period_corrector")
                track_corrector.finish_up()
                for _ in period_corrector.finish_up():
                    pass

            img_detected = dict(
                [
                    (int(d["cat_id"]), d["box"] + [int(d["cat_id"]), d["confidence"]])
                    for d in detection["detections"]
                    if d["confidence"] > min_confidence
                ]
            )
            create_time = datetime.datetime.strptime(detection["create_time"], "%Y-%m-%d-%H-%M-%S-%f")
            detection["create_time"] = create_time

            process_image(detection)

            finish_time = datetime.datetime.now()
            tracking_time = round((finish_time - received_time).total_seconds(), 4)
            process_time = round((finish_time - create_time).total_seconds(), 4)
            logger.info(
                f"Processed image: {detection['create_time']} \tTracking time: {tracking_time} \tProcess time: {process_time} "
                + f"\tDetected: {[k for k in img_detected]} "
                + f"\tBuffer sizes: {[len(b) for b in track_corrector.buffer]} {[len(b) for b in track_corrector.buffer_origin]}"
            )

            last_received_time = received_time

    receive_task = asyncio.create_task(receive_messages())
    process_task = asyncio.create_task(process_messages())

    await asyncio.gather(receive_task, process_task)


async def main():
    try:
        from tracking_mcp.server import start_in_loop

        mcp_task = start_in_loop(live_state, History(db))
    except ImportError as e:  # mcp / uvicorn not installed: run without the MCP server
        logger.warning(f"Tracking MCP server not started: {e}")
    async with websockets.serve(handle_websocket, "0.0.0.0", 8000):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
