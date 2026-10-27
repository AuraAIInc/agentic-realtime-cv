"""WebSocket client that streams detection JSON to the Aura server.

Same behaviour as v2: a bounded outbound queue, a sender thread, and
reconnect-on-failure. Messages from the server are logged.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import websocket

logger = logging.getLogger(__name__)


class WebSocketSender:
    def __init__(self, url: str, q_max: int, on_server_message=None):
        self.url = url
        self.q_max = q_max
        self.on_server_message = on_server_message
        self._queue: deque[str] = deque()
        self._qlock = threading.Lock()
        self._app: websocket.WebSocketApp | None = None
        self._app_lock = threading.Lock()
        self._ready = threading.Event()
        self._reconnecting = threading.Event()
        self.sent = 0

    # ---- public ----
    def start(self) -> "WebSocketSender":
        websocket.enableTrace(False)
        self._start_app("ws_run_forever")
        threading.Thread(target=self._send_loop, name="ws_send", daemon=True).start()
        return self

    def enqueue(self, message: str) -> None:
        with self._qlock:
            self._queue.append(message)
            while len(self._queue) > self.q_max:
                self._queue.popleft()

    def qsize(self) -> int:
        return len(self._queue)

    @property
    def connected(self) -> bool:
        return self._ready.is_set()

    # ---- callbacks ----
    def _on_open(self, ws):
        logger.info("WebSocket connection opened.")
        self._ready.set()
        self._reconnecting.clear()

    def _on_message(self, ws, message):
        logger.info(f"Received from server: {message}")
        if self.on_server_message:
            try:
                self.on_server_message(message)
            except Exception:
                logger.exception("on_server_message handler failed")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")
        time.sleep(1)

    def _on_close(self, ws, code, msg):
        logger.warning(f"WebSocket closed: {code} - {msg}")
        self._ready.clear()

    # ---- internals ----
    def _start_app(self, thread_name: str) -> None:
        with self._app_lock:
            self._app = websocket.WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            app = self._app
        threading.Thread(target=app.run_forever, kwargs={"ping_interval": 30}, name=thread_name, daemon=True).start()

    def _reconnect(self) -> None:
        if self._reconnecting.is_set():
            return
        self._reconnecting.set()
        while True:
            try:
                probe = websocket.WebSocket()
                probe.connect(self.url, timeout=5)
                probe.close()
                break
            except Exception as e:
                logger.error(f"Test connection failed: {e}")
                time.sleep(1)
        logger.info("Attempting to reconnect...")
        with self._app_lock:
            old = self._app
        try:
            old.close()
        except Exception:
            pass
        self._ready.clear()
        self._start_app("ws_reconnect_run_forever")
        self._ready.wait(30)
        self._reconnecting.clear()

    def _send_loop(self) -> None:
        self._ready.wait()
        while True:
            message = None
            with self._qlock:
                if self._queue:
                    message = self._queue.popleft()
            if message is None:
                time.sleep(0.1)
                continue
            try:
                with self._app_lock:
                    app = self._app
                app.send(message)
                self.sent += 1
            except Exception as e:
                logger.error(f"Failed to send to server: {e}")
                with self._qlock:
                    if len(self._queue) < self.q_max:
                        self._queue.appendleft(message)
                self._reconnect()
                time.sleep(1)
