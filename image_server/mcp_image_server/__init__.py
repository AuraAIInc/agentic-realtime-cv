"""MCP server for the Raspberry Pi image SSD.

Standalone: it does not import `src/` (which opens MySQL and S3 connections on
import), so it starts instantly and only needs read access to the image folder.

Run with:  python -m mcp_image_server
"""
