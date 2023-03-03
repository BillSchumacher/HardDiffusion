"""
    Websocket consumers for the generate app

    This file contains the consumers for the generate app. The consumers are
    responsible for handling the websocket connections and sending messages
    regarding the generation of images.
"""
import json
from typing import Any, Dict

from channels.generic.websocket import AsyncWebsocketConsumer


class GenerateConsumer(AsyncWebsocketConsumer):
    """Consumer for the generation of images"""

    async def connect(self) -> None:
        """Called when the websocket is handshaking as part of initial connection."""
        self.room_group_name = "generate"
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()
        await self.send(
            text_data=json.dumps({"message": "Connected to HardDiffusion!"})
        )

    async def disconnect(self, close_code: int) -> None:
        """Called when the websocket closes for any reason.

        Args:
            close_code (int): The code indicating why the connection closed
        """
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    # Receive message from WebSocket
    async def receive(self, text_data: str) -> None:
        """Called when we get a text frame. Channels will JSON-decode the payload for
        us and pass it as a dict to this method.

        Args:
          text_data (str): The data sent over the websocket
        """
        print("td", text_data)
        text_data_json = json.loads(text_data)
        message = text_data_json["message"]
        print("message", message)
        # Send message to room group
        await self.channel_layer.group_send(
            "damap", {"type": "event_message", "message": message}
        )

    # Receive message from room group
    async def event_message(self, event):
        """Called when we get a message from the room group."""
        print(event)
        message = event["message"]
        # Send message to WebSocket
        await self.send(
            text_data=json.dumps({"message": message, "event": event["event"]})
        )
