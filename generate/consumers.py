"""
    Websocket consumers for the generate app

    This file contains the consumers for the generate app. The consumers are
    responsible for handling the websocket connections and sending messages
    regarding the generation of images.
"""
import json

from channels.generic.websocket import AsyncWebsocketConsumer


class GenerateConsumer(AsyncWebsocketConsumer):
    """Consumer for the generation of images"""

    async def connect(self) -> None:
        """Called when the websocket is handshaking as part of initial connection."""
        await self.accept()

    async def disconnect(self, close_code: int) -> None:
        """Called when the websocket closes for any reason.

        Args:
            close_code (int): The code indicating why the connection closed
        """
        pass

    async def receive(self, text_data: str) -> None:
        """Called when we get a text frame. Channels will JSON-decode the payload for
        us and pass it as a dict to this method.

        Args:
          text_data (str): The data sent over the websocket
        """
        text_data_json = json.loads(text_data)
        message = text_data_json["message"]

        await self.send(text_data=json.dumps({"message": message}))
