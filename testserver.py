import asyncio

import websockets


async def test_websocket():
    uri = "ws://localhost:8000/app"
    # uri="wss://echo.websocket.events"
    try:
        async with websockets.connect(uri) as websocket:
            # Initial message
            await websocket.send("You are a frenchman in love with Lyon. You will talk with an extremely french accent. Go!")

            # Continuously receive messages
            while True:
                try:
                    # Wait for a message with a timeout
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    print(response, end="", flush=True)
                    # print("Received:", response)

                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed by the server.")
                    break

    except websockets.exceptions.ConnectionRefusedError:
        print("Could not connect to the WebSocket server. Is the server running?")

    except Exception as e:
        print(f"An error occurred: {e}")



asyncio.run(test_websocket())
