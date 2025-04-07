import asyncio
import logging

from starlette.websockets import WebSocket, WebSocketDisconnect

from ragged.models import model_manager


async def app(scope, receive, send):
    async def generate(llm_output):
        for token in llm_output:
            text = token["choices"][0]["text"]
            yield text

    # await model_manager.initialize()
    await model_manager._init_llm()
    llm = await model_manager.llm

    websocket = WebSocket(scope=scope, receive=receive, send=send)
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            print(f"Received message: {message}")
            output = llm.create_completion(
                message,  # Prompt
                max_tokens=256,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
                echo=False,  # Echo the prompt back in the output
                stream=True,
                temperature=0.3,
            )
            # print(output)
            async for token in generate(output):
                await websocket.send_text(token)
                await asyncio.sleep(0)

            # Send an end-of-response marker
            await websocket.send_text("[END]")

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        print(f"Error in WebSocket connection: {e}")
    finally:
        await websocket.close()
