import asyncio
import logging

from starlette.websockets import WebSocket

from ragged.models import model_manager


async def app(scope, receive, send):
    async def generate(llm_output):
        for token in llm_output:
            text = token["choices"][0]["text"]
            yield text

    await model_manager.initialize()

    websocket = WebSocket(scope=scope, receive=receive, send=send)
    await websocket.accept()
    try:
        message = await websocket.receive_text()
        print(f"Received message: {message}")
        llm = await model_manager.llm
        output = llm.create_completion(
            message,  # Prompt
            max_tokens=256,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
            echo=False,  # Echo the prompt back in the output
            stream=True,
            temperature=1.0,
        )
        # print(output)
        async for token in generate(output):
            await websocket.send_text(token)
            await asyncio.sleep(0)

    except Exception as e:
        print(f"Error in WebSocket connection: {e}")
    finally:
        await websocket.close()
