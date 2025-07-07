from fastapi import FastAPI
from fastapi.responses import StreamingResponse  # For streaming responses
from pydantic import BaseModel
import uuid
from typing import Dict, List, Any, Optional
import os
import uvicorn
import asyncio
from ag_ui.core import (
    RunAgentInput,
    StateDeltaEvent,
    Message,
    StateSnapshotEvent,
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    TextMessageStartEvent,
    TextMessageEndEvent,
    TextMessageContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolCallArgsEvent,
)
import ast
from ag_ui.encoder import EventEncoder
from copilotkit import CopilotKitState
from datetime import datetime
from stock_analysis import StockAnalysisFlow
import json

app = FastAPI()


class AgentState(CopilotKitState):
    """
    This is the state of the agent.
    It is a subclass of the MessagesState class from langgraph.
    """

    tools: list
    messages: list


@app.post("/crewai-agent")
async def crewai_agent(input_data: RunAgentInput):

    async def event_generator():
        encoder = EventEncoder()
        # query = input_data.messages[-1].content
        message_id = str(uuid.uuid4())  # Generate a unique ID for this message
        event_queue = asyncio.Queue()

        def emit_event(event):
            event_queue.put_nowait(event)

        yield encoder.encode(
            RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=input_data.thread_id,
                run_id=input_data.run_id,
            )
        )

        flow = StockAnalysisFlow(
            messages=input_data.messages,
            handlers=[emit_event],
            frontend_tools=input_data.tools,
        )
        agent_output = await flow.kickoff_async()
        print(json.dumps(agent_output[-1].tool_calls[0].function.arguments),"agent output")
        if agent_output[-1].role == "assistant":
            if len(agent_output[-1].tool_calls) > 0:
                py_dict = ast.literal_eval(agent_output[-1].tool_calls[0].function.arguments)  # Safely convert string to dict
                json_args = json.dumps(py_dict)
                yield encoder.encode(
                    ToolCallStartEvent(
                        type=EventType.TOOL_CALL_START,
                        tool_call_id=agent_output[-1].tool_calls[0].id,
                        tool_call_name=agent_output[-1].tool_calls[0].function.name,
                    )
                )

                yield encoder.encode(
                    ToolCallArgsEvent(
                        type=EventType.TOOL_CALL_ARGS,
                        tool_call_id=agent_output[-1].tool_calls[0].id,
                        delta=json_args,
                    )
                )

                yield encoder.encode(
                    ToolCallEndEvent(
                        type=EventType.TOOL_CALL_END,
                        tool_call_id=agent_output[-1].tool_calls[0].id,
                    )
                )

        yield encoder.encode(
            RunFinishedEvent(
                type=EventType.RUN_FINISHED,
                thread_id=input_data.thread_id,
                run_id=input_data.run_id,
            )
        )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )


if __name__ == "__main__":
    main()
