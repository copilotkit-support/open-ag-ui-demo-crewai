from crewai.flow.flow import Flow, start, listen
from crewai import LLM, Agent
from typing import Callable, Any

# from ag_ui.core import (
# )
from ag_ui.core.types import AssistantMessage, ToolMessage
from litellm import completion
from dotenv import load_dotenv
from crewai.tools import tool, BaseTool

# from crewai.tools.base_tool import BaseTool
import yfinance as yf
import json
import pandas as pd
import os
import uuid
from pydantic import BaseModel

load_dotenv()


class chart_data(BaseModel):
    x: str | int
    y: str | int


@tool
def get_stock_price_tool(tickers: list[str]) -> str:
    """Get the stock price of a given list of tickers"""
    try:
        # config.get("configurable").get("tool_logs")["items"].append({
        #     "toolName": "GET_STOCK_PRICE",
        #     "status": "inProgress"
        # })
        # config.get("configurable").get("emit_event")(
        #     StateDeltaEvent(
        #         type=EventType.STATE_DELTA,
        #         delta=[
        #             {
        #                 "op": "add",
        #                 "path": "/items/-",
        #                 "value": {
        #                     "toolName": "GET_STOCK_PRICE",
        #                     "status": "inProgress"
        #                 }
        #             }
        #         ]
        #     )
        # )
        # await asyncio.sleep(2)

        # tickers_list = json.loads(tickers)["tickers"]
        tikers = [yf.Ticker(ticker) for ticker in tickers]
        results = []
        for ticker_obj, symbol in zip(tikers, tickers):
            hist = ticker_obj.history(period="1d")
            info = ticker_obj.info
            if not hist.empty:
                price = hist["Close"].iloc[0]
            else:
                price = None
            company_name = info.get("longName", "N/A")
            revenue = info.get("totalRevenue", "N/A")
            results.append(
                {
                    "ticker": symbol,
                    "price": price,
                    "company_name": company_name,
                    "revenue": revenue,
                }
            )
        # index = len(config.get("configurable").get("tool_logs")["items"]) - 1
        # config.get("configurable").get("emit_event")(
        #     StateDeltaEvent(
        #         type=EventType.STATE_DELTA,
        #         delta=[
        #             {
        #                 "op": "replace",
        #                 "path": f"/items/{index}/status",
        #                 "value": "completed"
        #             }
        #         ]
        #     )
        # )
        # await asyncio.sleep(0)
        return {"results": results}
    except Exception as e:
        print(e)
        return f"Error: {e}"


@tool
def get_revenue_data_tool(tickers: list[str]) -> str:
    """Get the revenue data of a given list of tickers"""
    try:
        # config.get("configurable").get("tool_logs")["items"].append({
        #     "toolName": "GET_REVENUE_DATA",
        #     "status": "inProgress"
        # })
        # config.get("configurable").get("emit_event")(
        #     StateDeltaEvent(
        #         type=EventType.STATE_DELTA,
        #         delta=[
        #             {
        #                 "op": "add",
        #                 "path": "/items/-",
        #                 "value": {
        #                     "toolName": "GET_REVENUE_DATA",
        #                     "status": "inProgress"
        #                 }
        #             }
        #         ]
        #     )
        # )
        # await asyncio.sleep(2)
        # tickers_list = json.loads(tickers)["tickers"]
        tikers = [yf.Ticker(ticker) for ticker in tickers]
        results = []
        for ticker_obj, symbol in zip(tikers, tickers):
            info = ticker_obj.info
            company_name = info.get("longName", "N/A")
            # Get annual financials (income statement)
            financials = ticker_obj.financials
            # financials is a DataFrame with columns as years (ending date)
            # Revenue is usually under "Total Revenue" or "TotalRevenue"
            revenue_row = None
            for key in ["Total Revenue", "TotalRevenue"]:
                if key in financials.index:
                    revenue_row = financials.loc[key]
                    break
            if revenue_row is not None:
                # Get the last 5 years (or less if not available)
                revenue_dict = {
                    str(year.year): (
                        int(revenue_row[year])
                        if not pd.isna(revenue_row[year])
                        else None
                    )
                    for year in revenue_row.index[:5]
                }
            else:
                revenue_dict = {}
            results.append(
                {
                    "ticker": symbol,
                    "company_name": company_name,
                    "revenue_by_year": revenue_dict,
                }
            )
        # index = len(config.get("configurable").get("tool_logs")["items"]) - 1
        # config.get("configurable").get("emit_event")(
        #     StateDeltaEvent(
        #         type=EventType.STATE_DELTA,
        #         delta=[
        #             {
        #                 "op": "replace",
        #                 "path": f"/items/{index}/status",
        #                 "value": "completed"
        #             }
        #         ]
        #     )
        # )
        # await asyncio.sleep(0)
        return json.dumps({"results": results})
    except Exception as e:
        print(e)
        return f"Error: {e}"


@tool
def render_bar_chart(topic: str, data: list[chart_data]) -> str:
    """Render a bar chart with the given data. The data would be very generic"""
    return json.dumps({"data": data, "topic": topic})


# class render_bar_chart(BaseTool):
#     name : str = "render_bar_chart"
#     description : str = "Render a bar chart with the given data. The data would be very generic"
#     result_as_answer : bool = True
#     def _run(self, topic: str, data: list[chart_data]) -> str:
#         return json.dumps({"data": data,"topic": topic})

model = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key=os.getenv("GOOGLE_API_KEY"),
    additional_kwargs={"response_format": "json"},
)
print(os.getenv("GOOGLE_API_KEY"), "api key")
# stock_agent = Agent(
#     role="Stock Analyst",
#     backstory="You are a stock analyst who uses the tools to get the stock price and revenue data of the given tickers",
#     goal="Use the tools to get the stock price and revenue data of the given tickers",
#     llm=model,
#     tools=[get_stock_price_tool, get_revenue_data_tool],
#     verbose=True,
# )


def convert_tool_call(tool_call: dict):
    return {
        "id": str(uuid.uuid4()),
        "type": "function",
        "function": {
            "name": tool_call["tool_name"],
            "arguments": str(tool_call["tool_arguments"]),
        },
    }


class StockAnalysisFlow(Flow):
    def __init__(
        self,
        messages: list[str],
        handlers: list[Callable[[str], Any]],
        frontend_tools: list[Callable[[str], Any]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Option A: keep them as attributes
        self.messages = messages
        self.handlers = handlers
        self.tools = frontend_tools
        # Option B: put them into unstructured state
        self.state["messages"] = messages
        self.state["handlers"] = handlers

    @start()
    def chat(self):

        msgs = getattr(self, "messages", self.state["messages"])
        hnds = getattr(self, "handlers", self.state["handlers"])
        messages = []
        for msg in msgs:
            if msg.role == "assistant":
                if(msg.tool_calls is not None):
                    if len(msg.tool_calls) > 0:
                        msg.id = str(uuid.uuid4())
                        msg.tool_calls = [tool_call.dict() for tool_call in msg.tool_calls]
                    
            if hasattr(msg, "name") and msg.name is None:
                msg.name = ""
            if hasattr(msg, "content") and msg.content is None:
                msg.content = ""
            # if(msg.role == 'tool'):
            #     del msg.id

            messages.append(msg.dict())

        chat_agent = Agent(
            role="Chat Agent",
            backstory="You are an amazing assistant who can answer any questions which is posed by the user. You will be given a list of messages and you will have to answer the question based on the messages context. The messages will be in the form of a conversation between the user and the AI assistant.",
            goal='When you provide an answer, you should strictly provide it in a json format like this : {"answer": "This is the answer from the assistant", "isStockAnalyse": true }. The isStockAnalyse is a boolean value which indicates whether the question is related to stock analysis or not. If the answer is related to stock analysis, then the isStockAnalyse should be true, otherwise it should be false.',
            llm=model,
            verbose=True,
        )
        # chat_agent_result = chat_agent.kickoff(messages=messages)
        # chat = model.call(messages=messages)
        stock_agent = Agent(
            role="Stock Analyst",
            backstory="You are a stock analyst who uses the tools to get the stock price and revenue data of the given tickers",
            goal="""
            Your response MUST be a single, valid JSON object with the following format:

                {
                "tool_calls": [
                    {
                    "tool_name": "render_bar_chart",
                    "tool_arguments": {
                        "topic": "Amazon Revenue (Last 4 Years)",
                        "data": [{"x": "2021", "y": 469822000000}, ...]
                    }
                    }
                ],
                "data": "Explanation or commentary here"
                }

                Any deviation from this format will result in a system error.
                Do not use markdown. Do not add extra text. Just return valid JSON.
            """,
            llm=model,
            tools=[get_stock_price_tool, get_revenue_data_tool, render_bar_chart],
            verbose=True,
        )
        if messages[-1]['role'] == "tool":
            try:
                response = completion(
                    model="gemini/gemini-2.0-flash",  # Note the format: "gemini/<model-name>"
                    messages=messages,
                    api_key=os.getenv("GOOGLE_API_KEY"),
                )
                print(response)
                self.messages.append(
                    AssistantMessage(
                        role="assistant",
                        content=response.choices[0].message.content,
                        id=str(uuid.uuid4()),
                    )
                )
            except Exception as e:
                print(e)
        else:  
            agent_result = stock_agent.kickoff(messages=messages)

            if agent_result.raw.startswith("```json"):
                agent_result.raw = agent_result.raw.replace("```json", "").replace(
                    "```", ""
                )
                agent_result.raw = json.loads(agent_result.raw)
                print(agent_result.raw)
                # return agent_result
            else:
                agent_result.raw = json.loads(agent_result.raw)
            if len(agent_result.raw["tool_calls"]) > 0:
                tool_calls = [
                    convert_tool_call(tool_call)
                    for tool_call in agent_result.raw["tool_calls"]
                ]
                self.messages.append(
                    AssistantMessage(
                        role="assistant",
                        content="",
                        tool_calls=tool_calls,
                        id=str(uuid.uuid4()),
                    )
                )
            print(agent_result.raw["tool_calls"])
        return "done"

    @listen(chat)
    def stock_analysis(self, _):
        return self.messages


# -> "Dispatched 2 messages"
