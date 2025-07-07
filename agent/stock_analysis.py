from crewai.flow.flow import Flow, start, listen
from crewai import LLM, Agent
from typing import Callable, Any
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
from litellm import completion
from dotenv import load_dotenv
from crewai.tools import tool
import yfinance as yf
import json
import pandas as pd
import os

load_dotenv()


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

        tickers_list = json.loads(tickers)["tickers"]
        tikers = [yf.Ticker(ticker) for ticker in tickers_list]
        results = []
        for ticker_obj, symbol in zip(tikers, tickers_list):
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
        tickers_list = json.loads(tickers)["tickers"]
        tikers = [yf.Ticker(ticker) for ticker in tickers_list]
        results = []
        for ticker_obj, symbol in zip(tikers, tickers_list):
            info = ticker_obj.info
            company_name = info.get("longName", "N/A")
            # Get annual financials (income statement)
            financials = ticker_obj.financials
            # financials is a DataFrame with columns as years (ending date)
            # Revenue is usually under 'Total Revenue' or 'TotalRevenue'
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


model = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key=os.getenv("GOOGLE_API_KEY"),
)
print(os.getenv("GOOGLE_API_KEY"),"api key")
stock_agent = Agent(
    role="Stock Analyst",
    backstory="You are a stock analyst who uses the tools to get the stock price and revenue data of the given tickers",
    goal="Use the tools to get the stock price and revenue data of the given tickers",
    llm=model,
    tools=[get_stock_price_tool, get_revenue_data_tool],
    verbose=True,
)


class StockAnalysisFlow(Flow):
    def __init__(
        self, messages: list[str], handlers: list[Callable[[str], Any]], **kwargs
    ):
        super().__init__(**kwargs)
        # Option A: keep them as attributes
        self.messages = messages
        self.handlers = handlers
        # Option B: put them into unstructured state
        self.state["messages"] = messages
        self.state["handlers"] = handlers
        self.agent = stock_agent

    @start()
    def chat(self):

        msgs = getattr(self, "messages", self.state["messages"])
        hnds = getattr(self, "handlers", self.state["handlers"])
        messages = []
        for msg in msgs:
            if msg.name is None:
                msg.name = ""
            messages.append(msg.dict())
        

        agent_result = self.agent.kickoff(messages=messages)
        return "done"

    @listen(chat)
    def stock_analysis(self, _):
        return f"Dispatched {len(self.state['messages'])} messages"


# -> "Dispatched 2 messages"
