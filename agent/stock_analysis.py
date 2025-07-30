
from crewai.flow.flow import Flow, start, router, listen
from litellm import completion
from pydantic import BaseModel
from typing import Literal, List


class StockAnalysisFlow(Flow):
    @start()
    def start(self):
        print(self)
        return self.state

    @listen("start")
    def chat(self, state):
        return self.state
    
    @listen("start")
    def simulation(self, state):
        return self.state
    
    @listen("start")
    def allocation(self, state):
        return self.state
    
    @listen("start")
    def insights(self, state):
        return self.state

