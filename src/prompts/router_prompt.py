from langchain_core.prompts import ChatPromptTemplate


ROUTER_SYSTEM_PROMPT="""
Classify user financial query.

Return ONLY one:

summary
chart
comparison
stock_price

Rules:

chart:
mentions page, graph, table, figure

comparison:
compare, versus, vs

stock_price:
price, ticker

summary:
all other financial questions

Use conversation history if needed.
"""


ROUTER_PROMPT=ChatPromptTemplate.from_messages([

("system",ROUTER_SYSTEM_PROMPT),

("placeholder","{messages}")

])