from agent_ai_copywriter.agentstate import AgentState, ValidityCheckState
from agent_ai_copywriter.nodes import keyword_fetcher_node, copywriter_node, formatting_node, publishing_node, refiner_node, routing_node
from langgraph.graph import StateGraph, START, END
import os
import json

os.makedirs("articoli", exist_ok = True)

# Initializing the state graph for the agent workflow.
workflow = StateGraph(AgentState)

# Adding nodes to the workflow.
workflow.add_node("keyword_fetcher", keyword_fetcher_node)
workflow.add_node("copywriter", copywriter_node)
workflow.add_node("refiner", refiner_node)
workflow.add_node("formatting", formatting_node)
workflow.add_node("publishing", publishing_node)
# workflow.add_node("routing", routing_node)

# Defining the edges to connect the nodes in the workflow.
workflow.add_edge(START, "keyword_fetcher")
workflow.add_edge("keyword_fetcher", "copywriter")
workflow.add_edge("copywriter", "refiner")
workflow.add_edge("refiner", "formatting")
workflow.add_conditional_edges(
    "formatting",
    routing_node,
    {
        "publish": "publishing",
        "end": END
    }
)

workflow.add_edge("publishing", END)
app_seo = workflow.compile()

# === ESECUZIONE DELL'AGENTE ===
if __name__ == "__main__":
    topic = input("Inserisci il topic dell'articolo da scrivere: ")
    input_iniziale = {"topic": topic}
    risultato_finale = app_seo.invoke(input_iniziale)
    filename_json = f"articoli/{topic.replace(' ', '_')}.json"
    with open(filename_json, 'w', encoding =  'utf-8' ) as f_json:
        json.dump(risultato_finale, f_json, indent = 4)