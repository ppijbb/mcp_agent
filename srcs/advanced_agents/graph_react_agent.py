import asyncio
import logging
import re
import uuid
from typing import List, Dict, Any, Tuple
import json
import inspect

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from srcs.common.utils import setup_agent_app
import graphviz
import os
from logging import Logger
from srcs.utils.graph_database.connector import Neo4jConnector

async def clear_graph(logger: Logger):
    """Deletes all nodes and relationships from the graph."""
    connector = Neo4jConnector()
    try:
        await connector.connect()
        logger.info("Clearing the graph database...")
        _, summary = await connector.query("MATCH (n) DETACH DELETE n")
        if summary:
            logger.info(f"Graph database cleared. Deleted {summary.counters.nodes_deleted} nodes and {summary.counters.relationships_deleted} relationships.")
        else:
            logger.info("Graph database cleared.")
    except Exception as e:
        logger.error(f"Failed to clear the graph: {e}", exc_info=True)
    finally:
        await connector.close()

class GraphReActAgent(Agent):
    """
    An agent that implements the Graph-based ReAct pattern.
    It's designed to be controlled by an Orchestrator.
    """
    def __init__(self, **kwargs):
        # Standard agent initialization
        super().__init__(
            name="GraphReActAgent",
            instruction="You are a powerful AI agent that uses a knowledge graph to reason step-by-step (ReAct pattern) to answer complex queries. You can expand, evaluate, prune, and synthesize thoughts in the graph.",
            **kwargs
        )
        self.max_iterations = 10
        self.model = "gemini-2.5-flash-lite-preview-06-07"
        self.graph = Neo4jConnector()
        # self.orchestrator and self.logger will be set by the MCPApp/Orchestrator
        # that this agent is part of.
        
    async def execute_react_cycle(self, user_query: str):
        """
        This is the main entry point for the agent's logic, to be called by the orchestrator.
        """
        session_id = str(uuid.uuid4())
        self.logger.info(f"Executing ReAct cycle for query: '{user_query}' with session ID: {session_id}")
        
        observation = await self._action_initiate(user_query, session_id)
        
        for i in range(self.max_iterations):
            self.logger.info(f"--- Iteration {i+1}/{self.max_iterations} ---")
            
            thought_prompt = self._get_thought_prompt(user_query, observation)
            thought_text = await self.orchestrator.generate_str(
                message=thought_prompt,
                request_params=RequestParams(model=self.model, temperature=0.4)
            )
            self.logger.info(f"THOUGHT: {thought_text}")

            action_name, action_input = self._parse_action(thought_text)
            
            if action_name == "finish":
                self.logger.info(f"ACTION: Finishing with answer: {action_input}")
                await self._action_cleanup(session_id)
                return action_input

            observation = await self._execute_action(action_name, action_input, session_id)
            self.logger.info(f"OBSERVATION: {observation}")

        final_answer = await self._action_synthesize("Max iterations reached.", session_id)
        await self._action_cleanup(session_id)
        return final_answer

    # Helper methods now take session_id
    def _get_thought_prompt(self, original_query: str, observation: str) -> str:
        """Generates the prompt for the THOUGHT step."""
        return f"""
        You are a GraphReAct agent. Your goal is to answer the user's query: "{original_query}"
        You operate by manipulating a knowledge graph.
        Your last action resulted in this observation: "{observation}"

        Based on the observation, what is the next logical action to take?
        Your available actions are:
        - `check_status()`: Get a summary of the graph (e.g., number of nodes by status).
        - `expand()`: Expand on 'new' or 'expanded' thoughts to generate more ideas.
        - `evaluate()`: Evaluate 'unevaluated' thoughts to assign a validity score.
        - `prune()`: Prune thoughts with low scores to clean the graph.
        - `g_search(query)`: Use Google Search to get external information for a thought.
        - `synthesize(reason)`: If the graph is mature enough, synthesize the final answer.
        - `finish(answer)`: Provide the final answer if it's explicitly known.
        
        Think step-by-step. What is the single best action to perform next?
        Example: `expand()` or `g_search(What is the current market trend for AI assistants?)`
        """

    def _parse_action(self, thought_text: str) -> Tuple[str, str]:
        """Parses the action and its input from the LLM's thought."""
        action_match = re.search(r"(\w+)\((.*)\)", thought_text)
        if action_match:
            name = action_match.group(1).strip()
            input_val = action_match.group(2).strip().strip("'\"")
            return name, input_val
        
        action_word = thought_text.split()[0].strip().replace('`', '')
        if action_word in ["expand", "evaluate", "prune", "check_status", "synthesize"]:
            return action_word, ""
            
        return "finish", "Could not parse a valid action from the thought."

    async def _execute_action(self, name: str, input_val: str, session_id: str) -> str:
        """Executes the parsed action."""
        action_map = {
            "initiate": self._action_initiate,
            "expand": self._action_expand,
            "evaluate": self._action_evaluate,
            "prune": self._action_prune,
            "synthesize": self._action_synthesize,
            "check_status": self._action_check_status,
            "g_search": self._action_g_search,
        }
        action_func = action_map.get(name.lower())

        if action_func:
            return await action_func(input_val, session_id)
        else:
            return f"Unknown action: {name}. Please choose from the available actions."

    async def _action_initiate(self, query: str, session_id: str) -> str:
        self.logger.info(f"Initiating graph for session {session_id}")
        cypher = "CREATE (t:Thought {id: $id, session_id: $session_id, text: $text, status: 'new', is_initial: true, timestamp: datetime()})"
        self.graph.execute_query(cypher, {"id": str(uuid.uuid4()), "session_id": session_id, "text": query})
        return f"Initiated graph with first thought."

    async def _action_expand(self, _, session_id: str) -> str:
        self.logger.info("Expanding thoughts.")
        find_cypher = "MATCH (t:Thought {session_id: $session_id, status: 'new'}) RETURN t LIMIT 1"
        nodes_to_expand = self.graph.execute_query(find_cypher, {"session_id": session_id})
        if not nodes_to_expand:
            return "No new thoughts to expand."
        thought = nodes_to_expand[0]['t']
        self.graph.execute_query("MATCH (t:Thought {id: $id}) SET t.status = 'expanding'", {"id": thought['id']})
        prompt = f"Given the thought: '{thought['text']}', generate two distinct follow-up thoughts. Phrase them as complete sentences."
        generated_text = await self.orchestrator.generate_str(message=prompt, request_params=RequestParams(model=self.model))
        new_thoughts = [t.strip() for t in generated_text.split('\n') if t.strip()]
        for new_text in new_thoughts:
            cypher = "MATCH (p:Thought {id:$p_id}) CREATE (c:Thought {id:$c_id, session_id:$s_id, text:$text, status:'unevaluated', ts:datetime()})-[:DERIVES_FROM]->(p)"
            self.graph.execute_query(cypher, {"p_id":thought['id'], "c_id":str(uuid.uuid4()), "s_id":session_id, "text":new_text})
        self.graph.execute_query("MATCH (t:Thought {id: $id}) SET t.status = 'expanded'", {"id": thought['id']})
        return f"Expanded thought {thought['id']} into {len(new_thoughts)} new thoughts."

    async def _action_evaluate(self, _, session_id: str) -> str:
        self.logger.info("Evaluating thoughts.")
        find_cypher = "MATCH (t:Thought {session_id: $session_id, status: 'unevaluated'}) RETURN t LIMIT 1"
        nodes_to_evaluate = self.graph.execute_query(find_cypher, {"session_id": session_id})
        if not nodes_to_evaluate:
            return "No unevaluated thoughts to evaluate."
        thought = nodes_to_evaluate[0]['t']
        prompt = f"Evaluate the validity of this thought: '{thought['text']}'. Return a score between 0.0 and 1.0 and a brief rationale. Format as 'Score: [score], Rationale: [rationale]'."
        evaluation_text = await self.orchestrator.generate_str(message=prompt, request_params=RequestParams(model=self.model, temperature=0.2))
        score_match = re.search(r"Score:\s*(\d\.?\d*)", evaluation_text)
        score = float(score_match.group(1)) if score_match else 0.0
        update_cypher = "MATCH (t:Thought {id: $id}) SET t.validity_score = $score, t.status = 'evaluated', t.rationale = $rationale"
        self.graph.execute_query(update_cypher, {"id": thought['id'], "score": score, "rationale": evaluation_text})
        return f"Evaluated thought {thought['id']} with score {score}."

    async def _action_prune(self, _, session_id: str, threshold=0.2) -> str:
        self.logger.info(f"Pruning thoughts with score below {threshold}.")
        cypher = "MATCH (t:Thought {session_id: $session_id, status: 'evaluated'}) WHERE t.validity_score < $threshold SET t.status = 'pruned' RETURN count(t) AS pruned_count"
        result = self.graph.execute_query(cypher, {"session_id": session_id, "threshold": threshold})
        pruned_count = result[0]['pruned_count'] if result else 0
        return f"Pruned {pruned_count} thoughts with a score below {threshold}."

    async def _action_synthesize(self, reason: str, session_id: str) -> str:
        self.logger.info(f"Synthesizing final answer. Reason: {reason}")
        cypher = "MATCH p=(r:Thought {session_id:$sid, is_initial:true})-[:DERIVES_FROM*..]->(l) WHERE NOT (l)-[:DERIVES_FROM]->() AND all(n IN nodes(p) WHERE n.status <> 'pruned') WITH p, reduce(s=0.0, n IN nodes(p) | s+coalesce(n.validity_score,0)) AS total ORDER BY total DESC LIMIT 1 RETURN [n IN nodes(p) | n.text] AS texts"
        result = self.graph.execute_query(cypher, {"sid": session_id})
        if not result or not result[0]['texts']:
            return "Could not find a valid path to synthesize a conclusion."
        path_texts = "\n- ".join(result[0]['texts'])
        prompt = f"Based on the following sequence of thoughts, provide a comprehensive final answer.\n\nThought Path:\n- {path_texts}\n\nFinal Answer:"
        final_answer = await self.orchestrator.generate_str(message=prompt, request_params=RequestParams(model=self.model))
        return final_answer

    async def _action_check_status(self, _, session_id: str) -> str:
        self.logger.info("Checking graph status.")
        cypher = "MATCH (t:Thought {session_id: $session_id}) RETURN t.status AS status, count(t) AS count ORDER BY status"
        result = self.graph.execute_query(cypher, {"session_id": session_id})
        if not result:
            return "The graph for this session is empty."
        status_summary = ", ".join([f"{row['status']}: {row['count']}" for row in result])
        return f"Current graph status: {status_summary}."

    async def _action_g_search(self, entity: str) -> str:
        """
        Tool: Searches for an entity in the graph database.
        """
        self.logger.info(f"Executing graph search for entity: {entity}")
        query = "MATCH (n {name: $entity}) RETURN n"
        connector = Neo4jConnector()
        try:
            records, _ = await connector.query(query, {"entity": entity})
            if records:
                return f"Found entity '{entity}' in the graph: {records}"
            else:
                return f"Entity '{entity}' not found in the graph."
        except Exception as e:
            return f"Error searching in graph: {e}"
        finally:
            await connector.close()

    async def _action_g_add(self, source: str, relationship: str, target: str) -> str:
        """
        Tool: Adds a new relationship between two nodes in the graph.
        """
        self.logger.info(f"Executing graph add: ({source})-[{relationship}]->({target})")
        query = (
            "MERGE (a:Entity {name: $source}) "
            "MERGE (b:Entity {name: $target}) "
            "MERGE (a)-[r:%s]->(b) "
            "RETURN a, r, b" % relationship.upper().replace(" ", "_")
        )
        connector = Neo4jConnector()
        try:
            await connector.query(query, {"source": source, "target": target})
            return f"Successfully added relationship: ({source})-[{relationship}]->({target})"
        except Exception as e:
            return f"Error adding to graph: {e}"
        finally:
            await connector.close()

    async def _action_cleanup(self, session_id: str) -> str:
        self.logger.info(f"GraphReAct process finished for session {session_id}.")
        # query = "MATCH (n {session_id: $session_id}) DETACH DELETE n"
        # self.graph.execute_query(query, {"session_id": session_id})
        return "Cleanup complete."

    async def run(self, query: str, session_id: str) -> str:
        self.step = 1
        self.history = ""
        thought = "I need to start reasoning about the user's query."
        prompt_template = """
You are a reasoning agent that uses a graph as a workspace.
Your task is to answer the following query: "{query}"

You have the following tools at your disposal:
{tools}

Follow this process:
1.  **Think**: Break down the problem and decide which tool to use.
2.  **Act**: Use a tool. The available tools are `g_search`, `g_add`, and `finish`.
3.  **Observe**: See the result of the action.
4.  **Repeat**: Continue until you have enough information to answer the query.
5.  **Finish**: When you have the final answer, use the `finish` tool with your complete answer.

**Constraint**: You MUST use the graph to build your reasoning. Start by adding initial concepts to the graph and expand from there. Search the graph to see what you already know before adding new information.

Here is an example of a single step:
Thought: I need to start by adding the central concept of 'AI Agent' to the graph to anchor my reasoning.
Action: {{
    "tool_name": "g_add",
    "tool_args": {{
        "source": "AI Agent",
        "relationship": "BASED_ON",
        "target": "Graph Database"
    }}
}}

Here is the current state of your reasoning graph:
{graph_status}

Begin!

Thought:
{thought}
"""
        while self.step < self.max_steps:
            graph_status = await self._get_graph_status()
            tools_str = self.format_tools()
            prompt = prompt_template.format(query=query, tools=tools_str, graph_status=graph_status, thought=thought)
            
            llm_response = await self.orchestrator.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model="gemini-2.5-flash-lite-preview-06-07",
                json_response=True
            )
            
            observation = ""
            try:
                response_text = llm_response.choices[0].message.content
                response_json = json.loads(response_text)
                thought = response_json.get("thought", "")
                action = response_json.get("action", {})
                tool_name = action.get("tool_name")
                tool_args = action.get("tool_args", {})

                self.logger.info(f"--- STEP {self.step} ---")
                self.logger.info(f"THOUGHT: {thought}")
                self.logger.info(f"ACTION: Calling tool '{tool_name}' with args: {tool_args}")

                if tool_name == "finish":
                    final_answer = tool_args.get("answer", "No answer provided.")
                    self.logger.info(f"--- FINAL ANSWER --- \n{final_answer}")
                    return final_answer

                if tool_name in self.tools:
                    tool_function = self.tools[tool_name]
                    # Ensure all required args are present, otherwise provide a helpful error message.
                    required_params = inspect.signature(tool_function).parameters
                    missing_params = [p for p in required_params if p not in tool_args and p != 'self']
                    if missing_params:
                        observation = f"Error: Missing required arguments for tool '{tool_name}': {', '.join(missing_params)}"
                    else:
                        observation = await tool_function(**tool_args)
                else:
                    observation = f"Error: Unknown tool '{tool_name}'. Please use one of the available tools: {list(self.tools.keys())}"
            
            except json.JSONDecodeError:
                observation = "Error: Invalid JSON response. Please provide a valid JSON object with 'thought' and 'action'."
                thought = "The last response was not valid JSON. I must correct my output format."
            except Exception as e:
                observation = f"An unexpected error occurred: {e}"
                self.logger.error(f"Error during agent execution: {e}", exc_info=True)
            
            self.history += f"\nObservation {self.step}: {observation}\n"
            self.step += 1

        return "Agent stopped after reaching max steps."

app = MCPApp(
    name="graph_react_agent_app", 
    settings=get_settings("configs/mcp_agent.config.yaml"),
    human_input_callback=None
)

async def main():
    """Main function to run the GraphReActAgent."""
    async with app.run() as app_context:
        logger = app_context.logger
        try:
            await clear_graph(logger)
            
            # Create an instance of our agent first
            graph_agent = GraphReActAgent()

            # The orchestrator is the main entry point
            # Pass the agent instance to the orchestrator upon creation
            orchestrator = Orchestrator(
                llm_factory=GoogleAugmentedLLM,
                available_agents=[graph_agent]
            )
            
            # Define the high-level task for the orchestrator
            query = "Should we build a new AI agent based on graph databases? What are the pros and cons?"
            task = f"""
            Use the "GraphReActAgent" to answer the following query: "{query}"
            
            To do this, you must call the agent's `execute_react_cycle` method with the user's query.
            The agent will then perform a step-by-step reasoning process using a knowledge graph.
            Return the final answer provided by the agent.
            """

            logger.info("Starting GraphReAct workflow via orchestrator...")
            final_answer = await orchestrator.generate_str(task)
            
            logger.info("\n--- FINAL ANSWER ---")
            logger.info(final_answer)

        except Exception as e:
            logger.error(f"An error occurred in main: {e}", exc_info=True)
        finally:
            logger.info("Agent execution finished. Visualizing graph...")
            await visualize_graph(logger)

async def visualize_graph(logger: Logger):
    """
    Connects to the Neo4j database, fetches the graph data,
    and visualizes it using Graphviz.
    Saves the graph as 'graph.png' in the 'travel_results' directory.
    """
    connector = Neo4jConnector()
    try:
        await connector.connect()

        nodes_result, _ = await connector.query("MATCH (n) RETURN n")
        nodes = [record["n"] for record in nodes_result]

        rels_result, _ = await connector.query("MATCH ()-[r]->() RETURN r")
        relationships = [record["r"] for record in rels_result]

        if not nodes:
            logger.warning("No nodes found in the graph to visualize.")
            return

        dot = graphviz.Digraph('KnowledgeGraph', comment='Agent-Generated Knowledge Graph')
        dot.attr(rankdir='LR', size='8,5')
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='skyblue')

        for node in nodes:
            node_id = str(node.element_id)
            label = str(node.get("name", "Unnamed Node"))
            dot.node(node_id, label)

        for rel in relationships:
            start_node_id = str(rel.start_node.element_id)
            end_node_id = str(rel.end_node.element_id)
            label = str(rel.type)
            dot.edge(start_node_id, end_node_id, label=label)

        # Ensure the output directory exists
        output_dir = "travel_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the figure
        output_path = os.path.join(output_dir, "knowledge_graph")
        dot.render(output_path, format='png', view=False, cleanup=True)
        logger.info(f"Graph visualization saved to {output_path}.png")

    except Exception as e:
        logger.error(f"Failed to visualize graph: {e}", exc_info=True)
    finally:
        await connector.close()

if __name__ == "__main__":
    asyncio.run(main()) 