import readLine from "node:readline/promises";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import { ChatGroq } from "@langchain/groq";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { TavilySearch } from "@langchain/tavily";
import { MemorySaver } from "@langchain/langgraph";

// 1. Define Node function
// 2. Build the Graph
// 3. Comple and invoke the graph

// Initialize memory to persist state between graph runs
const checkpointer = new MemorySaver();

const tool = new TavilySearch({
  // used for web search
  maxResults: 3,
  topic: "general",
  // includeAnswer: false,
  // includeRawContent: false,
  // includeImages: false,
  // includeImageDescriptions: false,
  // searchDepth: "basic",
  // timeRange: "day",
  // includeDomains: [],
  // excludeDomains: [],
});

const tools = [tool];
const toolNode = new ToolNode(tools);

const llm = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0,
  maxRetries: 2,
  // apiKey here no need to pass because by default it takes from env api key autmatically
  // other params...
}).bindTools(tools);

// 1. Define Node function
async function callModel(state: typeof MessagesAnnotation.State) {
  console.log("Calling LLM...");

  const response = await llm.invoke(state.messages); // here state.messages contains user query message , AI response message, and if tools present in future conatins as well

  return { messages: [response] };
}

// whether to call a tool or not
function shouldContinue(state: any) {
  const lastMessage = state.messages[state.messages.length - 1];

  // If the LLM makes a tool call, then we route to the "tools" node
  if (lastMessage?.tool_calls?.length > 0) {
    return "tools";
  }
  // Otherwise, we stop (reply to the user) using the special "__end__" node
  return "__end__";
}

// 2. Build the Graph
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addEdge("__start__", "agent") // __start__ is a special name for the entrypoint
  .addNode("tools", toolNode)
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue);
//   .addNode("tools", toolNode)
//   .addEdge("tools", "agent")
//   .addConditionalEdges("agent", shouldContinue);

// 3. Comple and invoke the graph
const app = workflow.compile({ checkpointer });

async function main() {
  const rl = readLine.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  while (true) {
    const userInput = await rl.question("You: ");

    if (userInput === "/bye") {
      break;
    }

    const finalState = await app.invoke(
      {
        messages: [{ role: "user", content: userInput }],
      },
      { configurable: { thread_id: "1" } } // here thread means chat if you pass dynamic thread_id then it gets new chat like chatgpt everytime with new thread_id
    );

    const lastMessage = finalState?.messages[finalState?.messages.length - 1];
    console.log("AI: ", lastMessage?.content);
  }

  rl.close();
}

main();
