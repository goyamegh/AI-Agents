import { StateGraph, START, END } from "@langchain/langgraph";
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";
import { Logger } from "../../utils/logger";
import { ReactAgentState } from "./react-agent";

// Configuration constants
const REACT_MAX_ITERATIONS = 10; // Maximum tool execution cycles before forcing final response

export class ReactGraphBuilder {
  private logger: Logger;

  constructor(logger: Logger) {
    this.logger = logger;
  }

  /**
   * Build the LangGraph state graph with nodes and edges
   */
  buildStateGraph(
    processInputNode: (state: ReactAgentState) => Promise<Partial<ReactAgentState>>,
    callModelNode: (state: ReactAgentState) => Promise<Partial<ReactAgentState>>,
    executeToolsNode: (state: ReactAgentState) => Promise<Partial<ReactAgentState>>,
    generateResponseNode: (state: ReactAgentState) => Promise<Partial<ReactAgentState>>
  ): any {
    // Create state graph with channels
    const graph = new StateGraph<ReactAgentState>({
      channels: {
        messages: {
          value: (x: any[], y: any[]) => [...x, ...y],
          default: () => [],
        },
        currentStep: {
          value: (x: string, y: string) => y || x,
          default: () => "processInput",
        },
        // TODO: Test whether we need to accumulate tool calls or just replace
        toolCalls: {
          value: (x: any[], y: any[]) => y, // Replace instead of accumulate
          default: () => [],
        },
        toolResults: {
          value: (x: any, y: any) => ({ ...x, ...y }),
          default: () => ({}),
        },
        iterations: {
          value: (x: number, y: number) => y,
          default: () => 0,
        },
        maxIterations: {
          value: (x: number, y: number) => y || x,
          default: () => REACT_MAX_ITERATIONS,
        },
        shouldContinue: {
          value: (x: boolean, y: boolean) => y,
          default: () => true,
        },
        streamingCallbacks: {
          value: (x: any, y: any) => y || x,
          default: () => undefined,
        },
        // Client-provided inputs - preserve throughout graph execution
        clientState: {
          value: (x: any, y: any) => y || x,
          default: () => undefined,
        },
        clientContext: {
          value: (x: any[], y: any[]) => y || x,
          default: () => undefined,
        },
        clientTools: {
          value: (x: any[], y: any[]) => y || x,
          default: () => undefined,
        },
        threadId: {
          value: (x: string, y: string) => y || x,
          default: () => undefined,
        },
        runId: {
          value: (x: string, y: string) => y || x,
          default: () => undefined,
        },
        modelId: {
          value: (x: string, y: string) => y || x,
          default: () => undefined,
        },
        lastToolExecution: {
          value: (x: number, y: number) => y || x,
          default: () => undefined,
        },
      },
    });

    // Add nodes (avoiding reserved names)
    graph.addNode("processInput", processInputNode);
    graph.addNode("callModel", callModelNode);
    graph.addNode("executeTools", executeToolsNode);
    graph.addNode("generateResponse", generateResponseNode);

    // Add edges
    graph.addEdge(START as "__start__", "processInput" as "__end__");
    graph.addEdge("processInput" as "__start__", "callModel" as "__end__");

    // Conditional edge from callModel
    graph.addConditionalEdges(
      "callModel" as any,
      (state: ReactAgentState) => {
        // Log the decision for debugging
        // this.logger.info('🔄 Graph Decision: callModel -> next node', {
        //   toolCallsCount: state.toolCalls.length,
        //   hasToolCalls: state.toolCalls.length > 0,
        //   iterations: state.iterations,
        //   maxIterations: state.maxIterations,
        //   messageCount: state.messages.length,
        //   lastMessageRole: state.messages[state.messages.length - 1]?.role,
        //   nextNode: state.toolCalls.length > 0 ? "executeTools" : "generateResponse"
        // });

        if (state.toolCalls.length > 0) {
          return "executeTools";
        }
        return "generateResponse";
      }
    );

    // Edge from executeTools back to callModel or to response
    graph.addConditionalEdges(
      "executeTools" as "__start__",
      (state: ReactAgentState) => {
        const shouldContinue =
          state.iterations < state.maxIterations && state.shouldContinue;

        // Log the decision for debugging
        this.logger.info("🔄 Graph Decision: executeTools -> next node", {
          iterations: state.iterations,
          maxIterations: state.maxIterations,
          shouldContinue: state.shouldContinue,
          willContinue: shouldContinue,
          messageCount: state.messages.length,
          lastMessageRole: state.messages[state.messages.length - 1]?.role,
          hasToolResults: Object.keys(state.toolResults).length > 0,
          nextNode: shouldContinue ? "callModel" : "generateResponse",
        });

        if (shouldContinue) {
          return "callModel";
        }
        return "generateResponse";
      }
    );

    graph.addEdge("generateResponse" as "__start__", END as "__end__");

    // Compile the graph with SQLite checkpointer for memory persistence
    // Use in-memory SQLite database (no setup required)
    const checkpointer = SqliteSaver.fromConnString(":memory:");

    return graph.compile({ checkpointer });
  }

  /**
   * Create the initial state for the graph execution
   */
  createInitialState(
    messages: any[],
    additionalInputs?: {
      state?: any;
      context?: any[];
      tools?: any[];
      threadId?: string;
      runId?: string;
      modelId?: string;
    },
    streamingCallbacks?: any
  ): ReactAgentState {
    return {
      messages: messages, // Use the messages directly from UI
      currentStep: "processInput",
      toolCalls: [],
      toolResults: {},
      iterations: 0,
      maxIterations: REACT_MAX_ITERATIONS,
      shouldContinue: true,
      streamingCallbacks: streamingCallbacks,
      // Add client inputs to initial state
      clientState: additionalInputs?.state,
      clientContext: additionalInputs?.context,
      clientTools: additionalInputs?.tools,
      threadId: additionalInputs?.threadId,
      runId: additionalInputs?.runId,
      modelId: additionalInputs?.modelId,
    };
  }

  /**
   * Create graph configuration for stateless operation
   */
  createGraphConfig(threadId?: string, runId?: string): any {
    return {
      configurable: {
        thread_id: `${threadId || "session"}_${runId || Date.now()}`,
      },
    };
  }
}