import { BedrockRuntimeClient, ConverseStreamCommand } from "@aws-sdk/client-bedrock-runtime";
import { StateGraph, START, END } from "@langchain/langgraph";
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";
import { BaseMessage, HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import { readFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { BaseMCPClient, LocalMCPClient, HTTPMCPClient } from '../../mcp/index';
import { MCPServerConfig } from '../../types/mcp-types';
import { Logger } from '../../utils/logger';
import { BaseAgent, StreamingCallbacks } from '../base-agent';
import readline from 'readline';
import { v4 as uuidv4 } from 'uuid';

// LangGraph state interface
export interface LangGraphAgentState {
  messages: any[];
  currentStep: string;
  toolCalls: any[];
  toolResults: Record<string, any>;
  iterations: number;
  maxIterations: number;
  shouldContinue: boolean;
  streamingCallbacks?: StreamingCallbacks;
}

/**
 * LangGraph Agent Implementation
 * 
 * This agent uses LangGraph for orchestration while reusing:
 * - AWS Bedrock ConverseStream API (same as Jarvis Agent)
 * - MCP tool infrastructure
 * - System prompts
 * - Streaming callbacks
 * 
 * The key difference is the graph-based state management and workflow orchestration
 */
export class LangGraphAgent implements BaseAgent {
  private bedrockClient: BedrockRuntimeClient;
  private mcpClients: Record<string, BaseMCPClient> = {};
  private conversationHistory: any[] = [];
  private systemPrompt: string = '';
  private logger: Logger;
  private graph?: StateGraph<LangGraphAgentState>;
  private compiledGraph?: any;
  private rl?: readline.Interface;

  constructor() {
    this.logger = new Logger();
    const region = process.env.AWS_REGION || 'us-east-1';
    
    this.bedrockClient = new BedrockRuntimeClient({
      region: region
    });
    
    this.logger.info('LangGraph Agent initialized', {
      region: region,
      hasAwsAccessKey: !!process.env.AWS_ACCESS_KEY_ID,
      hasAwsSecretKey: !!process.env.AWS_SECRET_ACCESS_KEY,
      hasAwsProfile: !!process.env.AWS_PROFILE,
      hasAwsSessionToken: !!process.env.AWS_SESSION_TOKEN
    });
  }

  getAgentType(): string {
    return 'langgraph';
  }

  async initialize(
    configs: Record<string, MCPServerConfig>, 
    customSystemPrompt?: string
  ): Promise<void> {
    this.logger.info('Initializing LangGraph Agent', {
      serverCount: Object.keys(configs).length,
      servers: Object.keys(configs)
    });

    // Connect to all MCP servers (reuse existing infrastructure)
    for (const [name, config] of Object.entries(configs)) {
      this.logger.info(`Connecting to MCP server: ${name}`);
      
      // Create appropriate client based on config type
      let client: BaseMCPClient;
      if (config.type === 'http') {
        client = new HTTPMCPClient(config, name, this.logger);
      } else {
        client = new LocalMCPClient(config, name, this.logger);
      }
      
      await client.connect();
      this.mcpClients[name] = client;
      
      const tools = client.getTools();
      this.logger.info(`Connected to ${name}, available tools: ${tools.length}`);
    }

    // Load system prompt (use the same prompt as other agents)
    if (customSystemPrompt) {
      this.systemPrompt = customSystemPrompt;
    } else {
      const promptPath = join(__dirname, '../../prompts/claudecode.md');
      
      try {
        if (existsSync(promptPath)) {
          this.systemPrompt = readFileSync(promptPath, 'utf-8');
        } else {
          this.systemPrompt = 'You are a helpful AI assistant.';
        }
      } catch (error) {
        this.logger.warn('Failed to load system prompt, using default', { error });
        this.systemPrompt = 'You are a helpful AI assistant.';
      }
    }

    // Build the LangGraph state graph
    this.buildStateGraph();
    
    this.logger.info('LangGraph Agent initialization complete', {
      totalTools: this.getAllTools().length
    });
  }

  private buildStateGraph(): void {
    // Create state graph with channels
    this.graph = new StateGraph<LangGraphAgentState>({
      channels: {
        messages: {
          value: (x: any[], y: any[]) => [...x, ...y],
          default: () => []
        },
        currentStep: {
          value: (x: string, y: string) => y || x,
          default: () => "processInput"
        },
        toolCalls: {
          value: (x: any[], y: any[]) => [...x, ...y],
          default: () => []
        },
        toolResults: {
          value: (x: any, y: any) => ({ ...x, ...y }),
          default: () => ({})
        },
        iterations: {
          value: (x: number, y: number) => y,
          default: () => 0
        },
        maxIterations: {
          value: (x: number, y: number) => y || x,
          default: () => 5
        },
        shouldContinue: {
          value: (x: boolean, y: boolean) => y,
          default: () => true
        },
        streamingCallbacks: {
          value: (x: any, y: any) => y || x,
          default: () => undefined
        }
      }
    });

    // Add nodes (avoiding reserved names)
    this.graph.addNode("processInput", this.processInputNode.bind(this));
    this.graph.addNode("callModel", this.callModelNode.bind(this));
    this.graph.addNode("executeTools", this.executeToolsNode.bind(this));
    this.graph.addNode("generateResponse", this.generateResponseNode.bind(this));

    // Add edges
    this.graph.addEdge(START as "__start__", "processInput" as "__end__");
    this.graph.addEdge("processInput" as "__start__", "callModel" as "__end__");
    
    // Conditional edge from callModel
    this.graph.addConditionalEdges(
      "callModel" as any,
      (state: LangGraphAgentState) => {
        if (state.toolCalls.length > 0) {
          return "executeTools";
        }
        return "generateResponse";
      }
    );

    // Edge from executeTools back to callModel or to response
    this.graph.addConditionalEdges(
      "executeTools" as "__start__",
      (state: LangGraphAgentState) => {
        if (state.iterations < state.maxIterations && state.shouldContinue) {
          return "callModel";
        }
        return "generateResponse";
      }
    );

    this.graph.addEdge("generateResponse" as "__start__", END as "__end__");

    // Compile the graph with SQLite checkpointer for memory persistence
    // Use in-memory SQLite database (no setup required)
    const checkpointer = SqliteSaver.fromConnString(":memory:");
    
    this.compiledGraph = this.graph.compile({ checkpointer });
  }

  private async processInputNode(state: LangGraphAgentState): Promise<Partial<LangGraphAgentState>> {
    // Process input and prepare for model call
    return {
      currentStep: "processInput",
      iterations: state.iterations + 1
    };
  }

  private async callModelNode(state: LangGraphAgentState): Promise<Partial<LangGraphAgentState>> {
    const { messages, streamingCallbacks } = state;
    
    // Prepare messages for Bedrock (same as Jarvis)
    const bedrockMessages = this.prepareMessagesForBedrock(messages);
    
    // Get available tools
    const tools = this.getAllTools();
    const toolConfig = this.prepareToolConfig(tools);

    // Create the command for Bedrock ConverseStream
    const command = new ConverseStreamCommand({
      modelId: "us.anthropic.claude-sonnet-4-20250514-v1:0",
      messages: bedrockMessages,
      system: [{ text: this.systemPrompt }],
      toolConfig: toolConfig,
      inferenceConfig: {
        maxTokens: 4096,
        temperature: 0,
      }
    });

    try {
      const response = await this.bedrockClient.send(command);
      const processedResponse = await this.processStreamingResponse(response, streamingCallbacks);
      
      return {
        messages: [...messages, processedResponse.message],
        toolCalls: processedResponse.toolCalls || [],
        currentStep: "callModel"
      };
    } catch (error) {
      this.logger.error('Error calling model', { error });
      return {
        shouldContinue: false,
        currentStep: "callModel"
      };
    }
  }

  private async executeToolsNode(state: LangGraphAgentState): Promise<Partial<LangGraphAgentState>> {
    const { toolCalls, streamingCallbacks } = state;
    const toolResults: Record<string, any> = {};

    for (const toolCall of toolCalls) {
      const { toolName, toolUseId, input } = toolCall;
      
      streamingCallbacks?.onToolUseStart?.(toolName, toolUseId, input);
      
      try {
        const result = await this.executeToolCall(toolName, input);
        toolResults[toolUseId] = result;
        streamingCallbacks?.onToolResult?.(toolName, toolUseId, result);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        streamingCallbacks?.onToolError?.(toolName, toolUseId, errorMessage);
        toolResults[toolUseId] = { error: errorMessage };
      }
    }

    return {
      toolResults: { ...state.toolResults, ...toolResults },
      toolCalls: [], // Clear tool calls after execution
      currentStep: "executeTools"
    };
  }

  private async generateResponseNode(state: LangGraphAgentState): Promise<Partial<LangGraphAgentState>> {
    const { streamingCallbacks } = state;
    
    streamingCallbacks?.onTurnComplete?.();
    
    return {
      currentStep: "generateResponse",
      shouldContinue: false
    };
  }

  // Helper methods that reuse existing infrastructure
  private prepareMessagesForBedrock(messages: any[]): any[] {
    // Convert messages to Bedrock format (same as Jarvis)
    return messages.map(msg => ({
      role: msg.role || 'user',
      content: [{ text: msg.content || '' }]
    }));
  }

  private prepareToolConfig(tools: any[]): any {
    // Prepare tool configuration for Bedrock (same as Jarvis)
    if (tools.length === 0) return undefined;

    return {
      tools: tools.map(tool => ({
        toolSpec: tool.toolSpec
      }))
    };
  }

  private async processStreamingResponse(response: any, callbacks?: StreamingCallbacks): Promise<any> {
    // Process streaming response from Bedrock (similar to Jarvis)
    const result = {
      message: { role: 'assistant', content: '' },
      toolCalls: []
    };

    if (response.stream) {
      for await (const chunk of response.stream) {
        if (chunk.contentBlockStart) {
          const start = chunk.contentBlockStart.start;
          if (start?.text) {
            callbacks?.onTextStart?.(start.text);
            result.message.content += start.text;
          } else if (start?.toolUse) {
            const toolCall = {
              toolName: start.toolUse.name,
              toolUseId: start.toolUse.toolUseId,
              input: {}
            };
            result.toolCalls.push(toolCall);
          }
        }

        if (chunk.contentBlockDelta) {
          const delta = chunk.contentBlockDelta.delta;
          if (delta?.text) {
            callbacks?.onTextDelta?.(delta.text);
            result.message.content += delta.text;
          } else if (delta?.toolUse && result.toolCalls.length > 0) {
            const lastToolCall = result.toolCalls[result.toolCalls.length - 1];
            lastToolCall.input = JSON.parse(delta.toolUse.input);
          }
        }
      }
    }

    return result;
  }

  private async executeToolCall(toolName: string, input: any): Promise<any> {
    // Execute tool call through MCP (same as Jarvis)
    // Tool names come in format: serverName__toolName
    const parts = toolName.split('__');
    const serverName = parts[0];
    const actualToolName = parts.slice(1).join('__');
    
    const client = this.mcpClients[serverName];
    if (!client) {
      throw new Error(`MCP server ${serverName} not found for tool ${toolName}`);
    }
    
    const tools = client.getTools();
    const tool = tools.find(t => t.name === actualToolName);
    if (!tool) {
      throw new Error(`Tool ${actualToolName} not found in server ${serverName}`);
    }
    
    return await client.executeTool(actualToolName, input);
  }

  async processMessageWithCallbacks(message: string, callbacks: StreamingCallbacks): Promise<void> {
    try {
      // Create initial state
      const initialState: LangGraphAgentState = {
        messages: [{ role: 'user', content: message }],
        currentStep: "processInput",
        toolCalls: [],
        toolResults: {},
        iterations: 0,
        maxIterations: 5,
        shouldContinue: true,
        streamingCallbacks: callbacks
      };

      // Run the graph with thread configuration for memory persistence
      const config = { 
        configurable: { thread_id: "default_session" }
      };
      const finalState = await this.compiledGraph.invoke(initialState, config);
      
      // Add to conversation history
      this.conversationHistory.push(
        { role: 'user', content: [{ text: message }] },
        finalState.messages[finalState.messages.length - 1]
      );

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.logger.error('Error processing message with callbacks', { error: errorMessage });
      callbacks.onError?.(errorMessage);
    }
  }

  async sendMessage(message: string): Promise<void> {
    // For CLI mode - simple console output (same as Jarvis)
    const callbacks: StreamingCallbacks = {
      onTextStart: (text: string) => {
        process.stdout.write(text);
      },
      onTextDelta: (delta: string) => {
        process.stdout.write(delta);
      },
      onToolUseStart: (toolName: string, _toolUseId: string, input: any) => {
        console.log(`\n🔧 Using tool: ${toolName.split('__').pop() || toolName}`);
        console.log(`   Input: ${JSON.stringify(input, null, 2)}`);
      },
      onToolResult: (toolName: string, _toolUseId: string, result: any) => {
        console.log(`✅ Tool result:`, JSON.stringify(result, null, 2).substring(0, 500));
      },
      onToolError: (toolName: string, _toolUseId: string, error: string) => {
        console.log(`❌ Tool error:`, error);
      },
      onTurnComplete: () => {
        console.log('\n');
      },
      onError: (error: string) => {
        console.error('Error:', error);
      }
    };

    await this.processMessageWithCallbacks(message, callbacks);
  }

  getAllTools(): any[] {
    // Get all tools from all MCP clients and format for Bedrock
    const allTools: any[] = [];
    
    for (const [serverName, client] of Object.entries(this.mcpClients)) {
      const serverTools = client.getTools();
      
      for (const tool of serverTools) {
        // Format tool for Bedrock API (matching Jarvis format)
        allTools.push({
          toolSpec: {
            name: `${serverName}__${tool.name}`,
            description: tool.description || `Tool: ${tool.name}`,
            inputSchema: {
              json: tool.inputSchema
            }
          }
        });
      }
    }
    
    return allTools;
  }

  async startInteractiveMode(): Promise<void> {
    console.log('🤖 LangGraph Agent Ready!');
    console.log('📊 Using graph-based orchestration with MCP tools');
    console.log('💡 Type your message or "exit" to quit\n');

    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      prompt: '> '
    });

    // Return a Promise that resolves only when user quits
    return new Promise<void>((resolve) => {
      this.rl!.prompt();

      this.rl!.on('line', async (line) => {
        const input = line.trim();
        
        if (input.toLowerCase() === 'exit' || input.toLowerCase() === 'quit') {
          console.log('👋 Goodbye!');
          this.cleanup();
          resolve(); // Resolve the Promise instead of calling process.exit(0)
          return;
        }

        if (input) {
          await this.sendMessage(input);
        }

        this.rl!.prompt();
      });

      this.rl!.on('close', () => {
        console.log('\n👋 Goodbye!');
        this.cleanup();
        resolve(); // Resolve the Promise instead of calling process.exit(0)
      });
    });
  }

  cleanup(): void {
    if (this.rl) {
      this.rl.close();
    }
    
    // Disconnect all MCP clients
    for (const client of Object.values(this.mcpClients)) {
      client.disconnect();
    }
    
    this.logger.info('LangGraph Agent cleanup completed');
  }
}