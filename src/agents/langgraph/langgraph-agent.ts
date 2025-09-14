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
import { getPrometheusMetricsEmitter } from '../../utils/metrics-emitter';

// Configuration constants
const LANGGRAPH_MAX_ITERATIONS = 10; // Maximum tool execution cycles before forcing final response

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
  lastToolExecution?: number; // Timestamp of last tool execution
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
          default: () => LANGGRAPH_MAX_ITERATIONS
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
        // Log the decision for debugging
        this.logger.info('🔄 Graph Decision: callModel -> next node', {
          toolCallsCount: state.toolCalls.length,
          hasToolCalls: state.toolCalls.length > 0,
          iterations: state.iterations,
          maxIterations: state.maxIterations,
          messageCount: state.messages.length,
          lastMessageRole: state.messages[state.messages.length - 1]?.role,
          nextNode: state.toolCalls.length > 0 ? "executeTools" : "generateResponse"
        });
        
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
        const shouldContinue = state.iterations < state.maxIterations && state.shouldContinue;
        
        // Log the decision for debugging
        this.logger.info('🔄 Graph Decision: executeTools -> next node', {
          iterations: state.iterations,
          maxIterations: state.maxIterations,
          shouldContinue: state.shouldContinue,
          willContinue: shouldContinue,
          messageCount: state.messages.length,
          lastMessageRole: state.messages[state.messages.length - 1]?.role,
          hasToolResults: Object.keys(state.toolResults).length > 0,
          nextNode: shouldContinue ? "callModel" : "generateResponse"
        });
        
        if (shouldContinue) {
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
    this.logger.info('Processing input', {
      messageCount: state.messages.length,
      iterations: state.iterations,
      maxIterations: state.maxIterations
    });
    
    return {
      currentStep: "processInput"
      // Don't increment iterations here - only increment after tool execution
    };
  }

  private async callModelNode(state: LangGraphAgentState): Promise<Partial<LangGraphAgentState>> {
    const { messages, streamingCallbacks, iterations, toolResults } = state;
    
    this.logger.info('📥 callModelNode: Starting', {
      iterations,
      maxIterations: state.maxIterations,
      messageCount: messages.length,
      lastMessageRole: messages[messages.length - 1]?.role,
      lastMessageContent: messages[messages.length - 1]?.content ? 
        JSON.stringify(messages[messages.length - 1].content).substring(0, 100) : 'undefined'
    });
    
    // Log the raw messages before processing
    this.logger.info('Raw messages before Bedrock preparation', {
      messages: messages.map((msg, idx) => ({
        index: idx,
        role: msg.role,
        contentType: Array.isArray(msg.content) ? 'array' : typeof msg.content,
        contentLength: Array.isArray(msg.content) ? msg.content.length : 
                       typeof msg.content === 'string' ? msg.content.length : 0,
        contentBlocks: Array.isArray(msg.content) ? 
          msg.content.map((c: any) => ({
            hasText: c.text !== undefined,
            hasToolUse: c.toolUse !== undefined,
            hasToolResult: c.toolResult !== undefined,
            toolUseId: c.toolUse?.toolUseId || c.toolResult?.toolUseId
          })) : null
      }))
    });
    
    // Prepare messages for Bedrock (same as Jarvis)
    const bedrockMessages = this.prepareMessagesForBedrock(messages);
    
    // Get available tools - but don't provide tools if we're at max iterations (to force a final response)
    const tools = this.getAllTools();
    
    // Check if there are tool results in the message history (properly formatted)
    const hasToolResultsInHistory = messages.some(msg => 
      Array.isArray(msg.content) && 
      msg.content.some((c: any) => c.toolResult !== undefined)
    );
    
    // Only disable tools if we're at max iterations to force a final response
    // Let the model decide if it needs more tools based on the conversation context
    const atMaxIterations = iterations >= state.maxIterations - 1;
    const shouldDisableTools = atMaxIterations;
    
    this.logger.info('🔧 Tool configuration decision', {
      hasToolResultsInHistory,
      iterations,
      maxIterations: state.maxIterations,
      atMaxIterations,
      shouldDisableTools,
      toolCount: tools.length
    });
    
    const toolConfig = shouldDisableTools ? undefined : this.prepareToolConfig(tools);

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
      // Log the actual messages being sent with full detail
      this.logger.info('LLM Request Messages (detailed)', {
        messageCount: bedrockMessages.length,
        messages: bedrockMessages.map((msg, idx) => ({
          index: idx,
          role: msg.role,
          contentLength: JSON.stringify(msg.content).length,
          contentBlocks: msg.content.map((c: any) => ({
            type: c.text !== undefined ? 'text' : 
                  c.toolUse !== undefined ? 'toolUse' : 
                  c.toolResult !== undefined ? 'toolResult' : 'unknown',
            toolUseId: c.toolUse?.toolUseId || c.toolResult?.toolUseId,
            hasContent: c.text?.length > 0 || c.toolUse !== undefined || c.toolResult !== undefined
          }))
        }))
      });
      
      this.logger.info('LLM Request to Bedrock', {
        modelId: "us.anthropic.claude-sonnet-4-20250514-v1:0",
        systemPromptLength: this.systemPrompt.length,
        messageCount: bedrockMessages.length,
        toolCount: toolConfig ? tools.length : 0,
        toolsEnabled: !!toolConfig,
        hasToolResultsInHistory,
        iterations,
        maxIterations: state.maxIterations,
        atMaxIterations,
        shouldDisableTools,
        reason: shouldDisableTools ? 'max_iterations_reached' : 'tools_enabled',
        maxTokens: 4096,
        temperature: 0
      });

      // Emit warning and metric if we're forcing a final response due to max iterations
      if (atMaxIterations) {
        this.logger.warn('MAX_ITERATIONS_REACHED: Forcing final response from LLM', {
          iterations,
          maxIterations: state.maxIterations,
          messageCount: bedrockMessages.length
        });
        
        // Emit Prometheus metric
        const metricsEmitter = getPrometheusMetricsEmitter();
        metricsEmitter.emitCounter('langgraph_agent_max_iterations_reached_total', 1, {
          agent_type: 'langgraph',
          max_iterations: state.maxIterations.toString()
        });
      }

      const response = await this.bedrockClient.send(command);
      const processedResponse = await this.processStreamingResponse(response, streamingCallbacks);
      
      this.logger.info('📤 LLM Response from Bedrock', {
        contentBlocksCount: processedResponse.message.content.length,
        contentBlocks: processedResponse.message.content.map((block: any) => ({
          type: block.text !== undefined ? 'text' : block.toolUse !== undefined ? 'toolUse' : 'unknown',
          hasContent: block.text !== undefined ? block.text.length > 0 : block.toolUse !== undefined
        })),
        textContent: processedResponse.message.textContent?.length || 0,
        textContentPreview: processedResponse.message.textContent?.substring(0, 100) || '',
        toolCallsCount: processedResponse.toolCalls?.length || 0,
        toolCallNames: processedResponse.toolCalls?.map(tc => tc.toolName) || [],
        hasContent: processedResponse.message.content.length > 0,
        hasText: processedResponse.message.textContent?.length > 0,
        hasToolCalls: processedResponse.toolCalls?.length > 0
      });

      // Check if the response contains XML tool calls in the text (fallback for when Bedrock doesn't recognize tools)
      let extractedToolCalls = processedResponse.toolCalls || [];
      let assistantMessage = processedResponse.message; // Use the complete message from Bedrock
      
      if (extractedToolCalls.length === 0 && processedResponse.message.textContent.includes('<function_calls>')) {
        this.logger.warn('Tool calls found in text content, attempting to parse XML');
        extractedToolCalls = this.parseToolCallsFromXML(processedResponse.message.textContent);
        
        // For XML tool calls, we need to handle them differently
        // Remove the XML from the text content
        if (extractedToolCalls.length > 0) {
          const xmlStart = processedResponse.message.textContent.indexOf('<function_calls>');
          const xmlEnd = processedResponse.message.textContent.indexOf('</function_calls>') + '</function_calls>'.length;
          const cleanedText = processedResponse.message.textContent.substring(0, xmlStart).trim();
          
          // Update the assistant message to remove XML from text blocks
          assistantMessage = {
            role: 'assistant',
            content: cleanedText ? [{ text: cleanedText }] : []
          };
        }
      }
      
      // Handle message history properly based on UML diagram flow
      // First iteration: Add assistant message with tool calls
      // Subsequent iterations: Only add assistant message if there are no tool calls (final response)
      
      const isFirstCallInTurn = iterations === 0;
      
      let updatedMessages: any[];
      
      if (isFirstCallInTurn) {
        // First call in this turn - always add the assistant message
        updatedMessages = [...messages, assistantMessage];
        this.logger.info('📝 First iteration: Adding initial assistant message', {
          previousMessageCount: messages.length,
          iterations,
          hasToolCalls: extractedToolCalls.length > 0
        });
      } else if (extractedToolCalls.length === 0) {
        // Subsequent iteration with no tool calls - this is the final response
        updatedMessages = [...messages, assistantMessage];
        this.logger.info('📝 Final response: Adding assistant message without tool calls', {
          previousMessageCount: messages.length,
          iterations,
          hasToolCalls: false
        });
      } else {
        // Subsequent iteration with tool calls - DON'T add another assistant message
        // The previous assistant message with tool_use is already in the history
        // We just need to track the new tool calls to execute
        this.logger.info('📝 Continuation with tools: NOT adding duplicate assistant message', {
          previousMessageCount: messages.length,
          iterations,
          hasToolCalls: true,
          reason: 'Previous assistant message with tool_use already exists'
        });
        updatedMessages = messages; // Keep messages as-is
      }

      this.logger.info('📊 callModelNode: Returning state', {
        newMessageCount: updatedMessages.length,
        messagesChanged: updatedMessages.length !== messages.length,
        addedMessage: updatedMessages.length > messages.length,
        toolCallsCount: extractedToolCalls.length,
        toolCallNames: extractedToolCalls.map(tc => tc.toolName),
        iterations,
        isFirstCallInTurn
      });

      return {
        messages: updatedMessages,
        toolCalls: extractedToolCalls,
        currentStep: "callModel"
      };
    } catch (error) {
      // Enhanced error logging to capture all error details
      this.logger.error('Error calling model', { 
        error,
        errorName: error?.name,
        errorMessage: error?.message,
        errorStack: error?.stack,
        errorCode: error?.code,
        errorType: typeof error,
        errorKeys: error ? Object.keys(error) : [],
        errorString: String(error)
      });

      // Handle credential expiration
      if (error?.name === 'ExpiredTokenException' || error?.name === 'CredentialsProviderError') {
        streamingCallbacks?.onError?.('AWS credentials expired. Please refresh your credentials and try again.');
        return {
          shouldContinue: false,
          currentStep: "callModel"
        };
      }
      
      // Return error message to streaming callbacks
      const errorMessage = error?.message || error?.name || String(error) || 'Unknown error occurred';
      streamingCallbacks?.onError?.(errorMessage);
      
      return {
        shouldContinue: false,
        currentStep: "callModel"
      };
    }
  }

  private async executeToolsNode(state: LangGraphAgentState): Promise<Partial<LangGraphAgentState>> {
    const { toolCalls, streamingCallbacks, messages } = state;
    const toolResults: Record<string, any> = {};

    this.logger.info('Executing tools', {
      toolCallsCount: toolCalls.length,
      toolNames: toolCalls.map(tc => tc.toolName),
      currentIterations: state.iterations,
      maxIterations: state.maxIterations
    });

    // Check if we've already executed these exact tool calls to prevent duplicates
    // We need to check if there are corresponding toolResult blocks for these toolUse blocks
    const toolCallSignatures = toolCalls.map(tc => tc.toolUseId);

    // Look for tool results in user messages (these indicate executed tools)
    const previouslyExecutedToolIds = messages
      .filter(m => m.role === 'user')
      .flatMap(m => Array.isArray(m.content) ? m.content : [])
      .filter((c: any) => c.toolResult)
      .map((c: any) => c.toolResult.toolUseId);

    const newToolCalls = toolCalls.filter(tc => 
      !previouslyExecutedToolIds.includes(tc.toolUseId)
    );

    if (newToolCalls.length === 0 && toolCalls.length > 0) {
      this.logger.warn('All tool calls have already been executed, skipping redundant execution', {
        attemptedToolCallIds: toolCallSignatures,
        previouslyExecutedToolIds,
        iterations: state.iterations
      });
      
      // Emit Prometheus metric for redundant tool call attempts
      const metricsEmitter = getPrometheusMetricsEmitter();
      metricsEmitter.emitCounter('langgraph_agent_redundant_tool_calls_total', toolCalls.length, {
        agent_type: 'langgraph',
        iteration: state.iterations.toString()
      });
      
      // CRITICAL: We must remove the last assistant message with unexecuted toolUse blocks
      // Otherwise Bedrock will complain about toolUse without corresponding toolResult
      // This happens when the model tries to call the same tool again
      const cleanedMessages = state.messages.slice(0, -1);
      
      return {
        messages: cleanedMessages,
        toolCalls: [],
        shouldContinue: false,
        currentStep: "executeTools"
      };
    }

    // Execute only new tool calls
    for (const toolCall of newToolCalls) {
      const { toolName, toolUseId, input } = toolCall;
      
      this.logger.info('Tool execution started', {
        toolName,
        toolUseId,
        input
      });
      
      streamingCallbacks?.onToolUseStart?.(toolName, toolUseId, input);
      
      try {
        const result = await this.executeToolCall(toolName, input);
        toolResults[toolUseId] = result;
        
        this.logger.info('Tool execution completed', {
          toolName,
          toolUseId,
          resultType: typeof result,
          resultKeys: result && typeof result === 'object' ? Object.keys(result) : [],
          resultLength: typeof result === 'string' ? result.length : undefined
        });
        
        streamingCallbacks?.onToolResult?.(toolName, toolUseId, result);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        
        this.logger.error('Tool execution failed', {
          toolName,
          toolUseId,
          error: errorMessage,
          input
        });
        
        streamingCallbacks?.onToolError?.(toolName, toolUseId, errorMessage);
        toolResults[toolUseId] = { error: errorMessage };
      }
    }

    const newIterations = state.iterations + 1;
    const willContinue = newIterations < state.maxIterations && state.shouldContinue;

    this.logger.info('All tools executed', {
      toolResultsCount: Object.keys(toolResults).length,
      successfulTools: Object.entries(toolResults).filter(([, result]) => !result.error).length,
      failedTools: Object.entries(toolResults).filter(([, result]) => result.error).length,
      previousIterations: state.iterations,
      newIterations: newIterations,
      maxIterations: state.maxIterations,
      shouldContinue: state.shouldContinue,
      willContinue: willContinue
    });

    // Add tool results to the message history for the model to see
    // The assistant message with toolUse blocks should already be in the messages array
    // We just need to add the user message with toolResult blocks
    
    // Create user message with tool result blocks for all executed tools
    // CRITICAL: Ensure content array is not empty
    const toolResultContent = newToolCalls.map(tc => ({
      toolResult: {
        toolUseId: tc.toolUseId,
        content: [{ text: JSON.stringify(toolResults[tc.toolUseId] || { error: 'No result found' }) }]
      }
    }));
    
    // Only create the message if we have tool results
    if (toolResultContent.length === 0) {
      this.logger.warn('No tool results to send back to model');
      return {
        toolCalls: [],
        currentStep: "executeTools",
        shouldContinue: false
      };
    }
    
    const toolResultMessage = {
      role: 'user' as const,
      content: toolResultContent
    };

    // Emit metric for iteration count
    const metricsEmitter = getPrometheusMetricsEmitter();
    metricsEmitter.emitHistogram('langgraph_agent_iterations_per_request', newIterations, {
      agent_type: 'langgraph'
    });

    // Note: The assistant message with toolUse blocks is already in messages from callModelNode
    // We only need to add the toolResult message
    return {
      messages: [...messages, toolResultMessage],
      toolResults: { ...state.toolResults, ...toolResults },
      toolCalls: [], // Clear tool calls after execution
      currentStep: "executeTools",
      iterations: newIterations, // Set the new iterations count
      shouldContinue: true, // Keep this true to allow the graph to decide
      lastToolExecution: Date.now() // Track when tools were last executed
    };
  }

  private async generateResponseNode(state: LangGraphAgentState): Promise<Partial<LangGraphAgentState>> {
    const { streamingCallbacks, messages, iterations } = state;
    
    this.logger.info('✅ generateResponseNode: Generating final response', {
      iterations,
      maxIterations: state.maxIterations,
      messageCount: messages.length,
      lastMessageRole: messages[messages.length - 1]?.role,
      hasResponse: messages[messages.length - 1]?.role === 'assistant'
    });
    
    streamingCallbacks?.onTurnComplete?.();
    
    return {
      currentStep: "generateResponse",
      shouldContinue: false
    };
  }

  // Helper methods that reuse existing infrastructure
  private prepareMessagesForBedrock(messages: any[]): any[] {
    // Convert messages to Bedrock format
    // Keep all messages with valid content
    return messages
      .filter(msg => {
        // Keep all messages that have content (including empty arrays for assistant)
        // Bedrock needs to see the full conversation flow including tool use/result pairs
        if (msg.content === undefined || msg.content === null) {
          return false;
        }
        // Keep messages with empty content arrays (assistant messages with only tool calls)
        if (Array.isArray(msg.content) && msg.content.length === 0 && msg.role === 'assistant') {
          return false; // Skip truly empty assistant messages
        }
        return true;
      })
      .map(msg => ({
        role: msg.role || 'user',
        // If content is already an array (proper format), use it directly
        // This preserves toolUse and toolResult blocks
        content: Array.isArray(msg.content) ? msg.content : [{ text: msg.content || '' }]
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
    // Process streaming response from Bedrock
    // We need to preserve the complete message structure including content blocks
    const result = {
      message: { 
        role: 'assistant', 
        content: [],  // This will hold all content blocks (text and toolUse)
        textContent: '' // Keep text separately for convenience
      },
      toolCalls: []  // Keep this for backward compatibility
    };

    let currentTextBlock = '';
    let currentToolUseBlock = null;
    let hasAnyContent = false; // Track if we have any content at all

    if (response.stream) {
      for await (const chunk of response.stream) {
        if (chunk.contentBlockStart) {
          const start = chunk.contentBlockStart.start;
          if (start?.text) {
            // Start a new text block
            currentTextBlock = start.text;
            callbacks?.onTextStart?.(start.text);
            result.message.textContent += start.text;
            hasAnyContent = true;
          } else if (start?.toolUse) {
            // Start a new tool use block
            currentToolUseBlock = {
              toolUse: {
                toolUseId: start.toolUse.toolUseId,
                name: start.toolUse.name,
                input: {}
              },
              inputBuffer: ''
            };
            
            // Also track in toolCalls for backward compatibility
            const toolCall = {
              toolName: start.toolUse.name,
              toolUseId: start.toolUse.toolUseId,
              input: {}
            };
            result.toolCalls.push(toolCall);
            hasAnyContent = true;
          }
        }

        if (chunk.contentBlockDelta) {
          const delta = chunk.contentBlockDelta.delta;
          if (delta?.text) {
            currentTextBlock += delta.text;
            callbacks?.onTextDelta?.(delta.text);
            result.message.textContent += delta.text;
          } else if (delta?.toolUse && currentToolUseBlock && result.toolCalls.length > 0) {
            const lastToolCall = result.toolCalls[result.toolCalls.length - 1];
            try {
              // Handle streaming JSON input - may be incomplete
              if (delta.toolUse.input) {
                // Accumulate input
                currentToolUseBlock.inputBuffer += delta.toolUse.input;
                
                // Try to parse the accumulated input
                try {
                  const parsedInput = JSON.parse(currentToolUseBlock.inputBuffer);
                  currentToolUseBlock.toolUse.input = parsedInput;
                  lastToolCall.input = parsedInput;
                } catch (parseError) {
                  // JSON is incomplete, continue accumulating
                }
              }
            } catch (error) {
              this.logger.warn('Error processing tool use delta', { 
                error: error.message,
                input: delta.toolUse.input
              });
            }
          }
        }

        if (chunk.contentBlockStop) {
          // Finalize the current block and add it to content
          if (currentTextBlock) {
            result.message.content.push({ text: currentTextBlock });
            currentTextBlock = '';
          } else if (currentToolUseBlock) {
            // Remove the inputBuffer before adding to content
            const { inputBuffer, ...toolUseBlock } = currentToolUseBlock;
            result.message.content.push(toolUseBlock);
            currentToolUseBlock = null;
          }
        }
      }
    }
    
    // CRITICAL: If we have no content blocks at all (shouldn't happen but handle it),
    // add an empty text block to ensure valid message format
    if (result.message.content.length === 0 && !hasAnyContent) {
      this.logger.warn('No content blocks received from Bedrock, adding empty text block');
      result.message.content.push({ text: '' });
    }

    // Final cleanup - ensure all tool inputs are properly parsed
    for (const toolCall of result.toolCalls) {
      if (toolCall.inputBuffer && !toolCall.input) {
        try {
          toolCall.input = JSON.parse(toolCall.inputBuffer);
        } catch (error) {
          this.logger.error('Failed to parse final tool input', {
            error: error.message,
            buffer: toolCall.inputBuffer,
            toolCall: toolCall.toolName
          });
          toolCall.input = {}; // Fallback to empty object
        }
      }
      // Clean up the buffer
      delete toolCall.inputBuffer;
    }

    // Log the complete assembled response
    this.logger.info('Complete LLM Response Assembled', {
      messageRole: result.message.role,
      messageContent: result.message.content,
      toolCallsCount: result.toolCalls.length,
      toolCalls: result.toolCalls.map(tc => ({
        toolName: tc.toolName,
        toolUseId: tc.toolUseId,
        input: tc.input
      }))
    });

    return result;
  }

  private parseToolCallsFromXML(content: string): any[] {
    const toolCalls: any[] = [];
    
    try {
      // Extract the function_calls block
      const functionCallsMatch = content.match(/<function_calls>([\s\S]*?)<\/function_calls>/);
      if (!functionCallsMatch) return toolCalls;
      
      const functionCallsXML = functionCallsMatch[1];
      
      // Find all invoke blocks
      const invokeMatches = functionCallsXML.matchAll(/<invoke name="([^"]+)">([\s\S]*?)<\/invoke>/g);
      
      for (const match of invokeMatches) {
        const toolName = match[1];
        const paramsXML = match[2];
        
        // Parse parameters
        const params: Record<string, any> = {};
        const paramMatches = paramsXML.matchAll(/<parameter name="([^"]+)">([^<]*)<\/parameter>/g);
        
        for (const paramMatch of paramMatches) {
          const paramName = paramMatch[1];
          const paramValue = paramMatch[2];
          params[paramName] = paramValue;
        }
        
        // Generate a unique tool use ID
        const toolUseId = `tooluse_${Math.random().toString(36).substring(2, 15)}`;
        
        toolCalls.push({
          toolName: toolName,
          toolUseId: toolUseId,
          input: params
        });
        
        this.logger.info('Parsed tool call from XML', {
          toolName,
          toolUseId,
          input: params
        });
      }
    } catch (error) {
      this.logger.error('Failed to parse tool calls from XML', {
        error: error instanceof Error ? error.message : String(error),
        content: content.substring(0, 500)
      });
    }
    
    return toolCalls;
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
      // Create initial state - ensure content is in array format
      const initialState: LangGraphAgentState = {
        messages: [{ role: 'user', content: [{ text: message }] }],
        currentStep: "processInput",
        toolCalls: [],
        toolResults: {},
        iterations: 0,
        maxIterations: LANGGRAPH_MAX_ITERATIONS,
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