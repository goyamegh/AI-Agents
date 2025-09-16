import { BedrockRuntimeClient, ConverseStreamCommand } from "@aws-sdk/client-bedrock-runtime";
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import { BaseMCPClient, LocalMCPClient, HTTPMCPClient } from '../../mcp/index';
import { MCPServerConfig } from '../../types/mcp-types';
import { Logger } from '../../utils/logger';
import { BaseAgent, StreamingCallbacks } from '../base-agent';
import { truncateToolResult } from '../../utils/truncate-tool-result';

export class JarvisAgent implements BaseAgent {
  private bedrockClient: BedrockRuntimeClient;
  private mcpClients: Record<string, BaseMCPClient> = {};
  private systemPrompt: string = '';
  private logger: Logger;
  private cliConversationHistory: any[] = []; // For CLI mode only

  constructor() {
    this.logger = new Logger();
    const region = process.env.AWS_REGION || 'us-east-1';
    
    this.bedrockClient = new BedrockRuntimeClient({
      region: region
    });
    
    this.logger.info('Jarvis Agent initialized', {
      region: region,
      hasAwsAccessKey: !!process.env.AWS_ACCESS_KEY_ID,
      hasAwsSecretKey: !!process.env.AWS_SECRET_ACCESS_KEY,
      hasAwsProfile: !!process.env.AWS_PROFILE,
      hasAwsSessionToken: !!process.env.AWS_SESSION_TOKEN
    });
  }

  async initialize(
    configs: Record<string, MCPServerConfig>, 
    customSystemPrompt?: string
  ): Promise<void> {
    this.logger.info('Initializing Jarvis Agent', {
      serverCount: Object.keys(configs).length,
      servers: Object.keys(configs)
    });

    // Connect to all MCP servers
    for (const [name, config] of Object.entries(configs)) {
      this.logger.info(`Connecting to MCP server: ${name}`);
      
      // Create appropriate client based on config type
      let client: BaseMCPClient;
      if (config.type === 'http') {
        client = new HTTPMCPClient(config, name, this.logger);
      } else {
        client = new LocalMCPClient(config, name, this.logger);
      }
      
      this.mcpClients[name] = client;
      
      try {
        await client.connect();
        this.logger.info(`✅ Connected to ${name} (${config.type})`);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        this.logger.warn(`⚠️  Failed to connect to ${name}`, { error: errorMessage });
      }
    }

    // Load system prompt - now that MCP clients are connected, we can generate dynamic prompt
    if (customSystemPrompt) {
      this.systemPrompt = this.enhanceSystemPrompt(customSystemPrompt);
      this.logger.info('Using enhanced custom system prompt with dynamic content');
    } else {
      // Always use dynamic system prompt that describes actual MCP tools
      this.systemPrompt = this.getDefaultSystemPrompt();
      this.logger.info('Using dynamic system prompt with MCP tools', {
        promptLength: this.systemPrompt.length,
        connectedServers: Object.keys(this.mcpClients).length
      });
    }

    this.logger.info('Jarvis Agent initialized with MCP servers', {
      connectedServers: Object.keys(this.mcpClients).length,
      totalTools: this.getAllTools().length
    });
  }

  private getDefaultSystemPrompt(): string {
    // Load AI agent template and inject dynamic MCP tool information
    const aiAgentPromptPath = join(__dirname, '../../prompts/claudecode.md');

    if (!existsSync(aiAgentPromptPath)) {
      this.logger.warn('claudecode.md not found, falling back to basic prompt');
      return this.getFallbackSystemPrompt();
    }

    try {
      let aiAgentPrompt = readFileSync(aiAgentPromptPath, 'utf-8');
      return this.enhanceSystemPrompt(aiAgentPrompt);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.logger.error('Failed to load claudecode.md', { error: errorMessage });
      return this.getFallbackSystemPrompt();
    }
  }

  private enhanceSystemPrompt(prompt: string): string {
    // Replace template placeholders with dynamic content
    const toolDescriptions = this.generateToolDescriptions();
    const toolValidationRules = this.getToolValidationRules();

    return prompt
      .replace('{{MCP_TOOL_DESCRIPTIONS}}', toolDescriptions)
      .replace('{{TOOL_PARAMETER_VALIDATION_RULES}}', toolValidationRules);
  }

  private getFallbackSystemPrompt(): string {
    // Fallback prompt if claudecode.md cannot be loaded
    const toolDescriptions = this.generateToolDescriptions();
    
    return `You are Jarvis Agent, an AI assistant specialized in helping with software engineering tasks.

You have access to the following tools through MCP (Model Context Protocol) servers:

${toolDescriptions}

Key behaviors:
- Be concise and direct
- Focus on the specific task at hand
- Use available tools when appropriate
- Provide clear, actionable responses
- When asked about your tools, list ONLY the MCP tools shown above

${this.getToolValidationRules()}`;
  }

  private getToolValidationRules(): string {
    return `
CRITICAL: Tool Parameter Validation and Self-Correction

When using tools, you MUST follow this validation process:

1. ANALYZE TOOL SCHEMA: Before calling any tool, carefully examine its description and parameter requirements
   - Pay special attention to REQUIRED PARAMETERS listed in the tool description
   - Review parameter types and constraints
   - Ensure you understand what each parameter expects

2. PARAMETER VALIDATION: Before making a tool call, verify:
   - All required parameters are provided
   - Parameter values are in the correct format
   - No required parameters are null, undefined, or empty

3. SELF-CORRECTION FLOW: If a tool call fails due to parameter validation:
   - Immediately analyze the error message to identify missing or incorrect parameters
   - Review the tool's parameter requirements again
   - Ask the user for clarification if needed parameter values are not available in the context
   - Retry the tool call with the correct parameters
   - Do NOT make the same parameter mistake twice

4. MULTI-TURN CORRECTION: For complex tool interactions:
   - If you receive an error about missing parameters
   - Stop and identify what information you need
   - Either extract the required information from context or ask the user
   - Retry with complete parameter set

Example self-correction pattern:
- Tool call fails: "Missing required parameter: path"
- Response: "I need to provide the path parameter. Let me ask you for the file path or search for it in the context."
- Retry with correct parameters

Remember: Tool parameter validation errors should trigger immediate self-correction, not repeated failures.`;
  }

  private generateToolDescriptions(): string {
    if (Object.keys(this.mcpClients).length === 0) {
      return "No MCP tools currently available.";
    }

    const descriptions: string[] = [];
    
    for (const [serverName, client] of Object.entries(this.mcpClients)) {
      const serverTools = client.getTools();
      if (serverTools.length > 0) {
        descriptions.push(`## ${serverName} server tools:`);
        
        for (const tool of serverTools) {
          let toolDescription = `- **${tool.name}**: ${tool.description || 'No description available'}`;
          
          if (tool.inputSchema.required && tool.inputSchema.required.length > 0) {
            toolDescription += `\n  - Required parameters: ${tool.inputSchema.required.join(', ')}`;
          }
          
          descriptions.push(toolDescription);
        }
        descriptions.push(''); // Add blank line between servers
      }
    }
    
    return descriptions.join('\n');
  }

  getAgentType(): string {
    return 'jarvis';
  }

  public getAllTools(): any[] {
    const MAX_TOOLS = 100; // Limit tools sent to Bedrock to prevent input size issues
    const tools: any[] = [];
    let toolCount = 0;
    
    // Prioritize certain servers
    const serverPriority = ['filesystem', 'opensearch-mcp-server', 'amzn-mcp', 'builder-mcp'];
    const processedServers = new Set<string>();
    
    // Process priority servers first
    for (const priorityServer of serverPriority) {
      if (this.mcpClients[priorityServer] && toolCount < MAX_TOOLS) {
        const serverTools = this.mcpClients[priorityServer].getTools();
        const maxFromServer = Math.min(30, MAX_TOOLS - toolCount, serverTools.length);
        
        for (let i = 0; i < maxFromServer; i++) {
          const tool = serverTools[i];
          
          // Simplified description to reduce request size
          let description = tool.description || `Tool: ${tool.name}`;
          if (description.length > 200) {
            description = description.slice(0, 197) + '...';
          }
          
          if (tool.inputSchema.required && tool.inputSchema.required.length > 0) {
            description += `\n\nRequired: ${tool.inputSchema.required.join(', ')}`;
          }

          tools.push({
            toolSpec: {
              name: `${priorityServer}__${tool.name}`,
              description: description,
              inputSchema: {
                json: tool.inputSchema
              }
            }
          });
          toolCount++;
        }
        processedServers.add(priorityServer);
      }
    }
    
    // Process remaining servers if we have capacity
    for (const [serverName, client] of Object.entries(this.mcpClients)) {
      if (!processedServers.has(serverName) && toolCount < MAX_TOOLS) {
        const serverTools = client.getTools();
        const maxFromServer = Math.min(10, MAX_TOOLS - toolCount, serverTools.length);
        
        for (let i = 0; i < maxFromServer; i++) {
          const tool = serverTools[i];
          
          // Simplified description to reduce request size
          let description = tool.description || `Tool: ${tool.name}`;
          if (description.length > 200) {
            description = description.slice(0, 197) + '...';
          }

          tools.push({
            toolSpec: {
              name: `${serverName}__${tool.name}`,
              description: description,
              inputSchema: {
                json: tool.inputSchema
              }
            }
          });
          toolCount++;
        }
      }
    }

    this.logger.debug(`Limited tools sent to Bedrock: ${toolCount} of ${this.getTotalToolCount()} available`);
    return tools;
  }

  private getTotalToolCount(): number {
    let total = 0;
    for (const client of Object.values(this.mcpClients)) {
      total += client.getTools().length;
    }
    return total;
  }

  async sendMessage(userMessage: string): Promise<void> {
    this.logger.info('Received user message', { message: userMessage });
    console.log('\n🧑 User:', userMessage);

    // Add user message to CLI history
    const userMsg: any = {
      role: 'user',
      content: [{ text: userMessage }]
    };
    this.cliConversationHistory.push(userMsg);

    await this.processConversationTurn(undefined, this.cliConversationHistory);
  }

  /**
   * Public method to process a user message with custom callbacks
   */
  async processMessageWithCallbacks(userMessage: string, callbacks: StreamingCallbacks, conversationHistory?: any[]): Promise<void> {
    this.logger.info('Processing message with callbacks', { message: userMessage });

    // For server mode (with callbacks), use passed conversation history as-is
    // For CLI mode (no callbacks), manage local history
    if (callbacks && conversationHistory) {
      // Server mode: completely stateless, use client's history
      await this.processConversationTurn(callbacks, conversationHistory);
    } else {
      // CLI mode: manage local history
      const userMsg: any = {
        role: 'user',
        content: [{ text: userMessage }]
      };
      this.cliConversationHistory.push(userMsg);
      await this.processConversationTurn(callbacks, this.cliConversationHistory);
    }
  }

  private async processConversationTurn(callbacks?: StreamingCallbacks, conversationHistory: any[] = []): Promise<void> {
    const tools = this.getAllTools();
    this.logger.debug('Available tools for request', { toolCount: tools.length });

    try {
      const command = new ConverseStreamCommand({
        modelId: 'us.anthropic.claude-sonnet-4-20250514-v1:0',
        system: [{ text: this.systemPrompt }],
        messages: conversationHistory,
        toolConfig: tools.length > 0 ? { tools: tools } : undefined,
        inferenceConfig: {
          maxTokens: 4000,
          temperature: 0.3
        }
      });

      this.logger.debug('Sending request to Bedrock', {
        modelId: 'us.anthropic.claude-sonnet-4-20250514-v1:0',
        messageCount: conversationHistory.length,
        toolCount: tools.length
      });

      const response = await this.bedrockClient.send(command);
      
      let assistantMessage: any = {
        role: 'assistant',
        content: []
      };

      let currentText = '';
      const toolUses: any[] = [];
      const toolResults: any[] = [];
      
      // Tool parameter accumulation for streaming
      const activeToolStreams: Record<string, {
        toolUseId: string;
        name: string;
        input: Record<string, any>;
        inputBuffer: string; // Buffer for JSON string fragments
        isComplete: boolean;
      }> = {};

      if (response.stream) {
        if (!callbacks) {
          console.log('\n🤖 Jarvis:');
        }
        
        for await (const chunk of response.stream) {
          // Handle text streaming
          if (chunk.contentBlockStart && 'text' in chunk.contentBlockStart.start) {
            const text = (chunk.contentBlockStart.start as any).text as string;
            if (callbacks?.onTextStart) {
              callbacks.onTextStart(text);
            } else {
              process.stdout.write(text);
            }
            currentText += text;
          }
          
          if (chunk.contentBlockDelta?.delta?.text) {
            const deltaText = chunk.contentBlockDelta.delta.text;
            if (callbacks?.onTextDelta) {
              callbacks.onTextDelta(deltaText);
            } else {
              process.stdout.write(deltaText);
            }
            currentText += deltaText;
          }

          // Handle tool use start - initialize parameter accumulation
          if (chunk.contentBlockStart && 'toolUse' in chunk.contentBlockStart.start) {
            const toolUse = (chunk.contentBlockStart.start as any).toolUse;
            const blockIndex = chunk.contentBlockStart.contentBlockIndex || 0;
            
            this.logger.toolParameterDebug('TOOL_USE_STARTED', toolUse.name, {
              toolUseId: toolUse.toolUseId,
              blockIndex,
              initialInput: toolUse.input
            });
            
            // Initialize tool stream accumulator
            activeToolStreams[blockIndex] = {
              toolUseId: toolUse.toolUseId,
              name: toolUse.name,
              input: toolUse.input || {},
              inputBuffer: '', // Initialize empty buffer for JSON string fragments
              isComplete: false
            };
            
            if (callbacks?.onToolUseStart) {
              callbacks.onToolUseStart(toolUse.name, toolUse.toolUseId, toolUse.input || {});
            } else {
              console.log(`\n🔧 Using tool: ${toolUse.name}`);
            }
          }

          // Handle tool parameter streaming - accumulate parameters
          if (chunk.contentBlockDelta?.delta?.toolUse?.input) {
            const blockIndex = chunk.contentBlockDelta.contentBlockIndex || 0;
            const deltaInput = chunk.contentBlockDelta.delta.toolUse.input;
            
            if (activeToolStreams[blockIndex]) {
              // Buffer JSON string fragments or merge objects
              if (typeof deltaInput === 'string') {
                // Accumulate JSON string fragments
                activeToolStreams[blockIndex].inputBuffer += deltaInput;
                
                this.logger.toolParameterDebug('JSON_BUFFER_STREAMING', activeToolStreams[blockIndex].name, {
                  blockIndex,
                  deltaInput,
                  bufferLength: activeToolStreams[blockIndex].inputBuffer.length,
                  currentBuffer: activeToolStreams[blockIndex].inputBuffer.substring(0, 100) + (activeToolStreams[blockIndex].inputBuffer.length > 100 ? '...' : '')
                });
              } else {
                // Handle object case (initial state or pre-parsed input)
                Object.assign(activeToolStreams[blockIndex].input, deltaInput);
                
                this.logger.toolParameterDebug('OBJECT_MERGE_STREAMING', activeToolStreams[blockIndex].name, {
                  blockIndex,
                  deltaInput,
                  currentInput: activeToolStreams[blockIndex].input,
                  parameterCount: Object.keys(activeToolStreams[blockIndex].input).length
                });
              }
            }
          }

          // Handle tool use completion - execute with complete parameters
          if (chunk.contentBlockStop) {
            const blockIndex = chunk.contentBlockStop.contentBlockIndex || 0;
            
            if (activeToolStreams[blockIndex] && !activeToolStreams[blockIndex].isComplete) {
              const toolStream = activeToolStreams[blockIndex];
              toolStream.isComplete = true;
              
              // Parse buffered JSON if available
              if (toolStream.inputBuffer.trim()) {
                try {
                  const parsedInput = JSON.parse(toolStream.inputBuffer);
                  if (typeof parsedInput === 'object' && parsedInput !== null) {
                    Object.assign(toolStream.input, parsedInput);
                    
                    this.logger.toolParameterDebug('JSON_PARSED_SUCCESS', toolStream.name, {
                      blockIndex,
                      rawBuffer: toolStream.inputBuffer,
                      parsedInput,
                      finalInput: toolStream.input
                    });
                  }
                } catch (error) {
                  const errorMessage = error instanceof Error ? error.message : String(error);
                  this.logger.toolParameterDebug('JSON_PARSE_ERROR', toolStream.name, {
                    blockIndex,
                    rawBuffer: toolStream.inputBuffer,
                    error: errorMessage,
                    fallbackInput: toolStream.input
                  });
                  
                  // Continue with existing input on parse error
                  if (callbacks?.onError) {
                    callbacks.onError(`JSON parse error for ${toolStream.name}: ${errorMessage}`);
                  } else {
                    console.log(`⚠️ JSON parse error for ${toolStream.name}: ${errorMessage}`);
                  }
                }
              }
              
              this.logger.toolParameterDebug('TOOL_USE_COMPLETED', toolStream.name, {
                toolUseId: toolStream.toolUseId,
                blockIndex,
                finalInput: toolStream.input,
                finalParameterCount: Object.keys(toolStream.input).length,
                hadBufferedInput: toolStream.inputBuffer.length > 0
              });
              
              // Store tool use for assistant message (without results)
              toolUses.push({
                toolUse: {
                  toolUseId: toolStream.toolUseId,
                  name: toolStream.name,
                  input: toolStream.input
                }
              });

              // Execute tool with complete parameters
              const [serverName, actualToolName] = toolStream.name.split('__');
              const mcpClient = this.mcpClients[serverName];
              
              if (mcpClient) {
                try {
                  const toolResult = await mcpClient.executeTool(actualToolName, toolStream.input);
                  this.logger.info(`Tool execution successful`, {
                    toolName: toolStream.name,
                    serverName,
                    actualToolName,
                    input: toolStream.input,
                    result: toolResult
                  });
                  
                  if (callbacks?.onToolResult) {
                    callbacks.onToolResult(toolStream.name, toolStream.toolUseId, toolResult);
                  } else {
                    console.log(`✅ Tool result: ${JSON.stringify(toolResult, null, 2)}`);
                  }
                  
                  // Truncate tool result to prevent API input size errors
                  const truncatedResult = truncateToolResult(toolResult);
                  toolResults.push({
                    toolResult: {
                      toolUseId: toolStream.toolUseId,
                      content: [{ text: truncatedResult }]
                    }
                  });
                } catch (error) {
                  const errorMessage = error instanceof Error ? error.message : String(error);
                  this.logger.error(`Tool execution failed`, {
                    toolName: toolStream.name,
                    serverName,
                    actualToolName,
                    input: toolStream.input,
                    error: errorMessage
                  });
                  
                  if (callbacks?.onToolError) {
                    callbacks.onToolError(toolStream.name, toolStream.toolUseId, errorMessage);
                  } else {
                    console.log(`❌ Tool error: ${errorMessage}`);
                  }
                  
                  toolResults.push({
                    toolResult: {
                      toolUseId: toolStream.toolUseId,
                      content: [{ text: `Error executing tool: ${errorMessage}` }]
                    }
                  });
                }
              } else {
                const errorMessage = `MCP client not found for server ${serverName}`;
                this.logger.error(errorMessage, { serverName, toolName: toolStream.name });
                
                if (callbacks?.onToolError) {
                  callbacks.onToolError(toolStream.name, toolStream.toolUseId, errorMessage);
                } else {
                  console.log(`❌ Tool error: ${errorMessage}`);
                }
                
                toolResults.push({
                  toolResult: {
                    toolUseId: toolStream.toolUseId,
                    content: [{ text: `Error: ${errorMessage}` }]
                  }
                });
              }
            }
          }
        }

        // Add text content to assistant message
        if (currentText) {
          assistantMessage.content.push({ text: currentText });
        }

        // Add tool uses to assistant message (but not tool results)
        assistantMessage.content.push(...toolUses);

        // If there were tool results, create a follow-up conversation turn
        if (toolResults.length > 0) {
          // Add tool results as a user message to continue the conversation
          const toolResultsMessage: any = {
            role: 'user',
            content: toolResults
          };

          // Create updated conversation history with assistant message and tool results
          const updatedHistory = [...conversationHistory, assistantMessage, toolResultsMessage];

          // Update CLI history only if in CLI mode (no callbacks)
          if (!callbacks) {
            this.cliConversationHistory.push(assistantMessage, toolResultsMessage);
          }

          this.logger.info('Tool results added, continuing conversation', {
            toolResultCount: toolResults.length
          });

          // Process another conversation turn with the tool results
          await this.processConversationTurn(callbacks, updatedHistory);
        } else {
          // Update CLI history with final assistant message only if in CLI mode
          if (!callbacks) {
            this.cliConversationHistory.push(assistantMessage);
          }

          this.logger.info('Response completed', {
            conversationLength: conversationHistory.length,
            responseLength: currentText.length
          });

          if (callbacks?.onTurnComplete) {
            callbacks.onTurnComplete();
          } else {
            console.log('\n');
          }
        }
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.logger.error('Error in processConversationTurn', { error: errorMessage });
      
      if (callbacks?.onError) {
        callbacks.onError(errorMessage);
      } else {
        console.error('Error:', errorMessage);
      }
    }
  }

  async startInteractiveMode(): Promise<void> {
    this.logger.info('Starting interactive mode');
    console.log('\n🎯 Jarvis Agent is ready! Type your messages (or "quit" to exit)');
    
    const readline = require('readline');
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });

    const askQuestion = (): Promise<string> => {
      return new Promise((resolve) => {
        rl.question('\n> ', resolve);
      });
    };

    while (true) {
      try {
        const userInput = await askQuestion();
        
        if (userInput.toLowerCase().trim() === 'quit') {
          this.logger.info('User requested quit');
          break;
        }

        if (userInput.trim()) {
          await this.sendMessage(userInput);
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        this.logger.error('Error in interactive mode', { error: errorMessage });
        console.error('Error:', errorMessage);
      }
    }

    rl.close();
    this.logger.info('Interactive mode ended');
  }

  cleanup(): void {
    this.logger.info('Cleaning up connections');
    for (const client of Object.values(this.mcpClients)) {
      client.disconnect();
    }
    this.logger.info('Cleanup completed');
  }
}