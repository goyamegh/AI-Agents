import { MCPServerConfig } from '../types/mcp-types';

export interface StreamingCallbacks {
  onTextStart?: (text: string) => void;
  onTextDelta?: (delta: string) => void;
  onToolUseStart?: (toolName: string, toolUseId: string, input: any) => void;
  onToolResult?: (toolName: string, toolUseId: string, result: any) => void;
  onToolError?: (toolName: string, toolUseId: string, error: string) => void;
  onTurnComplete?: () => void;
  onError?: (error: string) => void;
}

export interface BaseAgent {
  /**
   * Initialize the agent with MCP server configurations
   */
  initialize(configs: Record<string, MCPServerConfig>, customSystemPrompt?: string): Promise<void>;

  /**
   * Process a user message with streaming callbacks
   */
  processMessageWithCallbacks(message: string, callbacks: StreamingCallbacks, conversationHistory?: any[]): Promise<void>;

  /**
   * Send a message and process response (for CLI mode)
   */
  sendMessage(message: string): Promise<void>;

  /**
   * Get all available tools from connected MCP servers
   */
  getAllTools(): any[];

  /**
   * Get the agent type identifier
   */
  getAgentType(): string;

  /**
   * Start interactive CLI mode
   */
  startInteractiveMode(): Promise<void>;

  /**
   * Clean up connections and resources
   */
  cleanup(): void;
}