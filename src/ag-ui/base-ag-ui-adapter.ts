/**
 * Base AG UI Adapter
 *
 * Agent-agnostic adapter that bridges any agent implementing BaseAgent interface
 * with the AG UI protocol using official AG UI types.
 * This adapter converts agent interactions into AG UI compliant events and messages.
 */

import { v4 as uuidv4 } from "uuid";
import { readFileSync, existsSync } from "fs";
import { resolve } from "path";
import { Observable } from "rxjs";
import {
  EventType,
  BaseEvent,
  TextMessageStartEvent,
  TextMessageContentEvent,
  TextMessageEndEvent,
  RunStartedEvent,
  RunFinishedEvent,
  RunErrorEvent,
  ToolCallStartEvent,
  ToolCallArgsEvent,
  ToolCallEndEvent,
  ToolCallResultEvent,
  StateSnapshotEvent,
  StateDeltaEvent,
  StepStartedEvent,
  StepFinishedEvent,
  Message,
  Tool,
  RunAgentInput,
  State,
} from "@ag-ui/core";
import { BaseAgent, StreamingCallbacks } from "../agents/base-agent";
import { MCPServerConfig } from "../types/mcp-types";
import { Logger } from "../utils/logger";
import { AGUIAuditLogger } from "../utils/ag-ui-audit-logger";

export interface BaseAGUIConfig {
  port?: number;
  host?: string;
  cors?: {
    origins: string[];
    credentials: boolean;
  };
  mcpConfigs?: Record<string, MCPServerConfig>;
}

export class BaseAGUIAdapter {
  private agent: BaseAgent;
  private logger: Logger;
  private auditLogger?: AGUIAuditLogger;

  // State management for AG UI events
  private stateHistory: State[] = [];
  // Accumulate state deltas during message streaming
  private pendingStateDeltas: any[] = [];

  constructor(
    agent: BaseAgent,
    config: BaseAGUIConfig = {},
    logger?: Logger,
    auditLogger?: AGUIAuditLogger
  ) {
    this.agent = agent;
    this.logger = logger || new Logger();
    this.auditLogger = auditLogger;
  }

  async initialize(
    mcpConfigs: Record<string, MCPServerConfig> = {}
  ): Promise<void> {
    const agentType = this.agent.getAgentType();
    this.logger.info(`Initializing ${agentType} AG UI Adapter`);

    // Check for custom system prompt file path from environment variable
    let customSystemPrompt: string | undefined = undefined;
    const systemPromptPath = process.env.SYSTEM_PROMPT;
    if (systemPromptPath) {
      try {
        // Resolve relative paths relative to the project root (where package.json is)
        const resolvedPath = resolve(process.cwd(), systemPromptPath);
        if (existsSync(resolvedPath)) {
          customSystemPrompt = readFileSync(resolvedPath, "utf-8");
          this.logger.info("Using custom system prompt from file", {
            originalPath: systemPromptPath,
            resolvedPath: resolvedPath,
          });
        } else {
          this.logger.warn("System prompt file not found", {
            originalPath: systemPromptPath,
            resolvedPath: resolvedPath,
          });
        }
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        this.logger.error("Failed to load system prompt file", {
          path: systemPromptPath,
          error: errorMessage,
        });
      }
    }

    // Initialize agent with MCP configs and custom system prompt
    await this.agent.initialize(mcpConfigs, customSystemPrompt);

    this.logger.info(`${agentType} AG UI Adapter initialized`);
  }

  /**
   * Run agent using AG UI protocol RunAgentInput with streaming content
   */
  async runAgent(input: RunAgentInput): Promise<Observable<BaseEvent>> {
    const agentType = this.agent.getAgentType();

    // Set logger context for correlation
    this.logger.setContext(input.threadId, input.runId);

    this.logger.info(`Running ${agentType} agent with AG UI input`, {
      threadId: input.threadId,
      runId: input.runId,
      messageCount: input.messages.length,
      toolCount: input.tools?.length || 0,
    });

    return new Observable<BaseEvent>((observer) => {
      // Start audit logging for this request
      this.auditLogger?.startRequest(input.threadId, input.runId);

      this.processAgentRequestWithEvents(input, observer).catch((error) => {
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        this.logger.error("Error in processAgentRequestWithEvents", {
          error: errorMessage,
          threadId: input.threadId,
          runId: input.runId,
          agentType,
        });

        // Emit error event and complete
        const errorEvent = {
          type: EventType.RUN_ERROR,
          message: errorMessage,
          code: "AGENT_ERROR",
          timestamp: Date.now(),
        } as RunErrorEvent;

        observer.next(errorEvent);
        this.auditLogger?.logEvent(input.threadId, input.runId, errorEvent);
        this.auditLogger?.endRequest(
          input.threadId,
          input.runId,
          "error",
          errorMessage
        );

        observer.complete();
      });
    });
  }

  /**
   * Emit event to observer and audit logger
   */
  private emitAndAuditEvent(
    event: BaseEvent,
    observer: any,
    threadId: string,
    runId: string
  ): void {
    observer.next(event);
    this.auditLogger?.logEvent(threadId, runId, event);
  }

  /**
   * Process agent request and emit events through observer
   */
  private async processAgentRequestWithEvents(
    input: RunAgentInput,
    observer: any
  ): Promise<void> {
    const agentType = this.agent.getAgentType();

    // Emit run started event
    this.emitAndAuditEvent(
      {
        type: EventType.RUN_STARTED,
        threadId: input.threadId,
        runId: input.runId,
        timestamp: Date.now(),
      } as RunStartedEvent,
      observer,
      input.threadId,
      input.runId
    );

    this.stateHistory.push(input.state || {});

    this.emitAndAuditEvent(
      {
        type: EventType.STATE_SNAPSHOT,
        snapshot: input.state || {},
        timestamp: Date.now(),
      } as StateSnapshotEvent,
      observer,
      input.threadId,
      input.runId
    );

    try {
      // Validate that we have messages
      if (!input.messages || input.messages.length === 0) {
        throw new Error("No messages found in input");
      }

      // Emit text message start event
      const messageId = uuidv4();
      this.emitAndAuditEvent(
        {
          type: EventType.TEXT_MESSAGE_START,
          messageId,
          role: "assistant",
          timestamp: Date.now(),
        } as TextMessageStartEvent,
        observer,
        input.threadId,
        input.runId
      );

      // Run the agent with streaming integration
      // Pass the full messages array instead of extracting text
      await this.runAgentWithStreamingEvents(
        input.messages,  // Pass full messages array
        messageId,
        observer,
        input.threadId,
        input.runId,
        input  // Pass the full input
      );

      // Emit thinking end after agent reasoning
      // observer.next({
      //   type: EventType.THINKING_TEXT_MESSAGE_END,
      //   timestamp: Date.now()
      // });

      // observer.next({
      //   type: EventType.THINKING_END,
      //   timestamp: Date.now()
      // } as ThinkingEndEvent);

      // Don't emit STEP_FINISHED for agent_processing - we removed the corresponding STEP_STARTED
      // this.emitAndAuditEvent({
      //   type: EventType.STEP_FINISHED,
      //   stepName: `${agentType}_agent_processing`,
      //   timestamp: Date.now()
      // } as StepFinishedEvent, observer, input.threadId, input.runId);

      // Emit text message end FIRST to close the message stream
      this.emitAndAuditEvent(
        {
          type: EventType.TEXT_MESSAGE_END,
          messageId,
          timestamp: Date.now(),
        } as TextMessageEndEvent,
        observer,
        input.threadId,
        input.runId
      );

      // Emit run finished event
      this.emitAndAuditEvent(
        {
          type: EventType.RUN_FINISHED,
          threadId: input.threadId,
          runId: input.runId,
          timestamp: Date.now(),
        } as RunFinishedEvent,
        observer,
        input.threadId,
        input.runId
      );

      // End audit logging for successful completion
      this.auditLogger?.endRequest(input.threadId, input.runId, "success");

      // Complete the stream
      observer.complete();
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      const errorStack = error instanceof Error ? error.stack : undefined;
      this.logger.error("Error running agent", {
        error: errorMessage,
        stack: errorStack,
        threadId: input.threadId,
        runId: input.runId,
      });

      // Emit run error event
      this.emitAndAuditEvent(
        {
          type: EventType.RUN_ERROR,
          message: errorMessage,
          code: "AGENT_ERROR",
          timestamp: Date.now(),
        } as RunErrorEvent,
        observer,
        input.threadId,
        input.runId
      );

      // End audit logging for error
      this.auditLogger?.endRequest(
        input.threadId,
        input.runId,
        "error",
        errorMessage
      );

      // Complete the stream
      observer.complete();
    }
  }

  /**
   * Detect and parse PPL query from text content
   */
  private detectPPLQuery(text: string): any | null {
    try {
      // Check for STATE_DELTA JSON block with PPL query
      const stateDeltaMatch = text.match(/\{\s*"type"\s*:\s*"STATE_DELTA"[\s\S]*?"ppl_query"[\s\S]*?\}[\s\S]*?\}/);
      if (stateDeltaMatch) {
        // Extract the JSON object
        const jsonStr = stateDeltaMatch[0];
        const parsed = JSON.parse(jsonStr);
        if (parsed.delta?.ppl_query) {
          this.logger.info('PPL query detected in STATE_DELTA format', {
            query: parsed.delta.ppl_query.query,
            dataset: parsed.delta.ppl_query.dataset
          });
          return parsed.delta.ppl_query;
        }
      }

      // Alternative: Check for PPL query in code blocks
      const pplBlockMatch = text.match(/```ppl\s*([\s\S]*?)```/i);
      if (pplBlockMatch) {
        const query = pplBlockMatch[1].trim();
        if (query) {
          this.logger.info('PPL query detected in code block', { query });
          // Try to extract dataset from query (source=dataset pattern)
          const datasetMatch = query.match(/source\s*=\s*([^\s|]+)/);
          const dataset = datasetMatch ? datasetMatch[1] : 'unknown';

          return {
            query: query,
            description: 'PPL query from code block',
            dataset: dataset,
            timestamp: new Date().toISOString()
          };
        }
      }

      // Check for inline PPL patterns
      const inlinePatterns = [
        /source\s*=\s*[^\s|]+.*?\|.*?(?:where|stats|fields|sort|head|tail)/i,
        /search\s+source\s*=\s*[^\s|]+/i
      ];

      for (const pattern of inlinePatterns) {
        const match = text.match(pattern);
        if (match) {
          const query = match[0];
          const datasetMatch = query.match(/source\s*=\s*([^\s|]+)/);
          const dataset = datasetMatch ? datasetMatch[1] : 'unknown';

          this.logger.info('Inline PPL query detected', { query, dataset });
          return {
            query: query,
            description: 'Inline PPL query',
            dataset: dataset,
            timestamp: new Date().toISOString()
          };
        }
      }
    } catch (error) {
      this.logger.debug('Error detecting PPL query', { error: error.message });
    }

    return null;
  }

  /**
   * Run agent with streaming content emission through observer
   */
  private async runAgentWithStreamingEvents(
    messages: any[],  // Changed from userMessage: string to messages array
    messageId: string,
    observer: any,
    threadId: string,
    runId: string,
    fullInput?: RunAgentInput  // Add parameter for full input
  ): Promise<void> {
    const agentType = this.agent.getAgentType();
    // Track accumulated text to detect multi-line PPL queries
    let accumulatedText = '';

    try {
      // Create callbacks to convert agent events to AG UI events
      const callbacks: StreamingCallbacks = {
        onTextStart: (text: string) => {
          accumulatedText = text; // Start accumulating text
          this.emitAndAuditEvent(
            {
              type: EventType.TEXT_MESSAGE_CONTENT,
              messageId,
              delta: text,
              timestamp: Date.now(),
            } as TextMessageContentEvent,
            observer,
            threadId,
            runId
          );
        },
        onTextDelta: (delta: string) => {
          accumulatedText += delta; // Continue accumulating text

          // Check for PPL query in accumulated text (for multi-line queries)
          const pplQuery = this.detectPPLQuery(accumulatedText);
          if (pplQuery) {
            // Check if we already have this query in pending deltas
            const existingQuery = this.pendingStateDeltas.find(
              d => d.ppl_query?.query === pplQuery.query
            );

            if (!existingQuery) {
              // Add to pending state deltas
              this.pendingStateDeltas.push({
                ppl_query: pplQuery
              });
              this.logger.info('PPL query added to pending state deltas', {
                query: pplQuery.query,
                pendingCount: this.pendingStateDeltas.length
              });
            }
          }

          this.emitAndAuditEvent(
            {
              type: EventType.TEXT_MESSAGE_CONTENT,
              messageId,
              delta,
              timestamp: Date.now(),
            } as TextMessageContentEvent,
            observer,
            threadId,
            runId
          );
        },
        onToolUseStart: (toolName: string, toolUseId: string, input: any) => {
          // Emit proper TOOL_CALL_START event
          const actualToolName = toolName.split("__")[1] || toolName;

          // Don't emit STEP_STARTED during TEXT_MESSAGE - it causes event ordering issues
          // this.emitAndAuditEvent({
          //   type: EventType.STEP_STARTED,
          //   stepName: `tool_execution_${actualToolName}`,
          //   timestamp: Date.now()
          // } as StepStartedEvent, observer, threadId, runId);

          // Emit thinking start for tool decision
          // this.emitAndAuditEvent({
          //   type: EventType.THINKING_START,
          //   timestamp: Date.now()
          // } as ThinkingStartEvent, observer, threadId, runId);

          // this.emitAndAuditEvent({
          //   type: EventType.THINKING_TEXT_MESSAGE_START,
          //   title: `Executing ${actualToolName}`,
          //   timestamp: Date.now()
          // }, observer, threadId, runId);

          this.emitAndAuditEvent(
            {
              type: EventType.TOOL_CALL_START,
              toolCallId: toolUseId,
              toolCallName: actualToolName,
              timestamp: Date.now(),
            } as ToolCallStartEvent,
            observer,
            threadId,
            runId
          );

          // Emit tool arguments as JSON string
          this.emitAndAuditEvent(
            {
              type: EventType.TOOL_CALL_ARGS,
              toolCallId: toolUseId,
              delta: JSON.stringify(input),
              timestamp: Date.now(),
            } as ToolCallArgsEvent,
            observer,
            threadId,
            runId
          );

          // Also add a text message for visibility in the chat
          // this.emitAndAuditEvent({
          //   type: EventType.TEXT_MESSAGE_CONTENT,
          //   messageId,
          //   delta: `\n\n🔧 Using tool: ${actualToolName}`,
          //   timestamp: Date.now()
          // } as TextMessageContentEvent, observer, threadId, runId);
        },
        onToolResult: (toolName: string, toolUseId: string, result: any) => {
          const actualToolName = toolName.split("__")[1] || toolName;

          // Emit TOOL_CALL_END event FIRST
          this.emitAndAuditEvent(
            {
              type: EventType.TOOL_CALL_END,
              toolCallId: toolUseId,
              timestamp: Date.now(),
            } as ToolCallEndEvent,
            observer,
            threadId,
            runId
          );

          // Then emit TOOL_CALL_RESULT event
          this.emitAndAuditEvent(
            {
              type: EventType.TOOL_CALL_RESULT,
              toolCallId: toolUseId,
              content: JSON.stringify(result),
              messageId: uuidv4(),
              timestamp: Date.now(),
            } as ToolCallResultEvent,
            observer,
            threadId,
            runId
          );

          // Track state changes after tool completion using JSON Patch format
          // NOTE: We cannot emit STATE_DELTA during TEXT_MESSAGE streaming
          // It will be accumulated and emitted before TEXT_MESSAGE_END
          // const stateDelta = [{
          //   op: 'add',
          //   path: `/toolExecutions/${toolUseId}`,
          //   value: {
          //     toolName: actualToolName,
          //     timestamp: Date.now()
          //   }
          // }];

          // this.emitAndAuditEvent({
          //   type: EventType.STATE_DELTA,
          //   delta: stateDelta,
          //   timestamp: Date.now()
          // } as StateDeltaEvent, observer, threadId, runId);

          // Emit thinking end for tool completion
          // this.emitAndAuditEvent({
          //   type: EventType.THINKING_TEXT_MESSAGE_END,
          //   timestamp: Date.now()
          // }, observer, threadId, runId);

          // this.emitAndAuditEvent({
          //   type: EventType.THINKING_END,
          //   timestamp: Date.now()
          // } as ThinkingEndEvent, observer, threadId, runId);

          // Don't emit STEP_FINISHED during TEXT_MESSAGE - it causes event ordering issues
          // this.emitAndAuditEvent({
          //   type: EventType.STEP_FINISHED,
          //   stepName: `tool_execution_${actualToolName}`,
          //   timestamp: Date.now()
          // } as StepFinishedEvent, observer, threadId, runId);

          // Also add a text message for visibility in the chat
          // const resultText = `\n\n✅ Tool ${actualToolName} result:\n${JSON.stringify(result, null, 2)}`;
          // this.emitAndAuditEvent({
          //   type: EventType.TEXT_MESSAGE_CONTENT,
          //   messageId,
          //   delta: resultText,
          //   timestamp: Date.now()
          // } as TextMessageContentEvent, observer, threadId, runId);
        },
        onToolError: (toolName: string, toolUseId: string, error: string) => {
          // Emit RUN_ERROR for tool failures
          const actualToolName = toolName.split("__")[1] || toolName;

          // Emit thinking end for failed tool execution
          // this.emitAndAuditEvent({
          //   type: EventType.THINKING_TEXT_MESSAGE_END,
          //   timestamp: Date.now()
          // }, observer, threadId, runId);

          // this.emitAndAuditEvent({
          //   type: EventType.THINKING_END,
          //   timestamp: Date.now()
          // } as ThinkingEndEvent, observer, threadId, runId);

          // Don't emit STEP_FINISHED during TEXT_MESSAGE - it causes event ordering issues
          // this.emitAndAuditEvent({
          //   type: EventType.STEP_FINISHED,
          //   stepName: `tool_execution_${actualToolName}`,
          //   timestamp: Date.now()
          // } as StepFinishedEvent, observer, threadId, runId);

          this.emitAndAuditEvent(
            {
              type: EventType.RUN_ERROR,
              message: `Tool ${actualToolName} failed: ${error}`,
              code: "TOOL_ERROR",
              timestamp: Date.now(),
            } as RunErrorEvent,
            observer,
            threadId,
            runId
          );

          // Also add a text message for visibility in the chat
          const errorText = `\n\n❌ Tool ${actualToolName} error: ${error}`;
          this.emitAndAuditEvent(
            {
              type: EventType.TEXT_MESSAGE_CONTENT,
              messageId,
              delta: errorText,
              timestamp: Date.now(),
            } as TextMessageContentEvent,
            observer,
            threadId,
            runId
          );
        },
        onTurnComplete: () => {
          // Turn completed - emit any pending state deltas before message ends
          if (this.pendingStateDeltas.length > 0) {
            this.logger.info('Emitting pending STATE_DELTA events', {
              count: this.pendingStateDeltas.length
            });

            // Combine all pending deltas into one
            const combinedDelta: any = {};
            for (const delta of this.pendingStateDeltas) {
              Object.assign(combinedDelta, delta);
            }

            // Emit STATE_DELTA event
            this.emitAndAuditEvent(
              {
                type: EventType.STATE_DELTA,
                delta: combinedDelta,
                timestamp: Date.now(),
              } as StateDeltaEvent,
              observer,
              threadId,
              runId
            );

            // Clear pending deltas
            this.pendingStateDeltas = [];
          }
        },
        onError: (error: string) => {
          // Emit error as text content
          this.emitAndAuditEvent(
            {
              type: EventType.TEXT_MESSAGE_CONTENT,
              messageId,
              delta: `\n\nError: ${error}`,
              timestamp: Date.now(),
            } as TextMessageContentEvent,
            observer,
            threadId,
            runId
          );
        },
      };

      // Use agent's callback-based processing
      // Pass the messages array directly as the first parameter
      // Pass additional inputs if agent supports it (check parameter count)
      // Extract modelId from forwardedProps if it exists
      if (this.agent.processMessageWithCallbacks.length >= 3) {
        await this.agent.processMessageWithCallbacks(messages, callbacks, {
          state: fullInput?.state,
          context: fullInput?.context,
          tools: fullInput?.tools,  // Pass client tools from AG UI
          threadId: fullInput?.threadId,
          runId: fullInput?.runId,
          modelId: fullInput?.forwardedProps?.modelId  // Extract modelId from forwardedProps
        });
      } else {
        await this.agent.processMessageWithCallbacks(messages, callbacks);
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      this.logger.error("Error in agent streaming", {
        error: errorMessage,
        messageId,
        agentType,
      });

      // Emit error as text content
      this.emitAndAuditEvent(
        {
          type: EventType.TEXT_MESSAGE_CONTENT,
          messageId,
          delta: `Error: ${errorMessage}`,
          timestamp: Date.now(),
        } as TextMessageContentEvent,
        observer,
        threadId,
        runId
      );
    }
  }

  /**
   * Get available tools in AG UI format
   */
  async getTools(): Promise<Tool[]> {
    const agentTools = this.agent.getAllTools();

    return agentTools.map((tool) => ({
      name: tool.toolSpec.name,
      description: tool.toolSpec.description,
      parameters: tool.toolSpec.inputSchema.json,
    }));
  }

  /**
   * Cleanup resources
   */
  cleanup(): void {
    this.agent.cleanup();
    const agentType = this.agent.getAgentType();
    this.logger.info(`${agentType} AG UI Adapter cleanup completed`);
  }
}
