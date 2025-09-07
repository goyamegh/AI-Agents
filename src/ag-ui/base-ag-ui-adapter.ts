/**
 * Base AG UI Adapter
 * 
 * Agent-agnostic adapter that bridges any agent implementing BaseAgent interface
 * with the AG UI protocol using official AG UI types.
 * This adapter converts agent interactions into AG UI compliant events and messages.
 */

import { v4 as uuidv4 } from 'uuid';
import { Observable } from 'rxjs';
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
  State
} from '@ag-ui/core';
import { BaseAgent, StreamingCallbacks } from '../agents/base-agent';
import { MCPServerConfig } from '../types/mcp-types';
import { Logger } from '../utils/logger';
import { AGUIAuditLogger } from '../utils/ag-ui-audit-logger';

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
  private runningThreads = new Map<string, {
    runId: string;
    messages: Message[];
    state: State;
    tools: Tool[];
  }>();
  
  // State management for AG UI events
  private conversationState: State = {};
  private stateHistory: State[] = [];

  constructor(agent: BaseAgent, config: BaseAGUIConfig = {}, logger?: Logger, auditLogger?: AGUIAuditLogger) {
    this.agent = agent;
    this.logger = logger || new Logger();
    this.auditLogger = auditLogger;
  }

  async initialize(mcpConfigs: Record<string, MCPServerConfig> = {}): Promise<void> {
    const agentType = this.agent.getAgentType();
    this.logger.info(`Initializing ${agentType} AG UI Adapter`);
    
    // Initialize agent with MCP configs
    await this.agent.initialize(mcpConfigs);
    
    this.logger.info(`${agentType} AG UI Adapter initialized`);
  }

  /**
   * Run agent using AG UI protocol RunAgentInput with streaming content
   */
  async runAgent(input: RunAgentInput): Promise<Observable<BaseEvent>> {
    const agentType = this.agent.getAgentType();
    this.logger.info(`Running ${agentType} agent with AG UI input`, {
      threadId: input.threadId,
      runId: input.runId,
      messageCount: input.messages.length,
      toolCount: input.tools?.length || 0
    });

    return new Observable<BaseEvent>((observer) => {
      // Start audit logging for this request
      this.auditLogger?.startRequest(input.threadId, input.runId);
      
      this.processAgentRequestWithEvents(input, observer)
        .catch(error => {
          const errorMessage = error instanceof Error ? error.message : String(error);
          this.logger.error('Error in processAgentRequestWithEvents', { 
            error: errorMessage,
            threadId: input.threadId,
            runId: input.runId,
            agentType
          });
          
          // Emit error event and complete
          const errorEvent = {
            type: EventType.RUN_ERROR,
            message: errorMessage,
            code: 'AGENT_ERROR',
            timestamp: Date.now()
          } as RunErrorEvent;
          
          observer.next(errorEvent);
          this.auditLogger?.logEvent(input.threadId, input.runId, errorEvent);
          this.auditLogger?.endRequest(input.threadId, input.runId, 'error', errorMessage);
          
          observer.complete();
        });
    });
  }

  /**
   * Emit event to observer and audit logger
   */
  private emitAndAuditEvent(event: BaseEvent, observer: any, threadId: string, runId: string): void {
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
    this.emitAndAuditEvent({
      type: EventType.RUN_STARTED,
      threadId: input.threadId,
      runId: input.runId,
      timestamp: Date.now()
    } as RunStartedEvent, observer, input.threadId, input.runId);

    // Emit initial state snapshot
    const initialState = {
      threadId: input.threadId,
      runId: input.runId,
      agentType,
      messageCount: input.messages.length,
      toolCount: input.tools?.length || 0,
      startTime: Date.now()
    };

    this.conversationState = initialState;
    this.stateHistory.push(initialState);

    observer.next({
      type: EventType.STATE_SNAPSHOT,
      snapshot: initialState,
      timestamp: Date.now()
    } as StateSnapshotEvent);

    try {
      // Extract the last user message text from AG UI format
      const lastUserMessage = input.messages.filter(msg => msg.role === 'user').pop();
      if (!lastUserMessage) {
        throw new Error('No user message found in input');
      }
      
      const messageText = Array.isArray(lastUserMessage.content) 
        ? lastUserMessage.content.find(item => item.type === 'text')?.text || ''
        : lastUserMessage.content || '';
        
      if (!messageText) {
        throw new Error('No text content found in user message');
      }

      // Emit step started for agent processing
      observer.next({
        type: EventType.STEP_STARTED,
        stepName: `${agentType}_agent_processing`,
        timestamp: Date.now()
      } as StepStartedEvent);

      // Emit thinking start for agent reasoning
      // observer.next({
      //   type: EventType.THINKING_START,
      //   timestamp: Date.now()
      // } as ThinkingStartEvent);

      // observer.next({
      //   type: EventType.THINKING_TEXT_MESSAGE_START,
      //   title: 'Processing user request',
      //   timestamp: Date.now()
      // });

      // Emit text message start event
      const messageId = uuidv4();
      observer.next({
        type: EventType.TEXT_MESSAGE_START,
        messageId,
        role: 'assistant',
        timestamp: Date.now()
      } as TextMessageStartEvent);

      // Run the agent with streaming integration
      await this.runAgentWithStreamingEvents(messageText, messageId, observer);

      // Emit thinking end after agent reasoning
      // observer.next({
      //   type: EventType.THINKING_TEXT_MESSAGE_END,
      //   timestamp: Date.now()
      // });

      // observer.next({
      //   type: EventType.THINKING_END,
      //   timestamp: Date.now()
      // } as ThinkingEndEvent);


            // Emit text message end
      observer.next({
        type: EventType.TEXT_MESSAGE_END,
        messageId,
        timestamp: Date.now()
      } as TextMessageEndEvent);

      // Emit step finished for agent processing
      observer.next({
        type: EventType.STEP_FINISHED,
        stepName: `${agentType}_agent_processing`,
        timestamp: Date.now()
      } as StepFinishedEvent);



      // Emit final state delta before run completion using JSON Patch format
      const finalStateDelta = [{
        op: 'add',
        path: '/conversationCompleted',
        value: {
          messageCount: this.conversationState.messageCount || 0,
          duration: Date.now() - (this.conversationState.startTime || Date.now()),
          timestamp: Date.now()
        }
      }];

      observer.next({
        type: EventType.STATE_DELTA,
        delta: finalStateDelta,
        timestamp: Date.now()
      } as StateDeltaEvent);

      // Emit run finished event
      this.emitAndAuditEvent({
        type: EventType.RUN_FINISHED,
        threadId: input.threadId,
        runId: input.runId,
        timestamp: Date.now()
      } as RunFinishedEvent, observer, input.threadId, input.runId);

      // End audit logging for successful completion
      this.auditLogger?.endRequest(input.threadId, input.runId, 'success');

      // Complete the stream
      observer.complete();

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      const errorStack = error instanceof Error ? error.stack : undefined;
      this.logger.error('Error running agent', { 
        error: errorMessage,
        stack: errorStack,
        threadId: input.threadId,
        runId: input.runId
      });
      
      // Emit run error event
      this.emitAndAuditEvent({
        type: EventType.RUN_ERROR,
        message: errorMessage,
        code: 'AGENT_ERROR',
        timestamp: Date.now()
      } as RunErrorEvent, observer, input.threadId, input.runId);

      // End audit logging for error
      this.auditLogger?.endRequest(input.threadId, input.runId, 'error', errorMessage);
      
      // Complete the stream
      observer.complete();
    }
  }

  /**
   * Run agent with streaming content emission through observer
   */
  private async runAgentWithStreamingEvents(
    userMessage: string, 
    messageId: string, 
    observer: any
  ): Promise<void> {
    const agentType = this.agent.getAgentType();
    try {
      // Create callbacks to convert agent events to AG UI events
      const callbacks: StreamingCallbacks = {
        onTextStart: (text: string) => {
          observer.next({
            type: EventType.TEXT_MESSAGE_CONTENT,
            messageId,
            delta: text,
            timestamp: Date.now()
          } as TextMessageContentEvent);
        },
        onTextDelta: (delta: string) => {
          observer.next({
            type: EventType.TEXT_MESSAGE_CONTENT,
            messageId,
            delta,
            timestamp: Date.now()
          } as TextMessageContentEvent);
        },
        onToolUseStart: (toolName: string, toolUseId: string, input: any) => {
          // Emit proper TOOL_CALL_START event
          const actualToolName = toolName.split('__')[1] || toolName;
          
          // Emit step started for tool execution
          observer.next({
            type: EventType.STEP_STARTED,
            stepName: `tool_execution_${actualToolName}`,
            timestamp: Date.now()
          } as StepStartedEvent);

          // Emit thinking start for tool decision
          // observer.next({
          //   type: EventType.THINKING_START,
          //   timestamp: Date.now()
          // } as ThinkingStartEvent);

          // observer.next({
          //   type: EventType.THINKING_TEXT_MESSAGE_START,
          //   title: `Executing ${actualToolName}`,
          //   timestamp: Date.now()
          // });
          
          observer.next({
            type: EventType.TOOL_CALL_START,
            toolCallId: toolUseId,
            toolCallName: actualToolName,
            timestamp: Date.now()
          } as ToolCallStartEvent);

          // Emit tool arguments as JSON string
          observer.next({
            type: EventType.TOOL_CALL_ARGS,
            toolCallId: toolUseId,
            delta: JSON.stringify(input),
            timestamp: Date.now()
          } as ToolCallArgsEvent);

          // Also add a text message for visibility in the chat
          observer.next({
            type: EventType.TEXT_MESSAGE_CONTENT,
            messageId,
            delta: `\n\n🔧 Using tool: ${actualToolName}`,
            timestamp: Date.now()
          } as TextMessageContentEvent);
        },
        onToolResult: (toolName: string, toolUseId: string, result: any) => {
          // Emit proper TOOL_CALL_RESULT event
          const actualToolName = toolName.split('__')[1] || toolName;
          observer.next({
            type: EventType.TOOL_CALL_RESULT,
            toolCallId: toolUseId,
            content: JSON.stringify(result),
            messageId: uuidv4(),
            timestamp: Date.now()
          } as ToolCallResultEvent);

          // Emit TOOL_CALL_END event
          observer.next({
            type: EventType.TOOL_CALL_END,
            toolCallId: toolUseId,
            timestamp: Date.now()
          } as ToolCallEndEvent);

          // Track state changes after tool completion using JSON Patch format
          const stateDelta = [{
            op: 'add',
            path: `/toolExecutions/${toolUseId}`,
            value: {
              toolName: actualToolName,
              timestamp: Date.now()
            }
          }];

          observer.next({
            type: EventType.STATE_DELTA,
            delta: stateDelta,
            timestamp: Date.now()
          } as StateDeltaEvent);

          // Emit thinking end for tool completion
          // observer.next({
          //   type: EventType.THINKING_TEXT_MESSAGE_END,
          //   timestamp: Date.now()
          // });

          // observer.next({
          //   type: EventType.THINKING_END,
          //   timestamp: Date.now()
          // } as ThinkingEndEvent);

          // Emit step finished for tool execution
          observer.next({
            type: EventType.STEP_FINISHED,
            stepName: `tool_execution_${actualToolName}`,
            timestamp: Date.now()
          } as StepFinishedEvent);

          // Also add a text message for visibility in the chat
          const resultText = `\n\n✅ Tool ${actualToolName} result:\n${JSON.stringify(result, null, 2)}`;
          observer.next({
            type: EventType.TEXT_MESSAGE_CONTENT,
            messageId,
            delta: resultText,
            timestamp: Date.now()
          } as TextMessageContentEvent);
        },
        onToolError: (toolName: string, toolUseId: string, error: string) => {
          // Emit RUN_ERROR for tool failures
          const actualToolName = toolName.split('__')[1] || toolName;
          
          // Emit thinking end for failed tool execution
          // observer.next({
          //   type: EventType.THINKING_TEXT_MESSAGE_END,
          //   timestamp: Date.now()
          // });

          // observer.next({
          //   type: EventType.THINKING_END,
          //   timestamp: Date.now()
          // } as ThinkingEndEvent);

          // Emit step finished for failed tool execution
          observer.next({
            type: EventType.STEP_FINISHED,
            stepName: `tool_execution_${actualToolName}`,
            timestamp: Date.now()
          } as StepFinishedEvent);
          
          observer.next({
            type: EventType.RUN_ERROR,
            message: `Tool ${actualToolName} failed: ${error}`,
            code: 'TOOL_ERROR',
            timestamp: Date.now()
          } as RunErrorEvent);

          // Also add a text message for visibility in the chat
          const errorText = `\n\n❌ Tool ${actualToolName} error: ${error}`;
          observer.next({
            type: EventType.TEXT_MESSAGE_CONTENT,
            messageId,
            delta: errorText,
            timestamp: Date.now()
          } as TextMessageContentEvent);
        },
        onTurnComplete: () => {
          // Turn completed - the calling method will handle message end events
        },
        onError: (error: string) => {
          // Emit error as text content
          observer.next({
            type: EventType.TEXT_MESSAGE_CONTENT,
            messageId,
            delta: `\n\nError: ${error}`,
            timestamp: Date.now()
          } as TextMessageContentEvent);
        }
      };

      // Use agent's callback-based processing
      await this.agent.processMessageWithCallbacks(userMessage, callbacks);
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.logger.error('Error in agent streaming', { 
        error: errorMessage,
        messageId,
        agentType
      });
      
      // Emit error as text content
      observer.next({
        type: EventType.TEXT_MESSAGE_CONTENT,
        messageId,
        delta: `Error: ${errorMessage}`,
        timestamp: Date.now()
      } as TextMessageContentEvent);
    }
  }


  /**
   * Get available tools in AG UI format
   */
  async getTools(): Promise<Tool[]> {
    const agentTools = this.agent.getAllTools();
    
    return agentTools.map(tool => ({
      name: tool.toolSpec.name,
      description: tool.toolSpec.description,
      parameters: tool.toolSpec.inputSchema.json
    }));
  }



  /**
   * Cleanup resources
   */
  cleanup(): void {
    this.agent.cleanup();
    this.runningThreads.clear();
    const agentType = this.agent.getAgentType();
    this.logger.info(`${agentType} AG UI Adapter cleanup completed`);
  }
}