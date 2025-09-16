/**
 * LangGraph AG UI Adapter
 * 
 * Specialized adapter for LangGraph agent that extends BaseAGUIAdapter
 * to provide graph-specific event emissions and state tracking.
 * 
 * Features:
 * - Graph node transition events
 * - Iteration tracking
 * - Tool loop visualization
 * - State graph debugging information
 */

import { Observable } from 'rxjs';
import {
  EventType,
  BaseEvent,
  StateSnapshotEvent,
  StateDeltaEvent,
  StepStartedEvent,
  StepFinishedEvent,
  TextMessageStartEvent,
  TextMessageEndEvent,
  RunAgentInput,
  State
} from '@ag-ui/core';
import { BaseAGUIAdapter, BaseAGUIConfig } from './base-ag-ui-adapter';
import { ReactAgent } from '../agents/langgraph/react-agent';
import { Logger } from '../utils/logger';
import { AGUIAuditLogger } from '../utils/ag-ui-audit-logger';
import { v4 as uuidv4 } from 'uuid';

export interface LangGraphState extends State {
  currentNode?: string;
  iterations?: number;
  maxIterations?: number;
  toolCallsPending?: number;
  toolCallsCompleted?: number;
  graphPath?: string[];
  nodeExecutions?: Record<string, number>;
}

export class LangGraphAGUIAdapter extends BaseAGUIAdapter {
  private reactAgent: ReactAgent;
  private graphState: LangGraphState = {};
  private activeSteps = new Set<string>();
  private textMessageActive = false;
  private pendingStepEvents: BaseEvent[] = [];

  // Dual text message state tracking
  private currentMessageId?: string;
  private hasToolCallsOccurred = false;
  private isSecondTextMessage = false;
  
  constructor(agent: ReactAgent, config: BaseAGUIConfig = {}, logger?: Logger, auditLogger?: AGUIAuditLogger) {
    super(agent, config, logger, auditLogger);
    this.reactAgent = agent;
  }

  /**
   * Override runAgent to add LangGraph-specific events
   */
  async runAgent(input: RunAgentInput): Promise<Observable<BaseEvent>> {
    const baseObservable = await super.runAgent(input);

    // Return a new observable that enhances base events with LangGraph specifics
    return new Observable<BaseEvent>((observer) => {
      // Initialize LangGraph state tracking
      this.graphState = {
        currentNode: 'START',
        iterations: 0,
        maxIterations: 5,
        toolCallsPending: 0,
        toolCallsCompleted: 0,
        graphPath: ['START'],
        nodeExecutions: {}
      };

      let runStartedEmitted = false;
      
      // Clear active steps at start and reset dual text message state
      this.activeSteps.clear();
      this.textMessageActive = false;
      this.pendingStepEvents = [];
      this.currentMessageId = undefined;
      this.hasToolCallsOccurred = false;
      this.isSecondTextMessage = false;

      // Subscribe to base events and enhance them
      const subscription = baseObservable.subscribe({
        next: (event: BaseEvent) => {
          // Track text message state
          if (event.type === EventType.TEXT_MESSAGE_START) {
            this.textMessageActive = true;
          } else if (event.type === EventType.TEXT_MESSAGE_END) {
            this.textMessageActive = false;

            // Forward the TEXT_MESSAGE_END event FIRST
            observer.next(event);

            // Then emit any pending NON-TOOL step events AFTER TEXT_MESSAGE_END
            // Tool events should flow through in real-time after text ends
            const nonToolEvents = this.pendingStepEvents.filter(e =>
              e.type !== EventType.TOOL_CALL_START &&
              e.type !== EventType.TOOL_CALL_END &&
              e.type !== EventType.TOOL_CALL_ARGS &&
              e.type !== EventType.TOOL_CALL_RESULT
            );

            for (const pendingEvent of nonToolEvents) {
              observer.next(pendingEvent);
            }

            // Clear only the non-tool events - tool events should have flowed through already
            this.pendingStepEvents = this.pendingStepEvents.filter(e =>
              e.type === EventType.TOOL_CALL_START ||
              e.type === EventType.TOOL_CALL_END ||
              e.type === EventType.TOOL_CALL_ARGS ||
              e.type === EventType.TOOL_CALL_RESULT
            );

            return; // Don't forward TEXT_MESSAGE_END again
          }
          // Emit LangGraph-specific state after RUN_STARTED
          if (event.type === EventType.RUN_STARTED && !runStartedEmitted) {
            runStartedEmitted = true;
            
            // Forward RUN_STARTED first
            observer.next(event);
            
            // Then emit LangGraph initial state
            observer.next({
              type: EventType.STATE_SNAPSHOT,
              snapshot: {
                ...this.graphState,
                graphType: 'LangGraph',
                currentNode: 'START',
                graphPath: ['START'],
                nodeExecutions: {},
                nodes: ['processInput', 'callModel', 'executeTools', 'generateResponse'],
                edges: {
                  'START': ['processInput'],
                  'processInput': ['callModel'],
                  'callModel': ['executeTools', 'generateResponse'],
                  'executeTools': ['callModel', 'generateResponse'],
                  'generateResponse': ['END']
                }
              },
              timestamp: Date.now()
            } as StateSnapshotEvent);
            
            return; // Don't forward the RUN_STARTED again
          }
          // Intercept and enhance specific events
          if (event.type === EventType.STEP_STARTED) {
            const stepEvent = event as StepStartedEvent;

            // Track graph node transitions
            // Don't emit node transitions during text message - causes event ordering issues
            // if (stepEvent.stepName?.includes('_agent_processing')) {
            //   this.emitNodeTransition(observer, 'processInput');
            // }
          } else if (event.type === EventType.TOOL_CALL_START) {
            // Handle the first tool call during active text message
            if (this.textMessageActive && !this.hasToolCallsOccurred) {
              // This is the first tool call - emit TEXT_MESSAGE_END for current message
              observer.next({
                type: EventType.TEXT_MESSAGE_END,
                messageId: this.currentMessageId,
                timestamp: Date.now()
              } as TextMessageEndEvent);

              this.textMessageActive = false;
              this.hasToolCallsOccurred = true;
            }

            // Track tool execution in graph state
            this.graphState.toolCallsPending = (this.graphState.toolCallsPending || 0) + 1;
            // Don't emit node transitions for tool events - they're handled during deferred processing
            // this.emitNodeTransition(observer, 'executeTools');
          } else if (event.type === EventType.TOOL_CALL_END) {
            // Update tool completion tracking
            this.graphState.toolCallsCompleted = (this.graphState.toolCallsCompleted || 0) + 1;
            this.graphState.toolCallsPending = Math.max(0, (this.graphState.toolCallsPending || 0) - 1);

            // Don't emit node transitions for tool events - they're handled during deferred processing
            // if (this.graphState.iterations! < this.graphState.maxIterations!) {
            //   this.emitNodeTransition(observer, 'callModel');
            //   this.graphState.iterations = (this.graphState.iterations || 0) + 1;
            // } else {\n            //   this.emitNodeTransition(observer, 'generateResponse');
            // }
          } else if (event.type === EventType.TEXT_MESSAGE_START) {
            // Track message ID for dual text message approach
            this.currentMessageId = (event as TextMessageStartEvent).messageId;
            this.hasToolCallsOccurred = false;
            this.isSecondTextMessage = false;
          } else if (event.type === EventType.TEXT_MESSAGE_CONTENT) {
            // Check if this is text continuation after tool calls
            if (this.hasToolCallsOccurred && !this.isSecondTextMessage && !this.textMessageActive) {
              // Start second text message for continuation
              const newMessageId = uuidv4();
              observer.next({
                type: EventType.TEXT_MESSAGE_START,
                messageId: newMessageId,
                role: 'assistant',
                timestamp: Date.now()
              } as TextMessageStartEvent);

              this.currentMessageId = newMessageId;
              this.textMessageActive = true;
              this.isSecondTextMessage = true;
            }

            // Update the message ID for content events if we're in the second message
            if (this.isSecondTextMessage && this.currentMessageId) {
              // Forward the event with the correct message ID
              const updatedEvent = {
                ...event,
                messageId: this.currentMessageId
              };
              observer.next(updatedEvent);
              return; // Don't forward the original event
            }
          } else if (event.type === EventType.RUN_FINISHED) {
            // Before forwarding RUN_FINISHED, emit all our final events
            
            // Finish the final node step
            if (this.graphState.currentNode && this.graphState.currentNode !== 'END') {
              const finalStepName = `graph_node_${this.graphState.currentNode}`;
              if (this.activeSteps.has(finalStepName)) {
                observer.next({
                  type: EventType.STEP_FINISHED,
                  stepName: finalStepName,
                  timestamp: Date.now()
                } as StepFinishedEvent);
                this.activeSteps.delete(finalStepName);
              }
            }
            
            // Finish all remaining active steps
            for (const activeStepName of this.activeSteps) {
              observer.next({
                type: EventType.STEP_FINISHED,
                stepName: activeStepName,
                timestamp: Date.now()
              } as StepFinishedEvent);
            }
            this.activeSteps.clear();
            
            // Emit final graph traversal summary
            observer.next({
              type: EventType.STATE_SNAPSHOT,
              snapshot: {
                ...this.graphState,
                currentNode: 'END',
                graphPath: [...(this.graphState.graphPath || []), 'END'],
                completionStatus: 'success',
                totalIterations: this.graphState.iterations,
                totalToolCalls: this.graphState.toolCallsCompleted
              },
              timestamp: Date.now()
            } as StateSnapshotEvent);
          }
          
          // Filter events based on text message state - only defer state/step events
          if (this.textMessageActive &&
              (event.type === EventType.STEP_STARTED ||
               event.type === EventType.STEP_FINISHED ||
               event.type === EventType.STATE_SNAPSHOT ||
               event.type === EventType.STATE_DELTA)) {
            // Don't forward these state events during text message - defer them until after TEXT_MESSAGE_END
            this.pendingStepEvents.push(event);
            return;
          }
          
          // Forward all other events
          observer.next(event);
        },
        error: (err) => observer.error(err),
        complete: () => {
          // All cleanup is now handled in RUN_FINISHED event handler above
          observer.complete();
        }
      });

      // Return teardown logic
      return () => subscription.unsubscribe();
    });
  }

  /**
   * Emit node transition events for graph visualization
   */
  private emitNodeTransition(observer: any, nodeName: string): void {
    // If text message is active, defer the step events
    if (this.textMessageActive) {
      this.deferStepEventsForNode(nodeName);
      return;
    }
    
    this.emitStepEventsForNode(observer, nodeName);
  }
  
  /**
   * Store step events for later emission when text message ends
   */
  private deferStepEventsForNode(nodeName: string): void {
    // Update graph state
    const previousNode = this.graphState.currentNode;
    this.graphState.currentNode = nodeName;
    this.graphState.graphPath = [...(this.graphState.graphPath || []), nodeName];
    
    // Track node execution count
    if (!this.graphState.nodeExecutions) {
      this.graphState.nodeExecutions = {};
    }
    this.graphState.nodeExecutions[nodeName] = (this.graphState.nodeExecutions[nodeName] || 0) + 1;

    // Create step events but don't emit them yet
    const stepName = `graph_node_${nodeName}`;
    this.activeSteps.add(stepName);
    
    // Store STEP_STARTED event
    this.pendingStepEvents.push({
      type: EventType.STEP_STARTED,
      stepName,
      metadata: {
        previousNode,
        currentNode: nodeName,
        iteration: this.graphState.iterations,
        executionCount: this.graphState.nodeExecutions[nodeName]
      },
      timestamp: Date.now()
    } as StepStartedEvent);

    // Store STATE_SNAPSHOT event
    this.pendingStepEvents.push({
      type: EventType.STATE_SNAPSHOT,
      snapshot: {
        graphType: 'LangGraph',
        currentNode: nodeName,
        previousNode,
        graphPath: this.graphState.graphPath,
        nodeExecutions: this.graphState.nodeExecutions,
        iterations: this.graphState.iterations,
        toolCallsPending: this.graphState.toolCallsPending,
        toolCallsCompleted: this.graphState.toolCallsCompleted,
        nodes: ['processInput', 'callModel', 'executeTools', 'generateResponse']
      },
      timestamp: Date.now()
    } as StateSnapshotEvent);

    // Store STEP_FINISHED for previous node
    if (previousNode && previousNode !== 'START') {
      const previousStepName = `graph_node_${previousNode}`;
      this.activeSteps.delete(previousStepName);
      
      this.pendingStepEvents.push({
        type: EventType.STEP_FINISHED,
        stepName: previousStepName,
        timestamp: Date.now()
      } as StepFinishedEvent);
    }
  }
  
  /**
   * Emit step events immediately
   */
  private emitStepEventsForNode(observer: any, nodeName: string): void {
    // This is the original implementation moved here
    // Update graph state
    const previousNode = this.graphState.currentNode;
    this.graphState.currentNode = nodeName;
    this.graphState.graphPath = [...(this.graphState.graphPath || []), nodeName];
    
    // Track node execution count
    if (!this.graphState.nodeExecutions) {
      this.graphState.nodeExecutions = {};
    }
    this.graphState.nodeExecutions[nodeName] = (this.graphState.nodeExecutions[nodeName] || 0) + 1;

    // Emit step event for node transition
    const stepName = `graph_node_${nodeName}`;
    this.activeSteps.add(stepName);
    
    observer.next({
      type: EventType.STEP_STARTED,
      stepName,
      metadata: {
        previousNode,
        currentNode: nodeName,
        iteration: this.graphState.iterations,
        executionCount: this.graphState.nodeExecutions[nodeName]
      },
      timestamp: Date.now()
    } as StepStartedEvent);

    // Emit updated state snapshot with LangGraph-specific data
    observer.next({
      type: EventType.STATE_SNAPSHOT,
      snapshot: {
        graphType: 'LangGraph',
        currentNode: nodeName,
        previousNode,
        graphPath: this.graphState.graphPath,
        nodeExecutions: this.graphState.nodeExecutions,
        iterations: this.graphState.iterations,
        toolCallsPending: this.graphState.toolCallsPending,
        toolCallsCompleted: this.graphState.toolCallsCompleted,
        nodes: ['processInput', 'callModel', 'executeTools', 'generateResponse']
      },
      timestamp: Date.now()
    } as StateSnapshotEvent);

    // Emit step finished for previous node
    if (previousNode && previousNode !== 'START') {
      const previousStepName = `graph_node_${previousNode}`;
      this.activeSteps.delete(previousStepName);
      
      observer.next({
        type: EventType.STEP_FINISHED,
        stepName: previousStepName,
        timestamp: Date.now()
      } as StepFinishedEvent);
    }
  }

  /**
   * Get LangGraph-specific metrics
   */
  getGraphMetrics(): LangGraphState {
    return { ...this.graphState };
  }

  /**
   * Get graph execution path
   */
  getExecutionPath(): string[] {
    return [...(this.graphState.graphPath || [])];
  }

  /**
   * Get node execution statistics
   */
  getNodeStatistics(): Record<string, number> {
    return { ...(this.graphState.nodeExecutions || {}) };
  }
}