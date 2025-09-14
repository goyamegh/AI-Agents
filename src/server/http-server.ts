import express from 'express';
import cors from 'cors';
import http from 'http';
import { Logger } from '../utils/logger';
import { AGUIAuditLogger } from '../utils/ag-ui-audit-logger';
import { BaseAGUIAdapter, BaseAGUIConfig } from '../ag-ui/base-ag-ui-adapter';
import { RunAgentInput, BaseEvent, EventType, RunErrorEvent, RunFinishedEvent } from '@ag-ui/core';

export class HTTPServer {
  private app: express.Application;
  private server: http.Server;
  private logger: Logger;
  private auditLogger?: AGUIAuditLogger;
  private config: BaseAGUIConfig;
  private adapter: BaseAGUIAdapter;
  private isShuttingDown: boolean = false;

  constructor(config: BaseAGUIConfig, adapter: BaseAGUIAdapter, logger: Logger, auditLogger?: AGUIAuditLogger) {
    this.config = config;
    this.adapter = adapter;
    this.logger = logger;
    this.auditLogger = auditLogger;
    this.app = express();
    this.server = http.createServer(this.app);
  }

  setupMiddleware(): void {
    // CORS configuration
    if (this.config.cors) {
      this.app.use(cors({
        origin: this.config.cors.origins,
        credentials: this.config.cors.credentials
      }));
    } else {
      this.app.use(cors());
    }

    // JSON parsing
    this.app.use(express.json({ limit: '10mb' }));
    
    // Request logging
    this.app.use((req, res, next) => {
      this.logger.info(`${req.method} ${req.path}`, {
        method: req.method,
        path: req.path,
        userAgent: req.get('User-Agent')
      });
      next();
    });
  }

  setupRoutes(): void {
    // Health check endpoint
    this.app.get('/health', (req, res) => {
      res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0'
      });
    });

    // AG UI protocol info endpoint
    this.app.get('/api/info', (req, res) => {
      res.json({
        name: 'AI Agent AG UI Server',
        version: '1.0.0',
        protocol: 'ag-ui',
        capabilities: {
          streaming: false, // HTTP-only, no WebSocket streaming
          tools: true,
          conversations: true,
          contextWindow: 200000,
          maxTokens: 4000,
          supportedModels: ['claude-sonnet-4']
        }
      });
    });

    // Get available tools
    this.app.get('/api/tools', async (req, res) => {
      try {
        const tools = await this.adapter.getTools();
        res.json({ tools });
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        this.logger.error('Error getting tools', { error: errorMessage });
        res.status(500).json({ error: errorMessage });
      }
    });

    // Run agent endpoint (SSE streaming)
    this.app.post('/', async (req, res) => {
      // Check if server is shutting down
      if (this.isShuttingDown) {
        res.status(503).json({ error: 'Server is shutting down' });
        return;
      }

      const input: RunAgentInput = req.body;
      
      try {
        // Validate input
        const validationResult = this.validateRunAgentInput(input);
        if (!validationResult.isValid) {
          this.handleValidationError(res, validationResult.errors, input.threadId, input.runId);
          return;
        }

        this.logger.info('Running agent via SSE streaming', {
          threadId: input.threadId,
          runId: input.runId,
          messageCount: input.messages.length
        });

        // Log HTTP request details for audit
        this.auditLogger?.logHttpRequest(input.threadId, input.runId, {
          method: req.method,
          path: req.path,
          userAgent: req.get('User-Agent'),
          contentLength: req.get('Content-Length') ? parseInt(req.get('Content-Length')!) : undefined,
          messageCount: input.messages.length,
          toolCount: input.tools?.length || 0
        });

        // Set SSE headers
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Cache-Control'
        });

        // Stream events in real-time
        const eventStream = await this.adapter.runAgent(input);
        
        // Declare subscription variable to track the observable subscription
        let subscription: any = null;
        
        subscription = eventStream.subscribe({
          next: (event) => {
            // Send event as SSE format
            res.write(`data: ${JSON.stringify(event)}\n\n`);
          },
          error: (error) => {
            const errorMessage = error instanceof Error ? error.message : String(error);
            const errorStack = error instanceof Error ? error.stack : undefined;
            this.logger.error('Error in event stream', { 
              error: errorMessage,
              stack: errorStack,
              threadId: input.threadId,
              runId: input.runId
            });
            // Send error event and close connection
            const errorEvent: RunErrorEvent = {
              type: EventType.RUN_ERROR,
              message: errorMessage,
              code: 'STREAM_ERROR',
              timestamp: Date.now()
            };
            res.write(`data: ${JSON.stringify(errorEvent)}\n\n`);
            res.end();
          },
          complete: () => {
            // Stream already sends RUN_FINISHED event, just close connection
            res.end();
          }
        });

        // Handle client disconnect - use once to avoid multiple listeners
        const handleClientDisconnect = () => {
          this.logger.info('Client disconnected from SSE stream', {
            threadId: input.threadId,
            runId: input.runId
          });
          // Clean up the event stream subscription
          if (subscription && typeof subscription.unsubscribe === 'function') {
            subscription.unsubscribe();
          }
        };
        
        req.once('close', handleClientDisconnect);
        req.once('error', handleClientDisconnect);

      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        const errorStack = error instanceof Error ? error.stack : undefined;
        this.logger.error('Error running agent', { 
          error: errorMessage,
          stack: errorStack,
          threadId: input.threadId,
          runId: input.runId
        });
        
        if (!res.headersSent) {
          res.status(500).json({ error: errorMessage });
        } else {
          const errorEvent: RunErrorEvent = {
            type: EventType.RUN_ERROR,
            message: errorMessage,
            code: 'AGENT_ERROR',
            timestamp: Date.now()
          };
          res.write(`data: ${JSON.stringify(errorEvent)}\n\n`);
          res.end();
        }
      }
    });
  }

  async start(): Promise<void> {
    return new Promise((resolve) => {
      this.server.listen(this.config.port, this.config.host, () => {
        this.logger.info('AI Agent AG UI HTTP Server started', {
          host: this.config.host,
          port: this.config.port,
          httpEndpoint: `http://${this.config.host}:${this.config.port}`
        });
        console.log(`🚀 AI Agent AG UI Server running at http://${this.config.host}:${this.config.port}`);
        resolve();
      });
    });
  }

  async stop(): Promise<void> {
    this.isShuttingDown = true;
    
    return new Promise((resolve, reject) => {
      // Close all active connections first
      this.server.closeAllConnections?.();
      
      // Remove all listeners to prevent memory leaks
      this.server.removeAllListeners();
      
      this.server.close((error) => {
        if (error) {
          this.logger.error('Error stopping HTTP server', { error: error.message });
          reject(error);
        } else {
          this.logger.info('AI Agent AG UI HTTP Server stopped');
          resolve();
        }
      });
    });
  }

  /**
   * Validate RunAgentInput structure and required fields
   */
  private validateRunAgentInput(input: RunAgentInput): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];
    
    // Validate required string fields
    if (!input.threadId || typeof input.threadId !== 'string') {
      errors.push('threadId must be a non-empty string');
    }
    
    if (!input.runId || typeof input.runId !== 'string') {
      errors.push('runId must be a non-empty string');
    }
    
    // Validate messages array
    if (!input.messages || !Array.isArray(input.messages) || input.messages.length === 0) {
      errors.push('messages must be a non-empty array');
    } else {
      // Validate individual message structure
      input.messages.forEach((msg, index) => {
        if (!msg.role || typeof msg.role !== 'string') {
          errors.push(`messages[${index}].role must be a string`);
        }
        
        // Assistant messages with toolCalls may not have content
        const isAssistantWithTools = msg.role === 'assistant' && msg.toolCalls;
        
        // Content is required unless it's an assistant message with tool calls
        if (!isAssistantWithTools) {
          if (!msg.content || (typeof msg.content !== 'string' && !Array.isArray(msg.content))) {
            errors.push(`messages[${index}].content must be a string or array`);
            // Log the problematic message for debugging
            this.logger.warn(`Invalid message content at index ${index}`, {
              index,
              role: msg.role,
              contentType: typeof msg.content,
              contentValue: msg.content,
              isNull: msg.content === null,
              isUndefined: msg.content === undefined,
              messageKeys: Object.keys(msg)
            });
          }
        } else if (msg.content && typeof msg.content !== 'string' && !Array.isArray(msg.content)) {
          // If content is provided for assistant with tools, it must be valid
          errors.push(`messages[${index}].content must be a string or array when provided`);
        }
      });
    }
    
    // Validate optional fields
    if (input.tools && !Array.isArray(input.tools)) {
      errors.push('tools must be an array if provided');
    }
    
    if (input.context && !Array.isArray(input.context)) {
      errors.push('context must be an array if provided');
    }
    
    return {
      isValid: errors.length === 0,
      errors
    };
  }

  /**
   * Handle validation errors with appropriate response format
   */
  private handleValidationError(res: express.Response, errors: string[], threadId?: string, runId?: string): void {
    this.logger.warn('Input validation failed', { 
      errors,
      threadId,
      runId
    });

    // Log validation error for audit
    if (threadId && runId) {
      this.auditLogger?.logValidationError(threadId, runId, errors);
    }
    
    const errorResponse: RunErrorEvent = {
      type: EventType.RUN_ERROR,
      message: `Input validation failed: ${errors.join(', ')}`,
      code: 'VALIDATION_ERROR',
      timestamp: Date.now()
    };
    
    if (!res.headersSent) {
      res.status(400).json(errorResponse);
    } else {
      res.write(`data: ${JSON.stringify(errorResponse)}\n\n`);
      res.end();
    }
  }
}