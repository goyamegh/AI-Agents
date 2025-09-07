import { join } from 'path';
import { BaseLogger } from './base-logger';

// Logger utility with hourly rotation
export class Logger extends BaseLogger {
  private currentLogFile: string = '';

  constructor(logDir: string = join(__dirname, '../../logs')) {
    super(logDir);
    this.updateLogFile();
  }

  private updateLogFile(): void {
    const timestamp = this.getTimestamp();
    const dateStr = this.getDateString(timestamp.unix);
    const hour = new Date(timestamp.unix).getHours().toString().padStart(2, '0');
    const newLogFile = join(this.logDir, `ai-agent-${dateStr}-${hour}.log`);
    
    if (this.currentLogFile !== newLogFile) {
      this.currentLogFile = newLogFile;
      // Log rotation message
      const rotationMsg = `[${timestamp.iso}] INFO: Log file rotated to ${this.currentLogFile}\n`;
      this.writeToFile(this.currentLogFile, rotationMsg);
    }
  }

  private formatMessage(level: string, message: string, data?: any): string {
    const timestamp = this.getTimestamp();
    const logEntry = `[${timestamp.iso}] ${level}: ${message}`;
    return data ? `${logEntry} ${JSON.stringify(data, null, 2)}` : logEntry;
  }

  info(message: string, data?: any): void {
    this.updateLogFile();
    const logEntry = this.formatMessage('INFO', message, data);
    console.log(logEntry);
    this.writeToFile(this.currentLogFile, logEntry + '\n');
  }

  warn(message: string, data?: any): void {
    this.updateLogFile();
    const logEntry = this.formatMessage('WARN', message, data);
    console.warn(logEntry);
    this.writeToFile(this.currentLogFile, logEntry + '\n');
  }

  error(message: string, data?: any): void {
    this.updateLogFile();
    const logEntry = this.formatMessage('ERROR', message, data);
    console.error(logEntry);
    this.writeToFile(this.currentLogFile, logEntry + '\n');
  }

  debug(message: string, data?: any): void {
    this.updateLogFile();
    const logEntry = this.formatMessage('DEBUG', message, data);
    if (process.env.DEBUG) {
      console.log(logEntry);
    }
    this.writeToFile(this.currentLogFile, logEntry + '\n');
  }

  // Specialized debugging for tool parameter streaming
  toolParameterDebug(phase: string, toolName: string, data: any): void {
    const debugData = {
      phase,
      toolName,
      timestamp: Date.now(),
      ...data
    };
    
    this.debug(`[TOOL_PARAM_STREAM] ${phase}`, debugData);
    
    // Always log tool parameter issues to console for immediate visibility
    if (phase === 'VALIDATION_FAILED' || phase === 'CIRCUIT_BREAKER_ACTIVATED') {
      console.log(`\n🚨 [TOOL DEBUG] ${phase} for ${toolName}:`, JSON.stringify(debugData, null, 2));
    } else if (process.env.DEBUG || process.env.TOOL_DEBUG) {
      console.log(`\n🔧 [TOOL DEBUG] ${phase} for ${toolName}:`, JSON.stringify(debugData, null, 2));
    }
  }

  // Log streaming agent output
  stream(message: string): void {
    this.updateLogFile();
    const timestamp = this.getTimestamp();
    this.writeToFile(this.currentLogFile, `[${timestamp.iso}] STREAM: ${message}\n`);
  }
}