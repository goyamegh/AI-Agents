import { appendFileSync, mkdirSync } from 'fs';
import { join } from 'path';

/**
 * Base logger with common file operations and directory management
 */
export abstract class BaseLogger {
  protected logDir: string;

  constructor(logDir: string) {
    this.logDir = logDir;
    // Ensure log directory exists
    mkdirSync(this.logDir, { recursive: true });
  }

  /**
   * Write entry to specified log file
   */
  protected writeToFile(logFile: string, content: string): void {
    try {
      appendFileSync(logFile, content);
    } catch (error) {
      // Fallback to console if file write fails
      console.error('Failed to write to log file:', error);
      console.log('Log content:', content);
    }
  }

  /**
   * Ensure directory exists for a file path
   */
  protected ensureDirectoryExists(filePath: string): void {
    const dir = join(filePath, '..');
    mkdirSync(dir, { recursive: true });
  }

  /**
   * Generate standardized timestamp
   */
  protected getTimestamp(): { unix: number; iso: string } {
    const now = Date.now();
    return {
      unix: now,
      iso: new Date(now).toISOString()
    };
  }

  /**
   * Generate date string for file naming (YYYY-MM-DD)
   */
  protected getDateString(timestamp?: number): string {
    const date = timestamp ? new Date(timestamp) : new Date();
    return date.toISOString().split('T')[0];
  }
}