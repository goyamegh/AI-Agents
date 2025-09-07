#!/usr/bin/env ts-node

/**
 * Universal Agent CLI with Bedrock ConverseStream API and MCP Server Integration
 * 
 * This script creates any agent that:
 * 1. Uses AWS Bedrock ConverseStream API for real-time responses
 * 2. Connects to local MCP servers via stdio
 * 3. Uses agent-specific system prompts
 * 4. Handles tool calls through MCP protocol
 * 
 * Supports multiple agent types: jarvis, langgraph, strands (future)
 */

import * as dotenv from 'dotenv';
import { AgentFactory } from './agents/agent-factory';
import { Logger } from './utils/logger';
import { ConfigLoader } from './config/config-loader';

// Load environment variables
dotenv.config();

// Main execution
async function main() {
  const logger = new Logger();
  
  // Parse CLI arguments for agent selection
  const args = process.argv.slice(2);
  const agentTypeIndex = args.findIndex(arg => arg === '--agent' || arg === '-a');
  
  // Get agent type from multiple sources (CLI args, env var, or default)
  const agentType = agentTypeIndex !== -1 && args[agentTypeIndex + 1] 
    ? args[agentTypeIndex + 1]
    : process.env.AGENT_TYPE || 
      args[0] || // Legacy support for direct agent type as first arg
      AgentFactory.getDefaultAgentType();
  
  logger.info(`🚀 Starting ${agentType} Agent with MCP Integration`);
  console.log(`🚀 Starting ${agentType} Agent with MCP Integration...\n`);

  // Load MCP configuration using shared config loader
  const mcpConfigs = await ConfigLoader.loadMCPConfig();

  try {
    const agent = AgentFactory.createAgent(agentType);
    
    await agent.initialize(mcpConfigs, undefined);
    await agent.startInteractiveMode();
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    logger.error('Failed to start agent', { error: errorMessage, agentType });
    console.error('Failed to start agent:', errorMessage);
    
    if (errorMessage.includes('Unknown agent type')) {
      const availableAgents = AgentFactory.getAvailableAgents();
      console.log('\nAvailable agent types:');
      availableAgents.forEach((type, index) => {
        const isDefault = type === AgentFactory.getDefaultAgentType();
        console.log(`  - ${type}${isDefault ? ' (default)' : ''}`);
      });
      console.log('\nUsage:');
      console.log('  npm start [agent-type]              # Legacy format');
      console.log('  npm start -- --agent langgraph      # Preferred format');
      console.log('  AGENT_TYPE=langgraph npm start      # Environment variable');
    }
  } finally {
    logger.info('Agent shutdown complete');
    process.exit(0);
  }
}

if (require.main === module) {
  main();
}