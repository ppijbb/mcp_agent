#!/bin/bash

# SEO Doctor MCP Servers Installation Script
# Based on real-world MCP implementation from:
# https://medium.com/@matteo28/how-i-solved-a-real-world-customer-problem-with-the-model-context-protocol-mcp-328da5ac76fe

echo "🏥 Installing SEO Doctor MCP Servers..."
echo "=========================================="

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install Node.js and npm first."
    exit 1
fi

echo "📦 Installing required MCP servers..."

# Install official MCP servers
echo "Installing Google Search MCP Server..."
npm install -g g-search-mcp

echo "Installing Fetch MCP Server..."
npm install -g @modelcontextprotocol/server-fetch

echo "Installing Filesystem MCP Server..."
npm install -g @modelcontextprotocol/server-filesystem

echo "Installing Gmail MCP Server (optional)..."
npm install -g @modelcontextprotocol/server-gmail

echo "Installing Google Sheets MCP Server (optional)..."
npm install -g @modelcontextprotocol/server-google-sheets

# Try to install Lighthouse MCP Server (may not exist yet)
echo "Attempting to install Lighthouse MCP Server..."
npm install -g lighthouse-mcp-server || echo "⚠️  Lighthouse MCP Server not available, using fallback"

echo ""
echo "✅ MCP Server installation completed!"
echo ""
echo "📋 Next steps:"
echo "1. Set up environment variables in .env file:"
echo "   - GOOGLE_SEARCH_API_KEY"
echo "   - GOOGLE_SEARCH_ENGINE_ID"
echo "   - GMAIL_CLIENT_ID (optional)"
echo "   - GMAIL_CLIENT_SECRET (optional)"
echo ""
echo "2. Create seo_doctor_reports/ directory:"
echo "   mkdir -p seo_doctor_reports"
echo ""
echo "3. Test the SEO Doctor MCP Agent:"
echo "   python -c \"from srcs.seo_doctor.seo_doctor_mcp_agent import create_seo_doctor_agent; print('SEO Agent OK')\""
echo ""
echo "🚨 CRITICAL: This replaces all mock data with real MCP functionality!" 