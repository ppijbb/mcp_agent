# Autonomous Multi-Agent Research System

A fully autonomous multi-agent research system powered by LLM-based decision making that self-analyzes user requests, dynamically decomposes tasks, orchestrates specialized agents, and generates comprehensive research deliverables.

## 🚀 Key Features

- **LLM-Powered Autonomy**: Uses advanced language models for intelligent decision making
- **Dynamic Task Analysis**: AI-driven analysis of user requests and objective extraction
- **Intelligent Task Decomposition**: LLM-based task breakdown and agent assignment
- **Real Web Research**: Actual web scraping and data collection capabilities
- **Multi-Agent Orchestration**: Coordinates specialized agents with LLM guidance
- **Critical Evaluation**: AI-powered result evaluation and quality assessment
- **Result Validation**: LLM-based validation against original objectives
- **Final Synthesis**: Intelligent generation of comprehensive deliverables

## 🧠 LLM Integration

This system is now fully agentic with:

- **Gemini 2.0 Flash**: Primary LLM for decision making and analysis
- **Real-time Web Search**: Actual web scraping and data collection
- **Intelligent Planning**: LLM-based research strategy development
- **Adaptive Learning**: System learns from previous executions
- **Dynamic Decision Making**: No hardcoded rules or templates

## 🏗️ Architecture

The system consists of LLM-powered specialized agents:

- **AutonomousOrchestrator**: LLM-based orchestration and decision making
- **TaskAnalyzerAgent**: AI-driven request analysis and objective extraction
- **TaskDecomposerAgent**: LLM-based task decomposition and agent assignment
- **ResearchAgent**: Real web research with LLM coordination
- **LLMMethods**: Core LLM integration and decision making

## 📦 Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (see Configuration section)
4. Run the system: `python main.py`

## ⚙️ Configuration

Set the following environment variables:

```bash
# Required
export GEMINI_API_KEY="your_gemini_api_key_here"

# Optional
export OPENAI_API_KEY="your_openai_api_key_here"
export MCP_SERVER_URL="your_mcp_server_url"
```

## 🚀 Usage

### Command Line Interface

```bash
# Start autonomous LLM-based research
python main.py research "Analyze AI trends in healthcare"

# Check research status
python main.py status <objective_id>

# List all research objectives
python main.py list

# Run in interactive mode
python main.py interactive
```

### Interactive Mode

The system runs in interactive mode for continuous research:

```bash
python main.py interactive
```

## 🔍 How It Works

1. **LLM Analysis**: The system uses Gemini to analyze your request and extract research objectives
2. **Intelligent Decomposition**: AI breaks down objectives into specific, executable tasks
3. **Real Research**: Agents perform actual web searches and data collection
4. **LLM Coordination**: AI coordinates multi-agent execution and decision making
5. **Quality Evaluation**: LLM evaluates results and suggests improvements
6. **Final Synthesis**: AI generates comprehensive research reports

## 📊 Example Output

The system generates:
- **Research Reports**: Comprehensive markdown reports
- **Data Analysis**: Structured analysis of collected information
- **Source Citations**: Proper attribution of all sources
- **Quality Metrics**: AI-assessed quality scores
- **Recommendations**: AI-generated insights and suggestions

## 🎯 Examples

### Basic Research

```bash
python main.py research "Compare renewable energy technologies"
```

### Complex Analysis

```bash
python main.py research "Analyze the impact of AI on healthcare delivery systems"
```

### Market Research

```bash
python main.py research "Study emerging trends in fintech startups"
```

## 🔧 Development

This system is production-ready with:
- **No Fallback Code**: All functionality is real and autonomous
- **LLM Integration**: Full AI-powered decision making
- **Real Web Research**: Actual data collection capabilities
- **Error Handling**: Robust error management and recovery
- **Logging**: Comprehensive logging for monitoring

## 📈 Performance

- **Autonomous Operation**: Requires minimal human intervention
- **Intelligent Planning**: AI optimizes research strategies
- **Quality Assurance**: Built-in quality evaluation and improvement
- **Scalable**: Handles complex research requests efficiently

## 🛡️ Security

- **API Key Management**: Secure handling of API credentials
- **Data Privacy**: No data is stored permanently
- **Error Handling**: Graceful handling of API failures

## 📝 License

MIT License

## 🤝 Contributing

This system is designed for autonomous operation. Contributions should focus on:
- Enhanced LLM integration
- Improved research capabilities
- Better error handling
- Performance optimization

## ⚠️ Requirements

- Python 3.8+
- Gemini API key
- Internet connection for web research
- Sufficient API quota for LLM calls