# Local Researcher Project

A fully autonomous multi-agent research system powered by AI with advanced web interface and real-time monitoring.

## 🚀 Features

### Core Research Capabilities
- **Autonomous Research**: Self-analyzing and self-executing research workflows
- **Multi-Agent Architecture**: Specialized agents for different research tasks
- **LangGraph Integration**: Advanced workflow management with LangGraph
- **LLM Integration**: Advanced language model integration for intelligent decision making
- **MCP Support**: Model Context Protocol integration for enhanced capabilities

### Web Interface & Visualization
- **Streamlit Web App**: Interactive web interface for research management
- **Real-time Dashboard**: Live monitoring of research progress and system status
- **Data Visualization**: Interactive charts and analytics using Plotly
- **Report Generation**: Automated report generation in multiple formats (PDF, HTML, DOCX, Markdown)

### Advanced Features
- **Browser Automation**: Automated web browsing and data collection
- **Real-time Monitoring**: System health monitoring and performance tracking
- **Multi-format Reports**: Executive summaries, detailed analysis, academic papers
- **Collaborative Research**: Multi-user support and shared workspaces

## 🛠️ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd local_researcher_project

# Install dependencies
pip install -r requirements.txt

# Install additional system dependencies (for PDF generation)
# Ubuntu/Debian:
sudo apt-get install wkhtmltopdf

# macOS:
brew install wkhtmltopdf

# Windows:
# Download from https://wkhtmltopdf.org/downloads.html
```

### 2. Environment Setup

```bash
# Set up environment variables
export GEMINI_API_KEY="your_api_key_here"
export OPENAI_API_KEY="your_openai_key_here"  # Optional
export ANTHROPIC_API_KEY="your_anthropic_key_here"  # Optional
```

### 3. Run the System

#### Command Line Interface
```bash
python main.py
```

#### Web Interface
```bash
python run_web_app.py
```
Then open http://localhost:8501 in your browser.

#### Streamlit Integration (from pages/)
```bash
# From the parent directory
streamlit run pages/research.py
```

## 🏗️ Architecture

### Core Components

- **LangGraphOrchestrator**: Main orchestrator using LangGraph for workflow management
- **Task Analyzer**: Analyzes research objectives and requirements
- **Task Decomposer**: Breaks down complex research into manageable tasks
- **Research Agent**: Conducts actual research and data collection with browser automation
- **Evaluation Agent**: Evaluates research quality and completeness
- **Validation Agent**: Validates results against original objectives
- **Synthesis Agent**: Synthesizes final deliverables

### Web Interface Components

- **Streamlit App**: Main web interface (`src/web/streamlit_app.py`)
- **Data Visualizer**: Advanced visualization capabilities (`src/visualization/data_visualizer.py`)
- **Report Generator**: Multi-format report generation (`src/generation/report_generator.py`)
- **System Monitor**: Real-time system monitoring (`src/monitoring/system_monitor.py`)

### Integration Points

- **Pages Integration**: Seamless integration with existing Streamlit pages
- **MCP Integration**: Enhanced tool capabilities through Model Context Protocol
- **Browser Automation**: Automated web research using browser-use
- **Real-time Monitoring**: System health and performance tracking

## 📊 Usage Examples

### Basic Research
```python
from src.core.autonomous_orchestrator import LangGraphOrchestrator
from src.agents.research_agent import ResearchAgent

# Initialize orchestrator
orchestrator = LangGraphOrchestrator(config_path, agents, mcp_manager)

# Start research
objective_id = await orchestrator.start_autonomous_research(
    "AI trends in 2024",
    context={"depth": "comprehensive", "domain": "technology"}
)
```

### Web Interface
1. Start the web app: `python run_web_app.py`
2. Navigate to Research Dashboard
3. Enter your research query
4. Configure research options
5. Click "Start Research"
6. Monitor progress in real-time
7. Download generated reports

### Data Visualization
```python
from src.visualization.data_visualizer import DataVisualizer

visualizer = DataVisualizer()
fig = visualizer.create_research_timeline(research_data)
fig.show()
```

## ⚙️ Configuration

The system can be configured through YAML configuration files in the `configs/` directory:

- **General Settings**: Basic system configuration
- **Research Settings**: Research-specific parameters
- **Display Settings**: UI and visualization preferences
- **Advanced Settings**: Advanced features and integrations

## 🔧 Advanced Features

### Browser Automation
- Automated web browsing and data collection
- JavaScript execution and interaction
- Screenshot capture and analysis
- Form filling and navigation

### Real-time Monitoring
- System performance metrics
- Research progress tracking
- Error monitoring and alerting
- Resource utilization analysis

### Report Generation
- Multiple output formats (PDF, HTML, DOCX, Markdown)
- Customizable templates
- Automated formatting and styling
- Citation management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License

## 🆘 Support

For support and questions:
- Check the documentation in the `docs/` directory
- Open an issue on GitHub
- Contact the development team

## 🔄 Updates

### Version 2.0.0
- Added LangGraph integration
- Implemented web interface
- Added data visualization capabilities
- Enhanced report generation
- Integrated browser automation
- Added real-time monitoring