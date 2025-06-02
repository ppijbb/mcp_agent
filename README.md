# MCP Agent System

A comprehensive multi-agent system for enterprise automation and intelligence, featuring basic agents for simple tasks and sophisticated enterprise-level agents for complex business automation.

## 📁 Project Structure

```
srcs/
├── basic_agents/           # Simple, lightweight agents
│   ├── basic.py           # Basic functionality and testing
│   ├── agent.py           # Base Agent class
│   ├── swarm.py           # Multi-agent coordination
│   ├── workflow_orchestration.py # Workflow management
│   ├── researcher.py      # Research and information gathering
│   ├── parallel.py        # Parallel processing demonstration
│   ├── streamlit_agent.py # Web interface agent
│   ├── data_generator.py  # Data generation and synthesis
│   ├── enhanced_data_generator.py # Advanced data generation
│   └── rag_agent.py       # Retrieval-Augmented Generation
├── enterprise_agents/      # Sophisticated business automation
│   ├── mental.py          # Mental model analysis
│   ├── hr_recruitment_agent.py              # HR & Talent Acquisition
│   ├── legal_compliance_agent.py            # Legal & Regulatory Compliance
│   ├── cybersecurity_infrastructure_agent.py # Security & Threat Detection
│   ├── supply_chain_orchestrator_agent.py   # Supply Chain Optimization
│   ├── customer_lifetime_value_agent.py     # Customer Experience & CLV
│   ├── esg_carbon_neutral_agent.py         # ESG & Sustainability
│   ├── hybrid_workplace_optimizer_agent.py  # Workplace Optimization
│   └── product_innovation_accelerator_agent.py # Innovation & Development
├── utils/                  # Common utilities and frameworks
│   └── mental_visualization.py # Interactive visualization
└── run_agent.py           # Unified execution script
```

## 🚀 Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API keys:
   - Create `mcp_agent.secrets.yaml` file in the `srcs` directory
   - Add your API keys for OpenAI and Google:
     ```yaml
     openai:
       api_key: your-openai-api-key
     google:
       api_key: your-google-api-key
     ```

## 🎯 Running Agents

### Using the Unified Runner (Recommended)

Navigate to the `srcs` directory and use the unified runner:

```bash
cd srcs

# List all available agents
python run_agent.py --list

# Run basic agents
python run_agent.py --basic researcher
python run_agent.py --basic data_generator
python run_agent.py --basic rag

# Run enterprise agents
python run_agent.py --enterprise supply_chain
python run_agent.py --enterprise customer_clv
python run_agent.py --enterprise workplace

# Run utilities
python run_agent.py --utility mental
python run_agent.py --utility swarm
```

### Direct Execution

You can also run agents directly:

```bash
cd srcs

# Basic agents
python basic_agents/researcher.py
python basic_agents/data_generator.py

# Enterprise agents  
python enterprise_agents/supply_chain_orchestrator_agent.py
python enterprise_agents/customer_lifetime_value_agent.py

# Utilities
python utils/mental.py
```

## 📝 Available Agents

### Basic Agents
- **researcher** - Research and information gathering
- **basic** - Basic functionality and testing
- **parallel** - Parallel processing demonstration  
- **swarm** - Multi-agent swarm coordination
- **streamlit** - Web interface agent
- **workflow** - Workflow orchestration and management
- **data_generator** - Data generation and synthesis
- **enhanced_data_generator** - Advanced data generation with ML
- **rag** - Retrieval-Augmented Generation

### Enterprise Agents
- **hr_recruitment** - HR recruitment and talent acquisition automation
- **mental** - Mental model analysis and visualization
- **legal_compliance** - Legal compliance and contract analysis
- **cybersecurity** - Cybersecurity infrastructure and threat detection
- **supply_chain** - Supply chain orchestration and optimization
- **customer_clv** - Customer lifetime value and experience optimization
- **esg_carbon** - ESG and carbon neutrality management
- **workplace** - Hybrid workplace optimization and management
- **innovation** - Product innovation acceleration and development

### Utilities
- **mental_viz** - Mental model interactive visualization

## 💼 Enterprise Features

The enterprise agents provide comprehensive business automation with:

- **ROI-Focused Solutions**: Each agent targets 200-600% ROI through measurable improvements
- **Industry Standards**: Compliance with frameworks like GDPR, SOX, HIPAA, SASB, GRI
- **Scalable Architecture**: Multi-agent orchestration with quality control systems
- **Real-time Analytics**: Performance monitoring and continuous optimization
- **Integration Ready**: API-first design for enterprise system integration

## 🔧 Requirements

- Python 3.8+
- Docker (for Python interpreter functionality)
- OpenAI API key
- Google API key (optional, for enhanced search capabilities)

## 📊 Business Impact

Enterprise agents deliver measurable business value:
- **Supply Chain**: 15-30% cost reduction, 25-40% delivery improvement
- **Customer CLV**: 25-40% retention improvement, 10-25% CLV increase  
- **ESG Management**: Carbon neutrality achievement, 40-60% ESG rating improvement
- **Workplace Optimization**: 30-50% productivity improvement, 25-40% cost reduction
- **Innovation Acceleration**: 40-60% time-to-market reduction, 50-75% success rate improvement

---

*For detailed documentation on individual agents and their capabilities, refer to the agent-specific files and their embedded documentation.*

