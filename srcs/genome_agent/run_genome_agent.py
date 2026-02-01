#!/usr/bin/env python3
"""
Run script for the Genome Agent

This script provides a command-line interface for running genome analysis workflows.
"""

import asyncio
import argparse
import json
import sys
from typing import Dict, Any

from genome_agent import run_genome_analysis, create_genome_agent


class GenomeAgentRunner:
    """Runner class for the Genome Agent"""

    def __init__(self):
        self.agent = None

    async def initialize_agent(self, output_dir: str = "genome_analysis_reports"):
        """Initialize the genome agent"""
        try:
            self.agent = await create_genome_agent(output_dir)
            print(f"✅ Genome agent initialized successfully")
            print(f"📁 Output directory: {output_dir}")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize genome agent: {e}")
            return False

    def display_menu(self):
        """Display the main menu"""
        print("\n" + "="*60)
        print("🧬 GENOME AGENT - MAIN MENU")
        print("="*60)
        print("1. 🔍 Run Genome Analysis")
        print("2. 📊 View Analysis Results")
        print("3. 💾 Manage Genome Data")
        print("4. 🔧 Agent Configuration")
        print("5. 📈 Performance Metrics")
        print("6. ❓ Help & Documentation")
        print("0. 🚪 Exit")
        print("="*60)

    async def run_genome_analysis_menu(self):
        """Run genome analysis workflow"""
        print("\n🔍 GENOME ANALYSIS WORKFLOW")
        print("-" * 40)

        # Get analysis request
        analysis_request = input("Enter analysis request (or press Enter for default): ").strip()
        if not analysis_request:
            analysis_request = "Analyze genetic variants in human genome for disease risk assessment"

        # Get data IDs
        data_ids_input = input("Enter data IDs (comma-separated, or press Enter for none): ").strip()
        data_ids = [id.strip() for id in data_ids_input.split(",")] if data_ids_input else []

        # Enable research
        enable_research_input = input("Enable research context? (y/n, default: y): ").strip().lower()
        enable_research = enable_research_input != 'n'

        # Execute plan
        execute_plan_input = input("Execute analysis plan? (y/n, default: n): ").strip().lower()
        execute_plan = execute_plan_input == 'y'

        print(f"\n🚀 Starting genome analysis...")
        print(f"📋 Request: {analysis_request}")
        print(f"📊 Data IDs: {data_ids if data_ids else 'None'}")
        print(f"🔬 Research: {'Enabled' if enable_research else 'Disabled'}")
        print(f"⚡ Execution: {'Enabled' if execute_plan else 'Disabled'}")

        try:
            result = await run_genome_analysis(
                analysis_request=analysis_request,
                data_ids=data_ids,
                enable_research=enable_research,
                execute_plan=execute_plan
            )

            if "error" in result:
                print(f"❌ Analysis failed: {result['error']}")
            else:
                print(f"✅ Analysis completed successfully!")
                print(f"📋 Plan ID: {result.get('plan', {}).get('plan_id', 'N/A')}")

                if execute_plan and "execution_result" in result:
                    exec_result = result["execution_result"]
                    print(f"⏱️  Execution time: {exec_result.get('execution_time', 0):.2f} seconds")
                    print(f"✅ Steps completed: {len(exec_result.get('steps_completed', []))}")
                    print(f"❌ Errors: {len(exec_result.get('errors', []))}")

                # Save result
                self._save_analysis_result(result)

        except Exception as e:
            print(f"❌ Error during analysis: {e}")

    def _save_analysis_result(self, result: Dict[str, Any]):
        """Save analysis result to file"""
        try:
            timestamp = result.get('plan', {}).get('plan_id', 'unknown')
            filename = f"analysis_result_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            print(f"💾 Analysis result saved to: {filename}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save result: {e}")

    async def view_analysis_results(self):
        """View analysis results"""
        print("\n📊 VIEWING ANALYSIS RESULTS")
        print("-" * 40)

        # This would typically list saved results
        print("📁 Available analysis results:")
        print("   (This feature would list saved analysis results)")
        print("   (In a full implementation, it would scan the output directory)")

    async def manage_genome_data(self):
        """Manage genome data"""
        print("\n💾 GENOME DATA MANAGEMENT")
        print("-" * 40)

        print("📁 Available genome data:")
        print("   (This feature would list saved genome data files)")
        print("   (In a full implementation, it would scan the data directory)")

    async def agent_configuration(self):
        """Configure agent settings"""
        print("\n🔧 AGENT CONFIGURATION")
        print("-" * 40)

        print("⚙️  Current configuration:")
        print(f"   • Output directory: {self.agent.output_dir if self.agent else 'Not initialized'}")
        print(f"   • MCP enabled: {self.agent.enable_mcp if self.agent else 'N/A'}")
        print(f"   • Supported databases: {len(self.agent.supported_databases) if self.agent else 0}")
        print(f"   • Supported tools: {len(self.agent.supported_tools) if self.agent else 0}")

    async def performance_metrics(self):
        """Show performance metrics"""
        print("\n📈 PERFORMANCE METRICS")
        print("-" * 40)

        print("📊 Agent performance statistics:")
        print("   (This feature would show execution statistics)")
        print("   (In a full implementation, it would track various metrics)")

    def show_help(self):
        """Show help and documentation"""
        print("\n❓ HELP & DOCUMENTATION")
        print("-" * 40)
        print("🧬 Genome Agent Help:")
        print("   This agent provides comprehensive genome analysis capabilities")
        print("   including sequence analysis, variant calling, and more.")
        print("\n📚 Key Features:")
        print("   • DNA sequence analysis")
        print("   • Gene expression analysis")
        print("   • Variant calling and interpretation")
        print("   • Phylogenetic analysis")
        print("   • Integration with genomic databases")
        print("\n🔗 Supported Databases:")
        print("   • NCBI, Ensembl, UCSC, UniProt")
        print("   • KEGG, Reactome, Gene Ontology")
        print("   • STRING, BioGRID, ChEMBL")
        print("\n🛠️  Supported Tools:")
        print("   • BLAST, BWA, GATK, Samtools")
        print("   • IGV, R, Python")

    async def run(self):
        """Main run loop"""
        print("🧬 Welcome to the Genome Agent!")
        print("Initializing...")

        # Initialize agent
        if not await self.initialize_agent():
            print("❌ Failed to initialize agent. Exiting.")
            return

        while True:
            try:
                self.display_menu()
                choice = input("\nSelect an option (0-6): ").strip()

                if choice == "0":
                    print("👋 Goodbye!")
                    break
                elif choice == "1":
                    await self.run_genome_analysis_menu()
                elif choice == "2":
                    await self.view_analysis_results()
                elif choice == "3":
                    await self.manage_genome_data()
                elif choice == "4":
                    await self.agent_configuration()
                elif choice == "5":
                    await self.performance_metrics()
                elif choice == "6":
                    self.show_help()
                else:
                    print("❌ Invalid choice. Please select 0-6.")

                input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                input("Press Enter to continue...")


async def run_quick_demo():
    """Run a quick demo of the genome agent"""
    print("🧬 GENOME AGENT - QUICK DEMO")
    print("=" * 50)

    # Example analysis request
    analysis_request = "Analyze genetic variants in BRCA1 and BRCA2 genes for breast cancer risk assessment"

    print(f"📋 Analysis Request: {analysis_request}")
    print("🚀 Starting analysis...")

    try:
        result = await run_genome_analysis(
            analysis_request=analysis_request,
            enable_research=True,
            execute_plan=False
        )

        if "error" in result:
            print(f"❌ Demo failed: {result['error']}")
        else:
            print("✅ Demo completed successfully!")
            print(f"📋 Generated plan ID: {result.get('plan', {}).get('plan_id', 'N/A')}")

            # Save demo result
            with open("demo_result.json", 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print("💾 Demo result saved to: demo_result.json")

    except Exception as e:
        print(f"❌ Demo error: {e}")


def parse_args() -> Dict[str, Any]:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Genome Agent Runner")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a quick demo instead of interactive mode"
    )
    parser.add_argument(
        "--output-dir",
        default="genome_analysis_reports",
        help="Output directory for analysis reports"
    )
    parser.add_argument(
        "--analysis-request",
        help="Analysis request to process"
    )
    parser.add_argument(
        "--data-ids",
        nargs="+",
        help="Data IDs to analyze"
    )
    parser.add_argument(
        "--enable-research",
        action="store_true",
        default=True,
        help="Enable research context"
    )
    parser.add_argument(
        "--execute-plan",
        action="store_true",
        help="Execute the analysis plan"
    )

    return vars(parser.parse_args())


async def main():
    """Main function"""
    args = parse_args()

    if args["demo"]:
        await run_quick_demo()
        return

    if args["analysis_request"]:
        # Run specific analysis
        result = await run_genome_analysis(
            analysis_request=args["analysis_request"],
            data_ids=args["data_ids"],
            enable_research=args["enable_research"],
            execute_plan=args["execute_plan"],
            output_dir=args["output_dir"]
        )

        print("Analysis Result:")
        print(json.dumps(result, indent=2, default=str))
        return

    # Run interactive mode
    runner = GenomeAgentRunner()
    await runner.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
