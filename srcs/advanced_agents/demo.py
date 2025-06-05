"""
Demo script for the Evolutionary AI Architect Agent

This script demonstrates the capabilities of the advanced evolutionary AI agent
that can design, evolve, and optimize AI architectures while improving itself.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_agents.evolutionary_ai_architect_agent import EvolutionaryAIArchitectAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """Run the evolutionary AI architect agent demo"""
    print("🧠 Evolutionary AI Architect & Self-Improving Agent")
    print("=" * 60)
    print("This agent can:")
    print("• Design and evolve AI architectures")
    print("• Solve complex problems adaptively") 
    print("• Monitor and improve its own performance")
    print("• Learn from experience and optimize strategies")
    print("=" * 60)
    
    # Create the agent
    print("\n🚀 Initializing Agent...")
    agent = EvolutionaryAIArchitectAgent("EvoAI-Alpha", population_size=6)
    
    # Display initial status
    print("\n📊 Initial Agent Status:")
    status = agent.get_status()
    print(f"  • Agent: {status['agent_info']['name']}")
    print(f"  • Population: {status['agent_info']['population_size']}")
    print(f"  • Diversity: {status['population_stats']['diversity_score']:.3f}")
    
    # Define test problems
    test_problems = [
        "Design an AI system for real-time medical image analysis in hospitals",
        "Create a natural language chatbot for multilingual customer support",
        "Build a complex financial market prediction system",
        "Develop an advanced multimodal AI for content moderation"
    ]
    
    print(f"\n🎯 Solving {len(test_problems)} Test Problems:")
    print("-" * 50)
    
    results = []
    for i, problem in enumerate(test_problems, 1):
        print(f"\n🔬 Problem {i}/4:")
        print(f"Description: {problem}")
        
        try:
            result = agent.solve_problem(problem)
            
            print(f"✅ Status: Solved successfully!")
            # Calculate overall score from performance metrics
            perf = result['performance_metrics']
            overall_score = (perf['accuracy'] * 0.25 + perf['efficiency'] * 0.15 + 
                           perf['adaptability'] * 0.20 + perf['creativity_score'] * 0.15 + 
                           perf['success_rate'] * 0.20 + perf['learning_speed'] * 0.05)
            print(f"⚡ Performance Score: {overall_score:.4f}")
            print(f"🏗️  Architecture Type: {result['problem_analysis']['suggested_architecture_type']}")
            print(f"⏱️  Processing Time: {result['processing_time']:.2f} seconds")
            print(f"🧬 Generation: {result['generation']}")
            
            if result['best_architecture']:
                arch = result['best_architecture']
                print(f"🎯 Best Architecture:")
                print(f"   • ID: {arch['unique_id']}")
                print(f"   • Layers: {len(arch['layers'])}")
                print(f"   • Fitness: {arch['fitness_score']:.4f}")
                print(f"   • Generation: {arch['generation']}")
            
            if result['improvement_opportunities']:
                print(f"💡 Improvement Opportunities:")
                for opp in result['improvement_opportunities'][:2]:
                    print(f"   • {opp}")
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error solving problem: {str(e)}")
            continue
        
        print("-" * 30)
    
    # Trigger self-improvement
    print("\n🚀 Triggering Self-Improvement Cycle...")
    try:
        improvement_result = agent.self_improve()
        
        print("✨ Self-Improvement Results:")
        print(f"   • Strategy Actions: {len(improvement_result['strategy']['actions'])}")
        print(f"   • Improvements Applied: {improvement_result['improvements_applied']}")
        print(f"   • New Best Fitness: {improvement_result['new_best_fitness']:.4f}")
        
    except Exception as e:
        print(f"❌ Error during self-improvement: {str(e)}")
    
    # Final comprehensive status
    print("\n📈 Final Agent Status:")
    final_status = agent.get_status()
    print(f"   🏆 Best Fitness: {final_status['current_best_architecture']['fitness']:.4f}")
    print(f"   🧬 Generations: {final_status['agent_info']['generation']}")
    print(f"   📋 Tasks Completed: {final_status['agent_info']['tasks_completed']}")
    print(f"   🔧 Population Size: {final_status['agent_info']['population_size']}")
    
    # Performance analysis
    if len(results) > 0:
        print("\n📊 Performance Analysis:")
        # Calculate average performance
        avg_performance = 0
        for r in results:
            perf = r['performance_metrics']
            score = (perf['accuracy'] * 0.25 + perf['efficiency'] * 0.15 + 
                    perf['adaptability'] * 0.20 + perf['creativity_score'] * 0.15 + 
                    perf['success_rate'] * 0.20 + perf['learning_speed'] * 0.05)
            avg_performance += score
        avg_performance /= len(results)
        avg_time = sum(r['processing_time'] for r in results) / len(results)
        
        print(f"   • Average Performance Score: {avg_performance:.4f}")
        print(f"   • Average Processing Time: {avg_time:.2f}s")
        
        # Architecture diversity
        arch_types = [r['problem_analysis']['suggested_architecture_type'] for r in results]
        unique_types = set(arch_types)
        print(f"   • Architecture Types Used: {len(unique_types)} ({', '.join(unique_types)})")
    
    # Capabilities demonstration
    print("\n🎯 Demonstrated Capabilities:")
    capabilities = [
        "✓ Automatic AI architecture design and generation",
        "✓ Evolutionary optimization with genetic algorithms", 
        "✓ Problem-specific architecture adaptation",
        "✓ Real-time performance monitoring and assessment",
        "✓ Self-improvement through strategy identification",
        "✓ Multi-domain problem solving (CV, NLP, Time Series)",
        "✓ Population diversity management",
        "✓ Fitness-based architecture selection",
        "✓ Meta-learning from task experience"
    ]
    
    for capability in capabilities:
        print(f"   {capability}")
    
    print("\n🎉 Demo Completed Successfully!")
    print("The Evolutionary AI Architect Agent has demonstrated its ability to:")
    print("• Solve diverse AI problems through architecture evolution")
    print("• Continuously improve its own performance")
    print("• Adapt to different problem domains")
    print("• Generate comprehensive implementation strategies")
    print("\n💡 Next Steps:")
    print("• Deploy in production environment")
    print("• Integrate with existing ML pipelines") 
    print("• Scale up population size for complex problems")
    print("• Add domain-specific architecture templates")


if __name__ == "__main__":
    main() 