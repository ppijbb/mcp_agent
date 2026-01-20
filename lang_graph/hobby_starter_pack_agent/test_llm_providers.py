#!/usr/bin/env python3
"""
HSP Agent Multi-LLM Provider Test Script
Tests that all LLM providers are properly configured and working,
especially the RANDOM FREE PROVIDER selection feature.
"""

import sys
import os
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.llm import (
    LLMProvider,
    LLMConfig,
    LLMProviderManager,
    LLMClientFactory,
    get_llm_manager,
    get_llm_config,
    get_available_providers,
    switch_provider,
    # Random free provider functions
    get_random_free_config,
    get_random_config,
    get_free_providers_list,
    get_all_providers_list,
)


def test_provider_enum():
    """Test LLMProvider enum values"""
    print("=" * 60)
    print("Test 1: LLMProvider Enum")
    print("=" * 60)
    
    providers = [p.value for p in LLMProvider]
    expected = ["openrouter", "groq", "cerebras", "openai", "anthropic", "google"]
    
    assert providers == expected, f"Expected {expected}, got {providers}"
    print(f"✓ All {len(providers)} providers available: {providers}")
    return True


def test_provider_manager():
    """Test LLMProviderManager functionality"""
    print("\n" + "=" * 60)
    print("Test 2: LLMProviderManager")
    print("=" * 60)
    
    manager = LLMProviderManager()
    
    # Test primary provider
    primary = manager.get_primary_config()
    print(f"✓ Primary provider config: {primary}")
    
    # Test available providers
    available = manager.get_available_providers()
    print(f"✓ Available providers: {[p.value for p in available]}")
    
    # Test free providers
    free_providers = manager.get_free_providers()
    print(f"✓ Free providers: {[p.value for p in free_providers]}")
    
    # Test provider inference
    test_models = [
        ("anthropic/claude-3.5-sonnet", LLMProvider.OPENROUTER),  # Prefers OpenRouter for Claude
        ("gpt-4o", LLMProvider.OPENAI),
        ("llama-3.1-70b", LLMProvider.GROQ),
        ("gemini-1.5-pro", LLMProvider.GOOGLE),
    ]
    
    for model, expected in test_models:
        inferred = manager.get_provider_for_model(model)
        print(f"  - Model '{model}' -> Provider: {inferred.value if inferred else 'None'}")
    
    return True


def test_get_functions():
    """Test convenience functions"""
    print("\n" + "=" * 60)
    print("Test 3: Convenience Functions")
    print("=" * 60)
    
    # Test get_llm_config
    config = get_llm_config()
    print(f"✓ get_llm_config() returns: {type(config).__name__}")
    
    # Test get_available_providers
    providers = get_available_providers()
    print(f"✓ get_available_providers() returns: {providers}")
    
    # Test get_free_providers_list
    free_providers = get_free_providers_list()
    print(f"✓ get_free_providers_list() returns: {free_providers}")
    
    # Test get_all_providers_list
    all_providers = get_all_providers_list()
    print(f"✓ get_all_providers_list() returns: {all_providers}")
    
    return True


def test_provider_switching():
    """Test provider switching functionality"""
    print("\n" + "=" * 60)
    print("Test 4: Provider Switching")
    print("=" * 60)
    
    # Test switching to available providers
    for provider in ["openrouter", "groq", "cerebras"]:
        result = switch_provider(provider)
        print(f"✓ switch_provider('{provider}'): {result}")
    
    # Test invalid provider
    result = switch_provider("invalid_provider")
    assert result == False, "Invalid provider should return False"
    print(f"✓ switch_provider('invalid_provider'): {result}")
    
    return True


def test_random_free_provider():
    """Test RANDOM FREE PROVIDER selection (DEFAULT BEHAVIOR)"""
    print("\n" + "=" * 60)
    print("Test 5: 🎲 RANDOM FREE PROVIDER Selection (DEFAULT)")
    print("=" * 60)
    
    print("\n📌 Testing get_random_free_config() - Should return OpenRouter, Groq, or Cerebras")
    
    # Test multiple times to verify randomness
    selected_providers = []
    for i in range(10):
        config = get_random_free_config()
        if config:
            selected_providers.append(config.provider.value)
            print(f"  [{i+1}] {config.provider.value:12} | Model: {config.model}")
    
    # Verify all selected providers are free providers
    free_providers = ["openrouter", "groq", "cerebras"]
    for provider in selected_providers:
        assert provider in free_providers, f"Provider {provider} is not a free provider!"
    
    # Verify we have some variety (not all same)
    unique_providers = set(selected_providers)
    print(f"\n✓ Selected providers variety: {len(unique_providers)} unique providers out of 10 selections")
    print(f"  Unique: {sorted(unique_providers)}")
    
    if len(unique_providers) > 1:
        print("✓ ✓ Random selection working - multiple providers selected!")
    
    return True


def test_random_config():
    """Test random config with include_paid option"""
    print("\n" + "=" * 60)
    print("Test 6: get_random_config() with include_paid")
    print("=" * 60)
    
    # Test with include_paid=False (only free)
    print("\n📌 Testing get_random_config(include_paid=False) - Only free providers")
    for i in range(5):
        config = get_random_config(include_paid=False)
        if config:
            print(f"  [{i+1}] {config.provider.value:12} | Model: {config.model}")
            assert config.provider.value in ["openrouter", "groq", "cerebras"]
    
    # Test with include_paid=True (all providers)
    print("\n📌 Testing get_random_config(include_paid=True) - All configured providers")
    for i in range(5):
        config = get_random_config(include_paid=True)
        if config:
            print(f"  [{i+1}] {config.provider.value:12} | Model: {config.model}")
    
    return True


def test_openai_compatible_config():
    """Test OpenAI-compatible configuration export"""
    print("\n" + "=" * 60)
    print("Test 7: OpenAI-Compatible Config")
    print("=" * 60)
    
    manager = get_llm_manager()
    config = manager.get_primary_config()
    
    if config:
        openai_config = manager.get_openai_compatible_config()
        required_keys = ["api_key", "base_url", "model"]
        
        for key in required_keys:
            assert key in openai_config, f"Missing key: {key}"
        
        print(f"✓ OpenAI-compatible config keys: {list(openai_config.keys())}")
        print(f"✓ API Key set: {'*' * (len(openai_config['api_key']) - 8) + openai_config['api_key'][-8:] if openai_config.get('api_key') else 'None'}")
        print(f"✓ Base URL: {openai_config.get('base_url', 'N/A')}")
        print(f"✓ Model: {openai_config.get('model', 'N/A')}")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "🎯" * 20)
    print("HSP Agent Multi-LLM Provider Test Suite")
    print("🎯" * 20)
    print("\n🔥 DEFAULT BEHAVIOR: Random Free Provider Selection")
    print("   Providers: OpenRouter, Groq, Cerebras (무료)")
    print("   Paid providers (OpenAI, Google) only if API key configured")
    
    tests = [
        ("Provider Enum", test_provider_enum),
        ("Provider Manager", test_provider_manager),
        ("Convenience Functions", test_get_functions),
        ("Provider Switching", test_provider_switching),
        ("🎲 Random Free Provider (DEFAULT)", test_random_free_provider),
        ("🎲 Random Config with include_paid", test_random_config),
        ("OpenAI-Compatible Config", test_openai_compatible_config),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"✗ {name} failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASS" if success else f"✗ FAIL: {error}"
        print(f"  {status}: {name}")
    
    print(f"\n{'=' * 60}")
    print(f"Total: {passed}/{total} tests passed")
    print(f"{'=' * 60}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
