# test_system_complete.py - Complete system test
# Audit Mode: Testing all system components
# Path: C:\Users\anapa\eziofilho-unified\05_testing_validation
# User: marcobarreto007
# Date: 2025-05-24 16:36:01 UTC

import sys
import os
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def print_test_header():
    """Print test header with timestamp"""
    print("=" * 80)
    print("🧪 EZIOFILHO SYSTEM - COMPLETE TEST SUITE")
    print(f"📅 Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"👤 User: marcobarreto007")
    print(f"📁 Path: {Path.cwd()}")
    print("=" * 80)

def test_imports():
    """Test 1: Import all required modules"""
    print("\n[TEST 1] Testing Imports...")
    print("-" * 40)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test core imports
    try:
        from core.multilingual_responses import FACTUAL_ANSWERS
        print("✅ multilingual_responses imported")
        tests_passed += 1
    except Exception as e:
        print(f"❌ multilingual_responses error: {e}")
        tests_failed += 1
        
    # Test main system
    try:
        from ezio_complete_system_fixed import EzioCompleteSystem
        print("✅ Main system imported")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Main system error: {e}")
        tests_failed += 1
        
    # Test dependencies
    modules = ["requests", "dotenv", "json", "re"]
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module} available")
            tests_passed += 1
        except:
            print(f"❌ {module} missing")
            tests_failed += 1
            
    print(f"\nResult: {tests_passed} passed, {tests_failed} failed")
    return tests_passed, tests_failed

def test_multilingual_responses():
    """Test 2: Test multilingual responses"""
    print("\n[TEST 2] Testing Multilingual Responses...")
    print("-" * 40)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        from core.multilingual_responses import FACTUAL_ANSWERS
        
        # Test Portuguese responses
        test_queries = {
            "pt": ["quem é você", "bom dia", "ajuda"],
            "en": ["who are you", "good morning", "help"],
            "fr": ["qui es-tu", "bonjour", "aide"]
        }
        
        for lang, queries in test_queries.items():
            print(f"\nTesting {lang.upper()}:")
            for query in queries:
                if lang in FACTUAL_ANSWERS and query in FACTUAL_ANSWERS[lang]:
                    response = FACTUAL_ANSWERS[lang][query]
                    print(f"  ✅ '{query}' -> {len(response)} chars")
                    tests_passed += 1
                else:
                    print(f"  ❌ '{query}' -> Not found")
                    tests_failed += 1
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        tests_failed += 1
        
    print(f"\nResult: {tests_passed} passed, {tests_failed} failed")
    return tests_passed, tests_failed

def test_system_initialization():
    """Test 3: Initialize main system"""
    print("\n[TEST 3] Testing System Initialization...")
    print("-" * 40)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        from ezio_complete_system_fixed import EzioCompleteSystem
        
        print("Creating system instance...")
        system = EzioCompleteSystem()
        print("✅ System created successfully")
        tests_passed += 1
        
        # Test language detection
        test_phrases = {
            "Olá, bom dia!": "pt",
            "Hello, good morning!": "en",
            "Bonjour, comment allez-vous?": "fr",
            "qual o preço do bitcoin": "pt",
            "what is the price of bitcoin": "en"
        }
        
        print("\nTesting language detection:")
        for phrase, expected_lang in test_phrases.items():
            detected = system.detect_language(phrase)
            if detected == expected_lang:
                print(f"  ✅ '{phrase[:20]}...' -> {detected}")
                tests_passed += 1
            else:
                print(f"  ❌ '{phrase[:20]}...' -> {detected} (expected {expected_lang})")
                tests_failed += 1
                
    except Exception as e:
        print(f"❌ System error: {e}")
        tests_failed += 1
        
    print(f"\nResult: {tests_passed} passed, {tests_failed} failed")
    return tests_passed, tests_failed

def test_api_connectivity():
    """Test 4: Test API connectivity"""
    print("\n[TEST 4] Testing API Connectivity...")
    print("-" * 40)
    
    tests_passed = 0
    tests_failed = 0
    
    import requests
    
    # Test CoinGecko ping
    try:
        response = requests.get("https://api.coingecko.com/api/v3/ping", timeout=5)
        if response.status_code == 200:
            print("✅ CoinGecko API: Connected")
            tests_passed += 1
        else:
            print(f"❌ CoinGecko API: Status {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"❌ CoinGecko API: {e}")
        tests_failed += 1
        
    # Test Bitcoin price
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": "bitcoin", "vs_currencies": "usd,brl"}
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if "bitcoin" in data:
                btc_usd = data["bitcoin"].get("usd", 0)
                btc_brl = data["bitcoin"].get("brl", 0)
                print(f"✅ Bitcoin Price: ${btc_usd:,.2f} / R$ {btc_brl:,.2f}")
                tests_passed += 1
            else:
                print("❌ Bitcoin data not found")
                tests_failed += 1
        else:
            print(f"❌ Bitcoin API: Status {response.status_code}")
            tests_failed += 1
            
    except Exception as e:
        print(f"❌ Bitcoin API error: {e}")
        tests_failed += 1
        
    print(f"\nResult: {tests_passed} passed, {tests_failed} failed")
    return tests_passed, tests_failed

def test_query_processing():
    """Test 5: Test query processing"""
    print("\n[TEST 5] Testing Query Processing...")
    print("-" * 40)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        from ezio_complete_system_fixed import EzioCompleteSystem
        system = EzioCompleteSystem()
        
        # Test queries
        test_queries = [
            ("bom dia", "greeting"),
            ("quem é você", "identity"),
            ("bitcoin", "crypto"),
            ("help", "help"),
            ("bonjour", "greeting")
        ]
        
        for query, expected_type in test_queries:
            print(f"\nTesting: '{query}'")
            try:
                response = system.process_query(query)
                if response and len(response) > 0:
                    print(f"✅ Response: {response[:50]}...")
                    tests_passed += 1
                else:
                    print(f"❌ Empty response")
                    tests_failed += 1
            except Exception as e:
                print(f"❌ Error: {e}")
                tests_failed += 1
                
    except Exception as e:
        print(f"❌ System error: {e}")
        tests_failed += 1
        
    print(f"\nResult: {tests_passed} passed, {tests_failed} failed")
    return tests_passed, tests_failed

def generate_report(results):
    """Generate final test report"""
    print("\n" + "=" * 80)
    print("📊 FINAL TEST REPORT")
    print("=" * 80)
    
    total_passed = sum(r[0] for r in results)
    total_failed = sum(r[1] for r in results)
    total_tests = total_passed + total_failed
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"✅ Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)")
    print(f"❌ Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)")
    
    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED! System is ready!")
    elif total_failed <= 2:
        print("\n⚠️  Minor issues detected. System mostly functional.")
    else:
        print("\n❌ Critical issues found. Please fix before using.")
        
    print("\n💡 Next steps:")
    print("1. Run: py ezio_complete_system_fixed.py")
    print("2. Test queries: 'bom dia', 'bitcoin', 'help'")
    print("3. Check crypto prices work correctly")

def main():
    """Run all tests"""
    print_test_header()
    
    # Run all test suites
    results = []
    results.append(test_imports())
    results.append(test_multilingual_responses())
    results.append(test_system_initialization())
    results.append(test_api_connectivity())
    results.append(test_query_processing())
    
    # Generate report
    generate_report(results)
    
    print("\n✅ Test suite completed!")
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()