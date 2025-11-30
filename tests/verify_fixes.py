import sys
import os
import re
from unittest.mock import MagicMock

# Mock heavy dependencies to avoid loading them
sys.modules["backend.db"] = MagicMock()
sys.modules["backend.llm"] = MagicMock()
sys.modules["backend.memory"] = MagicMock()
sys.modules["backend.media"] = MagicMock()
sys.modules["fastapi"] = MagicMock()
sys.modules["fastapi.middleware.cors"] = MagicMock()
sys.modules["fastapi.responses"] = MagicMock()
sys.modules["fastapi.staticfiles"] = MagicMock()
sys.modules["sse_starlette.sse"] = MagicMock()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the functions to test
# We need to be careful if core.py uses these mocks at module level
try:
    from backend.core import _clean_tts_text, _clean_display_text
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_cleaning():
    print("Running Token Cleaning Tests (Fast Mode)...")
    
    test_cases = [
        {
            "name": "Standard Generate Image",
            "input": "Here is a cat [GENERATE_IMAGE]",
            "expected_display": "Here is a cat",
            "expected_tts": "Here is a cat"
        },
        {
            "name": "Generate Image with Params",
            "input": "Here is a cat [GENERATE_IMAGE a cute cat]",
            "expected_display": "Here is a cat",
            "expected_tts": "Here is a cat"
        },
        {
            "name": "Emote Token",
            "input": "Hello [EMOTE: wave] world",
            "expected_display": "Hello world",
            "expected_tts": "Hello world"
        },
        {
            "name": "Mixed Tokens",
            "input": "Start [GENERATE_IMAGE] Middle [EMOTE: smile] End",
            "expected_display": "Start Middle End",
            "expected_tts": "Start Middle End"
        },
        {
            "name": "Asterisk Actions (Standard)",
            "input": "Hello *waves* world",
            "expected_display": "Hello waves world",
            "expected_tts": "Hello world"
        }
    ]

    failed = False
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        
        # Test Display Cleaning
        display_res = _clean_display_text(case['input'])
        if display_res != case['expected_display']:
            print(f"  [FAIL] Display Text Mismatch")
            print(f"    Input:    {case['input']}")
            print(f"    Expected: '{case['expected_display']}'")
            print(f"    Got:      '{display_res}'")
            failed = True
        else:
            print(f"  [PASS] Display Text")

        # Test TTS Cleaning
        tts_res = _clean_tts_text(case['input'])
        if tts_res != case['expected_tts']:
            print(f"  [FAIL] TTS Text Mismatch")
            print(f"    Input:    {case['input']}")
            print(f"    Expected: '{case['expected_tts']}'")
            print(f"    Got:      '{tts_res}'")
            failed = True
        else:
            print(f"  [PASS] TTS Text")

    if failed:
        print("\nSOME TESTS FAILED")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")
        sys.exit(0)

if __name__ == "__main__":
    test_cleaning()
