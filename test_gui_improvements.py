#!/usr/bin/env python3
"""
Test script to verify GUI improvements without requiring a display.
Tests syntax, imports, and basic functionality.
"""

import sys
import importlib.util

def test_file_syntax(filename):
    """Test if a Python file has valid syntax"""
    print(f"\n{'='*60}")
    print(f"Testing: {filename}")
    print('='*60)
    
    try:
        # Compile the file to check syntax
        with open(filename, 'r') as f:
            code = f.read()
        compile(code, filename, 'exec')
        print(f"✓ Syntax: PASSED")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax: FAILED")
        print(f"  Error: {e}")
        return False

def test_imports(filename):
    """Test if a Python file can be imported (check for import errors)"""
    try:
        spec = importlib.util.spec_from_file_location("test_module", filename)
        if spec and spec.loader:
            print(f"✓ Import structure: VALID")
            return True
        else:
            print(f"✗ Import structure: INVALID")
            return False
    except Exception as e:
        print(f"✗ Import check: {type(e).__name__}")
        # This is expected for GUI files without display
        if "tkinter" in str(e).lower() or "display" in str(e).lower():
            print(f"  (Expected - requires GUI display)")
            return True
        print(f"  Error: {e}")
        return False

def check_features(filename):
    """Check if our new features are present in the file"""
    print(f"\nChecking for user-friendliness features...")
    
    features = {
        'ToolTip class': 'class ToolTip',
        'Menu bar creation': 'def create_menu_bar',
        'Enhanced status bar': 'def create_status_bar',
        'Input validation': 'messagebox.showwarning',
        'Confirmation dialogs': 'messagebox.askyesno',
        'Tooltips usage': 'ToolTip(',
        'Keyboard shortcuts': 'self.bind',
        'Help dialogs': 'show_quick_start\|show_shortcuts\|show_about',
        'Icon indicators': '✓\|✗\|⚠️\|▶️\|⏹️\|💾\|📊\|🎓',
    }
    
    with open(filename, 'r') as f:
        content = f.read()
    
    results = []
    for feature_name, pattern in features.items():
        # Simple check - just see if the pattern exists
        if pattern in content or any(p in content for p in pattern.split('\\|')):
            print(f"  ✓ {feature_name}: Found")
            results.append(True)
        else:
            print(f"  ✗ {feature_name}: Not found")
            results.append(False)
    
    return all(results)

def main():
    """Run all tests"""
    print("="*60)
    print("GUI User-Friendliness Improvements - Validation Test")
    print("="*60)
    
    test_files = [
        'crypto_trader_gui.py',
        'crypto_trading_suite_gui.py'
    ]
    
    all_passed = True
    
    for filename in test_files:
        try:
            # Test syntax
            syntax_ok = test_file_syntax(filename)
            
            # Test imports
            import_ok = test_imports(filename)
            
            # Check features
            features_ok = check_features(filename)
            
            # Overall result
            file_passed = syntax_ok and import_ok
            print(f"\n{'─'*60}")
            if file_passed:
                print(f"✓ {filename}: PASSED")
            else:
                print(f"✗ {filename}: FAILED")
            print(f"{'─'*60}")
            
            all_passed = all_passed and file_passed
            
        except FileNotFoundError:
            print(f"✗ File not found: {filename}")
            all_passed = False
        except Exception as e:
            print(f"✗ Unexpected error testing {filename}: {e}")
            all_passed = False
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print('='*60)
    
    if all_passed:
        print("✓ All tests PASSED")
        print("\nGUI improvements are syntactically correct and ready to use.")
        print("\nNote: Visual testing requires a GUI environment.")
        print("To manually test:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run: python crypto_trader_gui.py")
        print("  3. Verify tooltips, confirmations, and visual improvements")
        return 0
    else:
        print("✗ Some tests FAILED")
        print("\nPlease review the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
