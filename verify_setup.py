"""
verify_setup.py
Run this FIRST, right after `pip install -r requirements.txt`, to catch
environment problems before they show up as confusing errors halfway
through the pipeline.

Run:
    python verify_setup.py
"""

import importlib
import sys

REQUIRED = ["pandas", "numpy", "sklearn", "xgboost", "shap", "streamlit",
            "plotly", "mlxtend", "matplotlib", "joblib"]
OPTIONAL = ["lightgbm", "ucimlrepo", "requests"]


def check(name):
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown version")
        print(f"  ✅ {name:<15} {version}")
        return True
    except ImportError as e:
        print(f"  ❌ {name:<15} NOT INSTALLED  ({e})")
        return False


def main():
    print(f"Python: {sys.version.split()[0]}\n")

    print("Required packages:")
    all_ok = all([check(name) for name in REQUIRED])

    print("\nOptional packages (pipeline works without these):")
    for name in OPTIONAL:
        check(name)

    print()
    if all_ok:
        print("✅ All required packages installed. You're good to run the pipeline:")
        print("   python src/download_data.py")
        print("   python src/preprocessing.py")
        print("   python src/feature_engineering.py")
        print("   python src/forecasting.py")
        print("   python src/explainability.py")
        print("   python src/segmentation.py")
        print("   python src/recommendation.py")
        print("   streamlit run app.py")
    else:
        print("❌ Some required packages are missing. Fix with:")
        print("   pip install -r requirements.txt")
        print("\nIf a specific package fails to install, try installing it alone")
        print("to see the real error, e.g.:")
        print("   pip install shap")
        sys.exit(1)


if __name__ == "__main__":
    main()
