import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the Gradio application
from app.gradio_app import launch

if __name__ == "__main__":
    launch()
