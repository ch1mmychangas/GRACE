## Setup
```bash
# 1. Enter the vision directory.
cd vision
# 2. Set up your virtual environment. (venv)
python -m venv .venv
# 3. Activate the virtual environment.

    # On Windows (Command Prompt):
    venv\Scripts\activate

    # On Windows (PowerShell):
    .\venv\Scripts\Activate.ps1
    
    # On macOS / Linux:
    source venv/bin/activate

# 4. Install the dependencies:
python -m pip install -r requirements.txt
```

Note: We use `python -m` as a prefix for pip to tell the system to use the virtual environment's pip`, and not your system's global pip.


### Test the Model
```bash
python test.py # no `-m` needed
```