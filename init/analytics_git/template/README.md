# Labskit Analytics

This is a starter repository for building an analytics project at Oliver Wyman. It defines the 
folder structure and provides the scaffolding for a new data analysis project in Python.


What do you get:

 1. Build a set of folders:
   - `config`: A folder for configuration information, settings files, and other properties
   - `data`: A folder for storing input data locally
   - `documentation`: A folder for project level documentation
   - `labskit`: A collection of core data and analytics utilities
   - `notebooks`: A folder for Jupyter notebooks. When you execute `jupyter notebook` from this 
   directory, you will automatically find all of your notebooks in this folder.
   - `outputs`: Finalized figures and reports/analysis.
   - `src`: The main code library for this project. Should contain all modules and scripts to 
   load data, perform analysis, and build reports.
   - `tests`: Unit tests for the code developed in `src`. Use the command `pytest` to execute the
    full test suite.
   
 2. Python environment specification via `requirements.txt`. Labskit will build the initial 
 environment for you. To activate it, use the command `source venv/bin/activate` on *nix machines
  and `venv/Scripts/activate.bat` on Windows computers (cmd).
 3. Initial project scaffolding. Loggers, error catchers, and settings specifications can be 
 found in the `labskit` folder directly. We recommend perusing the modules in there as
  to learn more about these tools.
 
 4. Git configurations and editor configurations. We ignore the standard ignore files and ensure 
 that cache objects and data files are not checked into git.
 
 5. An example workflow that loads the MMC stock price history, computes a rolling average, and 
 plots the result. To run the example workflow, run `python run.py` from this 
 directory. You will see the logging information print to the screen and see the success.
 
Enjoy!

### Contacts

David Strauss, Ian Lindblom, Dylan Hogg

