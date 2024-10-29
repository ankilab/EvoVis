from dotenv import load_dotenv
import os
import sys
import subprocess

##################################################

# MAIN MODULE EVOVIS

# The python script stores the ENAS run results path in an environmental variable
# After that it executes the Dash web app.

##################################################


### RUN RESULTS PATH
if len(sys.argv) == 2:
    with open('.env', 'w') as env:
        env.write(f'RUN_RESULTS_PATH={sys.argv[1]}\n')

else:
    load_dotenv()
    
    if 'RUN_RESULTS_PATH' in os.environ:
        
        RUN = os.getenv("RUN_RESULTS_PATH")
        print(f"ENAS run results files: {RUN}")
        
    else:
        print("Error: Please run the dashboard with the run results directory path specified after 'python3 app.py' or specify the environmental variable 'RUN_RESULTS_PATH'.")
        sys.exit(1)
        
### EXECUTE APP
if __name__ == "__main__":
    app = "./src/app.py"  
    
    try:
        subprocess.run([sys.executable, app], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing script: {e}")
