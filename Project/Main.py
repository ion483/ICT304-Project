import subprocess
import sys 

def run_script(script_name):
    print(f"--- Executing {script_name} ---")

    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")
        return False
    
    return True

scripts = ["SubsystemA.py", "SubsystemB.py", "SubsystemC.py"]

for script in scripts:
    success = run_script(script)

    if not success:
        print("Stopping execution due to error.")
        break

print("Full system run completed.")