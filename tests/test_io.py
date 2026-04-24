import os
from laser_modulator.io import read_file 

def test_read_file_correctly():
    # 1. Setup: Create a temporary dummy file to test
    dummy_filename = "dummy_params.txt"
    with open(dummy_filename, "w") as f:
        f.write("START\n")
        f.write("WL: 800e-9\n")
        f.write("END\n")
    
    # 2. Execution: Run your actual function
    parameters = read_file(dummy_filename)
    
    # 3. Assertion: Check if the function did its job
    assert "WL" in parameters
    assert parameters["WL"] == 800e-9
    
    # 4. Cleanup: Delete the dummy file
    os.remove(dummy_filename)
    

def test_read_file_ignores_data_without_start():
    # Setup: Create a file with no "START" marker
    dummy_filename = "bad_params.txt"
    with open(dummy_filename, "w") as f:
        f.write("WL: 800e-9\n") # Forgot to write START!
        f.write("END\n")
        
    # Execution
    parameters = read_file(dummy_filename)
    
    # Assertion: The dictionary should be completely empty
    assert len(parameters) == 0 
    assert "WL" not in parameters
    
    # Cleanup
    os.remove(dummy_filename)