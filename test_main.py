import os
import math
from signal_processing_app import Signal, SignalOperations  # Ensure these classes are correctly imported

def Compare_Signals(file_name, Your_indices, Your_samples):      
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()  # Line 1
        line = f.readline()  # Line 2
        line = f.readline()  # Line 3 (Number of samples)
        line = f.readline()  # Line 4 (First data line)
        while line:
            L = line.strip()
            if len(L.split(' ')) == 2:
                parts = L.split(' ')
                V1 = int(parts[0])
                V2 = float(parts[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break

    print("Current Output Test file is: ")
    print(file_name)
    print("\n")

    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        print("Test case failed, your signal has different length from the expected one")
        return

    for i in range(len(Your_indices)):
        if Your_indices[i] != expected_indices[i]:
            print("Test case failed, your signal has different indices from the expected one") 
            return
    # print("expected values: ", expected_samples)
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal has different values from the expected one") 
            return

    print("Test case passed successfully")


def load_signal(file_path):
    """
    Helper function to load a signal from a given file path.
    """
    try:
        signal = Signal.from_file(file_path)
        return signal
    except Exception as e:
        print(f"Failed to load signal from {file_path}: {e}")
        return None


def test_point1_correlation(testcase_dir):
    """
    Test Point1: Correlation
    Steps:
        1. Load Corr_input_signal1.txt and Corr_input_signal2.txt
        2. Compute correlation using SignalOperations.correlate()
        3. Extract only non-negative lags (0 to +4)
        4. Compare the result with CorrOutput.txt using Compare_Signals()
    """
    print("\n=== Testing Point1: Correlation ===")
    signal1_path = os.path.join(testcase_dir, "Corr_input signal1.txt")
    signal2_path = os.path.join(testcase_dir, "Corr_input signal2.txt")
    expected_corr_path = os.path.join(testcase_dir, "CorrOutput.txt")

    signal1 = load_signal(signal1_path)
    signal2 = load_signal(signal2_path)

    if signal1 is None or signal2 is None:
        print("Failed to load input signals for Correlation Test.")
        return

    # Compute normalized correlation
    corr_signal = SignalOperations.correlate(signal1, signal2)
    print(f"Computed Correlation Signal: Indices={corr_signal.indices}, Samples={corr_signal.samples}")

    # Extract only non-negative lags (0 to +4)
    try:
        zero_lag_index = corr_signal.indices.index(0)
    except ValueError:
        print("Lag 0 not found in computed correlation.")
        return

    # Ensure there are enough lags
    if zero_lag_index + 4 >= len(corr_signal.samples):
        print("Not enough lags in computed correlation to extract lags 0 to +4.")
        return

    lags_to_compare = list(range(0, 5))  # 0,1,2,3,4
    indices_to_compare = [corr_signal.indices[zero_lag_index + m] for m in lags_to_compare]
    samples_to_compare = [corr_signal.samples[zero_lag_index + m] for m in lags_to_compare]

    print(f"Extracted Non-negative Correlation Samples: Indices={indices_to_compare}, Samples={samples_to_compare}")

    # Compare with expected
    Compare_Signals(expected_corr_path, indices_to_compare, samples_to_compare)


def test_point2_time_analysis(testcase_dir):
    """
    Test Point2: Time Delay Estimation
    Steps:
        1. Load TD_input_signal1.txt and TD_input_signal2.txt
        2. Read Fs and expected time delay from Fs_and_expected_output.txt
        3. Compute correlation
        4. Estimate time delay
        5. Compare with expected time delay
    """
    print("\n=== Testing Point2: Time Analysis (Time Delay Estimation) ===")
    signal1_path = os.path.join(testcase_dir, "TD_input_signal1.txt")
    signal2_path = os.path.join(testcase_dir, "TD_input_signal2.txt")
    specs_path = os.path.join(testcase_dir, "Fs_and_expected_output.txt")

    signal1 = load_signal(signal1_path)
    signal2 = load_signal(signal2_path)

    if signal1 is None or signal2 is None:
        print("Failed to load input signals for Time Analysis Test.")
        return

    # Read Fs and expected time delay
    Fs = None
    expected_delay = None
    try:
        with open(specs_path, 'r') as f:
            for line in f:
                if "Fs=" in line:
                    Fs = float(line.strip().split('=')[1])
                elif "Excpected output =" in line:
                    delay_str = line.strip().split('=')[1].strip()
                    # Handle fraction like '5/100'
                    if '/' in delay_str:
                        numerator, denominator = delay_str.split('/')
                        expected_delay = float(numerator) / float(denominator)
                    else:
                        expected_delay = float(delay_str)
    except Exception as e:
        print(f"Failed to read specs from {specs_path}: {e}")
        return

    if Fs is None or expected_delay is None:
        print("Failed to read Fs or expected time delay from specs.")
        return

    # Compute correlation
    corr_signal = SignalOperations.correlate(signal1, signal2)

    # Estimate time delay
    delay = SignalOperations.time_delay_from_correlation(corr_signal)
    # Convert delay index to time using Fs
    delay_time = delay / Fs
    print(f"Estimated Time Delay: {delay_time} seconds")

    # Compare with expected
    print(f"Expected Time Delay: {expected_delay} seconds")
    if abs(delay_time - expected_delay) < 0.01:
        print("Time Delay Test Passed successfully")
    else:
        print("Time Delay Test Failed")


def load_class_templates(class_dir):
    """
    Load all template signals for a given class.
    Returns a list of Signal objects.
    """
    templates = []
    for filename in os.listdir(class_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(class_dir, filename)
            signal = load_signal(file_path)
            if signal is not None:
                templates.append(signal)
    return templates


def test_point3_classification(testcase_dir):
    """
    Test Point3: Classification
    Steps:
        1. Load template signals from Class1 and Class2 subdirectories
        2. Load test signals from TestSignals subdirectory
        3. For each test signal, perform classification using template matching
        4. Compare with expected classification if available
    """
    print("\n=== Testing Point3: Classification ===")
    instructions_path = os.path.join(testcase_dir, "instructions.txt")
    class1_dir = os.path.join(testcase_dir, "Class1")
    class2_dir = os.path.join(testcase_dir, "Class2")
    test_signals_dir = os.path.join(testcase_dir, "TestSignals")

    # Load template signals
    class1_templates = load_class_templates(class1_dir)
    class2_templates = load_class_templates(class2_dir)

    if not class1_templates or not class2_templates:
        print("Failed to load template signals for Classification Test.")
        return

    # Load test signals
    test_signals = []
    expected_classes = []
    # Assuming Test1.txt is 'down' (Class1) and Test2.txt is 'up' (Class2)
    # This assumption is based on instructions.txt
    test1_path = os.path.join(test_signals_dir, "Test1.txt")
    test2_path = os.path.join(test_signals_dir, "Test2.txt")
    test1 = load_signal(test1_path)
    test2 = load_signal(test2_path)
    if test1:
        test_signals.append(("Test1.txt", test1, "Class 1"))  # Assuming Test1 is Class 1
    if test2:
        test_signals.append(("Test2.txt", test2, "Class 2"))  # Assuming Test2 is Class 2

    if not test_signals:
        print("No test signals found for Classification Test.")
        return

    # Perform classification for each test signal
    for test_name, test_signal, expected_class in test_signals:
        print(f"\nClassifying {test_name}...")

        # Compute average correlation with Class1 templates
        corr_class1 = []
        for template in class1_templates:
            corr = SignalOperations.correlate(test_signal, template)
            max_corr = max(corr.samples, key=abs)
            corr_class1.append(abs(max_corr))

        avg_corr_class1 = sum(corr_class1) / len(corr_class1)

        # Compute average correlation with Class2 templates
        corr_class2 = []
        for template in class2_templates:
            corr = SignalOperations.correlate(test_signal, template)
            max_corr = max(corr.samples, key=abs)
            corr_class2.append(abs(max_corr))

        avg_corr_class2 = sum(corr_class2) / len(corr_class2)

        # Decide class based on higher average correlation
        predicted_class = "Class 1" if avg_corr_class1 > avg_corr_class2 else "Class 2"
        print(f"Predicted Class: {predicted_class}")
        print(f"Expected Class: {expected_class}")

        # Compare with expected
        if predicted_class == expected_class:
            print(f"Classification Test for {test_name} Passed successfully")
        else:
            print(f"Classification Test for {test_name} Failed")


def run_all_tests():
    """
    Aggregates all test points and runs them sequentially.
    """
    base_dir = "Correlation Task Files"  # Adjust this path if your tests directory is elsewhere

    # Define test points
    test_points = [
        ("Point1 Correlation", test_point1_correlation),
        ("Point2 TimeAnalysis", test_point2_time_analysis),
        ("Point3 Classification", test_point3_classification),
    ]

    for folder_name, test_func in test_points:
        testcase_dir = os.path.join(base_dir, folder_name)
        if not os.path.exists(testcase_dir):
            print(f"Test directory {testcase_dir} does not exist. Skipping.")
            continue
        print(f"\n=== Running {folder_name} ===")
        test_func(testcase_dir)

    print("\nAll tests completed.")


def test_filter_coefficients(testcase_dir):
    """
    Tests FIR filter coefficients against expected coefficients file.
    """
    print(f"--- Testing Filter Coefficients in {testcase_dir} ---")
    specs_file = os.path.join(testcase_dir, "Filter Specifications.txt")
    if not os.path.exists(specs_file):
        print(f"Error: {specs_file} does not exist.\n")
        return
    
    # Load filter specifications
    specs = {}
    with open(specs_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if '=' in line:
                key, val = line.split('=')
                specs[key.strip()] = val.strip()
    
    # Design FIR filter
    fir_filter = SignalOperations.design_fir_filter(specs)
    
    # Determine expected coefficients file name based on FilterType
    filter_type = specs.get('FilterType', 'Low pass').lower()
    if 'low pass' in filter_type or 'low' in filter_type:
        expected_file = os.path.join(testcase_dir, "LPFCoefficients.txt")
    elif 'high pass' in filter_type or 'high' in filter_type:
        expected_file = os.path.join(testcase_dir, "HPFCoefficients.txt")
    elif 'band pass' in filter_type or 'bandpass' in filter_type:
        expected_file = os.path.join(testcase_dir, "BPFCoefficients.txt")
    elif 'band stop' in filter_type or 'bandstop' in filter_type:
        expected_file = os.path.join(testcase_dir, "BSFCoefficients.txt")
    else:
        print(f"Unknown FilterType '{filter_type}' in {specs_file}. Skipping.")
        return
    
    if not os.path.exists(expected_file):
        print(f"Error: Expected coefficients file {expected_file} does not exist.\n")
        return
    
    # Compare designed coefficients with expected coefficients
    Compare_Signals(expected_file, fir_filter.indices, fir_filter.samples)

def test_filter_application(testcase_dir):
    """
    Tests the filtered output signal against expected filtered signal file.
    """
    print(f"--- Testing Filter Application in {testcase_dir} ---")
    specs_file = os.path.join(testcase_dir, "Filter Specifications.txt")
    input_signal_file = ""
    expected_filtered_file = ""
    
    if not os.path.exists(specs_file):
        print(f"Error: {specs_file} does not exist.\n")
        return
    
    # Load filter specifications
    specs = {}
    with open(specs_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if '=' in line:
                key, val = line.split('=')
                specs[key.strip()] = val.strip()
    
    # Determine input signal file and expected filtered file
    # Assuming the input signal file is named consistently, e.g., 'InputSignal.txt' or 'ecg400.txt'
    # And the expected output is '<InputSignal>_<FilterType>_filtered.txt'
    # You may need to adjust this logic based on your actual filenames
    
    # Find input signal file
    # Possible input signal filenames based on your Testcase7 and Testcase8
    possible_input_files = ["InputSignal.txt", "ecg400.txt"]
    for fname in possible_input_files:
        fpath = os.path.join(testcase_dir, fname)
        if os.path.exists(fpath):
            input_signal_file = fpath
            break
    if not input_signal_file:
        print(f"Error: No recognized input signal file found in {testcase_dir}.\n")
        return
    
    # Load input signal
    input_signal = Signal.from_file(input_signal_file)
    
    # Design FIR filter
    fir_filter = SignalOperations.design_fir_filter(specs)
    
    # Apply filter (time-domain convolution)
    filtered_signal = SignalOperations.filter_signal(input_signal, fir_filter, method='time')
    
    # Determine expected filtered file name based on FilterType and input signal
    filter_type = specs.get('FilterType', 'Low pass').lower()
    input_signal_basename = os.path.splitext(os.path.basename(input_signal_file))[0]
    
    if 'low pass' in filter_type or 'low' in filter_type:
        expected_filtered_file = os.path.join(testcase_dir, f"{input_signal_basename}_low_pass_filtered.txt")
    elif 'high pass' in filter_type or 'high' in filter_type:
        expected_filtered_file = os.path.join(testcase_dir, f"{input_signal_basename}_high_pass_filtered.txt")
    elif 'band pass' in filter_type or 'bandpass' in filter_type:
        expected_filtered_file = os.path.join(testcase_dir, f"{input_signal_basename}_band_pass_filtered.txt")
    elif 'band stop' in filter_type or 'bandstop' in filter_type:
        expected_filtered_file = os.path.join(testcase_dir, f"{input_signal_basename}_band_stop_filtered.txt")
    else:
        print(f"Unknown FilterType '{filter_type}' in {specs_file}. Skipping.")
        return
    
    if not os.path.exists(expected_filtered_file):
        print(f"Error: Expected filtered signal file {expected_filtered_file} does not exist.\n")
        return
    
    # Compare filtered signal with expected filtered signal
    Compare_Signals(expected_filtered_file, filtered_signal.indices, filtered_signal.samples)

def run_all_filter_tests():
    """
    Runs all testcases (1 through 8) by determining the type of test and executing accordingly.
    """
    base_dir = "FIR test cases"  
    
    # List of testcases
    testcases = [f"Testcase {i}" for i in range(1, 9)]  # Testcase 1 to Testcase 8
    
    for tc in testcases:
        testcase_dir = os.path.join(base_dir, tc)
        if not os.path.isdir(testcase_dir):
            print(f"Warning: {testcase_dir} is not a directory. Skipping.")
            continue
        
        # Read readme.txt to determine the type of test
        readme_file = os.path.join(testcase_dir, "readme.txt")
        if not os.path.exists(readme_file):
            print(f"Warning: {readme_file} does not exist. Skipping {tc}.")
            continue
        
        # Simple parsing to determine expected output
        with open(readme_file, 'r') as f:
            readme_content = f.read().lower()
        
        if "coefficients" in readme_content:
            # It's a filter coefficients test
            test_filter_coefficients(testcase_dir)
        elif "filtered" in readme_content:
            # It's a filter application test
            test_filter_application(testcase_dir)
        else:
            print(f"Warning: Unable to determine test type for {tc} from readme.txt. Skipping.")
    
    print("All testcases executed.\n")


if __name__ == "__main__":
    run_all_filter_tests()
