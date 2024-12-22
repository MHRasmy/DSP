import numpy as np
import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Signal Class
class Signal:
    def __init__(self, indices=None, samples=None, time_values=None):
        self.indices = indices if indices is not None else []
        self.samples = samples if samples is not None else []
        self.time_values = time_values if time_values is not None else []

    @classmethod
    def from_file(cls, file_path):
        indices = []
        samples = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) < 4:
                raise ValueError(f"File {file_path} format incorrect. Expected at least 4 lines.")
            try:
                N = int(lines[2].strip())  # Number of samples
            except ValueError:
                raise ValueError(f"File {file_path} has an invalid number of samples on line 3.")
            
            if len(lines) < 3 + N:
                raise ValueError(f"File {file_path} does not contain enough sample lines. Expected {N} samples.")
            
            for i, line in enumerate(lines[3:3+N], start=4):
                parts = line.strip().split()
                if len(parts) != 2:
                    print(f"Skipping malformed line {i} in {file_path}: '{line.strip()}'")
                    continue  # Skip malformed lines
                try:
                    idx = int(parts[0])
                    val = float(parts[1])
                    indices.append(idx)
                    samples.append(val)
                except ValueError:
                    print(f"Skipping line {i} due to invalid data types in {file_path}: '{line.strip()}'")
                    continue  # Skip lines with invalid data types
        return cls(indices, samples)

    def to_file(self, file_path):
        with open(file_path, 'w') as f:
            f.write("0\n0\n{}\n".format(len(self.samples)))
            for idx, val in zip(self.indices, self.samples):
                f.write(f"{idx} {val}\n")

    @classmethod
    def generate_sine(cls, amplitude, phase_shift, frequency, sampling_freq):
        N = 50  # Fixed number of samples
        t = np.arange(N) / sampling_freq  # Time vector
        samples = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)
        indices = list(range(N))
        return cls(indices, samples.tolist(), t.tolist())

    @classmethod
    def generate_cosine(cls, amplitude, phase_shift, frequency, sampling_freq):
        N = 50  # Fixed number of samples
        t = np.arange(N) / sampling_freq  # Time vector
        samples = amplitude * np.cos(2 * np.pi * frequency * t + phase_shift)
        indices = list(range(N))
        return cls(indices, samples.tolist(), t.tolist())

# Signal Operations
class SignalOperations:
    @staticmethod
    def add(signal1, signal2):
        # Create a sorted list of all unique indices
        all_indices = sorted(set(signal1.indices) | set(signal2.indices))
        
        # Create dictionaries for quick lookup
        samples1 = dict(zip(signal1.indices, signal1.samples))
        samples2 = dict(zip(signal2.indices, signal2.samples))
        
        # Perform addition with alignment
        added_samples = []
        for idx in all_indices:
            val1 = samples1.get(idx, 0)
            val2 = samples2.get(idx, 0)
            added_samples.append(val1 + val2)
        
        return Signal(all_indices, added_samples)

    @staticmethod
    def subtract(signal1, signal2):
        # Create a sorted list of all unique indices
        all_indices = sorted(set(signal1.indices) | set(signal2.indices))
        
        # Create dictionaries for quick lookup
        samples1 = dict(zip(signal1.indices, signal1.samples))
        samples2 = dict(zip(signal2.indices, signal2.samples))
        
        # Perform subtraction with alignment
        subtracted_samples = []
        for idx in all_indices:
            val1 = samples1.get(idx, 0)
            val2 = samples2.get(idx, 0)
            subtracted_samples.append(val1 - val2)
        
        return Signal(all_indices, subtracted_samples)

    @staticmethod
    def multiply(signal, constant):
        multiplied_samples = [s * constant for s in signal.samples]
        return Signal(signal.indices, multiplied_samples)

    @staticmethod
    def shift(signal, k):
        shifted_indices = [idx - k for idx in signal.indices]
        return Signal(shifted_indices, signal.samples)

    @staticmethod
    def fold(signal):
        folded_indices = list(reversed([-idx for idx in signal.indices]))
        folded_samples = list(reversed(signal.samples))
        return Signal(folded_indices, folded_samples)
    
    # Task 4: Moving Average + First Derivative + Second Derivative + Convolution
    @staticmethod
    def moving_average(signal, window_size):
        n_samples = len(signal.samples)
        if n_samples < window_size:
            return Signal([], [])
        indices = signal.indices[:n_samples - window_size + 1]
        samples = []
        for i in range(n_samples - window_size + 1):
            avg = sum(signal.samples[i:i + window_size]) / window_size
            samples.append(avg)
        return Signal(indices, samples)

    @staticmethod
    def first_derivative(signal):
        indices = signal.indices[:-1]
        samples = []
        for i in range(len(signal.samples) - 1):
            derivative = signal.samples[i + 1] - signal.samples[i]
            samples.append(derivative)
        return Signal(indices, samples)

    @staticmethod
    def second_derivative(signal):
        indices = signal.indices[:-2]
        samples = []
        for i in range(len(signal.samples) - 2):
            derivative = signal.samples[i + 2] - 2 * signal.samples[i + 1] + signal.samples[i]
            samples.append(derivative)
        return Signal(indices, samples)

    @staticmethod
    def convolve(signal1, signal2):
        x = signal1.samples
        h = signal2.samples
        y_length = len(x) + len(h) - 1
        y = [0] * y_length
        for i in range(len(x)):
            for j in range(len(h)):
                y[i + j] += x[i] * h[j]
        # Compute the indices for the convolution result
        start_index = signal1.indices[0] + signal2.indices[0]
        indices = [start_index + i for i in range(y_length)]
        return Signal(indices, y)
    
    # Task 5: DFT and IDFT
    @staticmethod
    def compute_dft(signal):
        N = len(signal.samples)
        X_real = []
        X_imag = []
        for k in range(N):
            real_part = 0
            imag_part = 0
            for n in range(N):
                angle = -2 * math.pi * k * n / N
                real_part += signal.samples[n] * math.cos(angle)
                imag_part += signal.samples[n] * math.sin(angle)
            X_real.append(real_part)
            X_imag.append(imag_part)
        return X_real, X_imag

    @staticmethod
    def compute_idft(X_real, X_imag):
        N = len(X_real)
        samples = []
        for n in range(N):
            sample = 0
            for k in range(N):
                angle = 2 * math.pi * k * n / N
                sample += X_real[k] * math.cos(angle) - X_imag[k] * math.sin(angle)
            samples.append(sample / N)
        return samples

    # -----------------------------
    # Task A: Correlation
    # -----------------------------
    @staticmethod
    def correlate(signal1, signal2):
        """
        Compute the normalized cross-correlation of two signals with windowed normalization.
        r_xy(m) = sum x(n)*y(n - m) / sqrt( sum x(n)^2 * sum y(n -m)^2 )
        where the sums are over overlapping samples for each m.
        """
        x = signal1.samples
        y = signal2.samples
        len_x = len(x)
        len_y = len(y)
        corr_len = len_x + len_y -1
        r_xy = [0] * corr_len

        # Indices for correlation from -(len_y-1) to +(len_x-1)
        start_index = -(len_y -1)

        for m in range(corr_len):
            shift = m - (len_y -1)
            sum_xy = 0
            sum_x2 = 0
            sum_y2 = 0
            for n in range(len_x):
                y_idx = n - shift
                if 0 <= y_idx < len_y:
                    sum_xy +=x[n]*y[y_idx]
                    sum_x2 +=x[n]**2
                    sum_y2 +=y[y_idx]**2
            if sum_x2 >0 and sum_y2 >0:
                r_xy[m] = sum_xy / math.sqrt(sum_x2 * sum_y2)
            else:
                r_xy[m] = 0  # Handle cases with no overlap or zero norm

        indices = [start_index + i for i in range(corr_len)]
        return Signal(indices, r_xy)

    @staticmethod
    def time_delay_from_correlation(corr_signal):
        samples = corr_signal.samples
        indices = corr_signal.indices
        max_val = max(samples, key=abs)
        max_idx = samples.index(max_val)
        delay = indices[max_idx]  # The index (lag) where max correlation occurs
        return delay

    @staticmethod
    def classify_signal_by_correlation(signal_to_classify, classA_template, classB_template):
        corrA = SignalOperations.correlate(signal_to_classify, classA_template)
        corrB = SignalOperations.correlate(signal_to_classify, classB_template)

        maxA = max(corrA.samples, key=abs)
        maxB = max(corrB.samples, key=abs)

        # Example rule: if abs(maxA) > abs(maxB), classify as A
        if abs(maxA) > abs(maxB):
            return "Class A"
        else:
            return "Class B"

    # -----------------------------
    # Task B: Filtering
    # -----------------------------
    @staticmethod
    def design_fir_filter(filter_specs):
        # Extract filter specifications
        ftype = filter_specs.get('FilterType', 'Low pass').lower()
        fs = float(filter_specs.get('FS', 8000))
        A_s = float(filter_specs.get('StopBandAttenuation', 50))
        # For low/high pass => 'FC' in specs
        # For band pass/stop => 'F1' and 'F2'

        # Define window methods in order with their Transition Width constants and StopBandAttenuation
        window_methods = [
            ('Rectangular', 0.9, 21, lambda n, N: 1.0),  # #CHANGE: Added transition_width_const and adjusted window_func to use N
            ('Hanning', 3.1, 44, lambda n, N: 0.5 + 0.5 * math.cos(2 * math.pi * n / N)),  # #CHANGE
            ('Hamming', 3.3, 53, lambda n, N: 0.54 + 0.46 * math.cos(2 * math.pi * n / N)),  # #CHANGE
            ('Blackman', 5.5, 74, lambda n, N: 0.42 + 0.5 * math.cos(2 * math.pi * n / N-1) + 0.08 * math.cos(4 * math.pi * n / N-1))  # #CHANGE
        ]

        # Select window method based on StopBandAttenuation
        selected_window = None
        for name, transition_width_const, attenuation, window_func in window_methods:  # #CHANGE: Unpacked transition_width_const
            if attenuation >= A_s:
                selected_window = (name, transition_width_const, attenuation, window_func)
                break
        if not selected_window:
            # If none satisfy, choose the one with the highest attenuation
            selected_window = window_methods[-1]  # Blackman
            name, transition_width_const, attenuation, window_func = selected_window
            print(f"No window satisfies the StopBandAttenuation >= {A_s}. Selected {name} window with attenuation {attenuation} dB.")  # #CHANGE
        else:
            name, transition_width_const, attenuation, window_func = selected_window  # #CHANGE

        # Now, proceed with filter design using the selected window
        # 1) Compute normalized transition width
        trans = float(filter_specs.get('TransitionBand', 500))  # Transition width in Hz
        delta_f_normalized = trans / fs  # Normalized transition width
        # Calculate filter order (N) based on window method's transition width formula
        # From the table: Transition Width = transition_width_const / N => N = transition_width_const / delta_f_normalized
        N_approx = transition_width_const / delta_f_normalized  # #CHANGE
        N = int(math.ceil(N_approx))  # #CHANGE
        if N % 2 == 0:
            N += 1  # Ensure it's odd
        # Enforce a minimum order if necessary
        N = max(N, 3)  # #CHANGE
        print(f"Selected Window: {name}, Transition Width Const: {transition_width_const}, Transition Width Normalized: {delta_f_normalized}, Calculated N: {N}")  # #CHANGE

        # 2) Adjust cutoff frequencies to center the transition
        if ftype == 'low pass':
            # For Low-Pass: f_c' = f_p + Delta_f / 2
            f_p = float(filter_specs.get('FC', 1500))
            f_c_prime = f_p + (trans / 2)  # #CHANGE
            fc_prime_normalized = f_c_prime / fs  # Normalized
            wc = 2 * math.pi * fc_prime_normalized  # #CHANGE
        elif ftype == 'high pass':
            # For High-Pass: f_c' = f_p - Delta_f / 2
            f_p = float(filter_specs.get('FC', 1500))
            f_c_prime = f_p - (trans / 2)  # #CHANGE
            fc_prime_normalized = f_c_prime / fs  # Normalized
            wc = 2 * math.pi * fc_prime_normalized  # #CHANGE
        elif ftype == 'band pass':
            # For Band-Pass: f1' = f1 - Delta_f / 2, f2' = f2 + Delta_f / 2
            f1 = float(filter_specs.get('F1', 150))
            f2 = float(filter_specs.get('F2', 250))
            f1_prime = f1 - (trans / 2)  # #CHANGE
            f2_prime = f2 + (trans / 2)  # #CHANGE
            w1 = 2 * math.pi * (f1_prime / fs)  # #CHANGE
            w2 = 2 * math.pi * (f2_prime / fs)  # #CHANGE
        elif ftype == 'band stop':
            # For Band-Stop: f1' = f1 + Delta_f / 2, f2' = f2 - Delta_f / 2
            f1 = float(filter_specs.get('F1', 150))
            f2 = float(filter_specs.get('F2', 250))
            f1_prime = f1 + (trans / 2)  # #CHANGE
            f2_prime = f2 - (trans / 2)  # #CHANGE
            w1 = 2 * math.pi * (f1_prime / fs)  # #CHANGE
            w2 = 2 * math.pi * (f2_prime / fs)  # #CHANGE
        else:
            raise ValueError("Unsupported Filter Type")

        # 3) Compute ideal filter hd(n):
        M = (N - 1) // 2
        hd = []
        for n in range(-M, M + 1):
            if ftype == 'low pass':
                if n == 0:
                    val = wc / math.pi
                else:
                    val = math.sin(wc * n) / (math.pi * n)
                hd.append(val)
            elif ftype == 'high pass':
                if n == 0:
                    val = 1 - wc / math.pi
                else:
                    val = -math.sin(wc * n) / (math.pi * n)
                hd.append(val)
            elif ftype == 'band pass':
                if n == 0:
                    val = (w2 - w1) / math.pi
                else:
                    val = (math.sin(w2 * n) - math.sin(w1 * n)) / (math.pi * n)
                hd.append(val)
            elif ftype == 'band stop':
                if n == 0:
                    val = 1 - (w2 - w1) / math.pi
                else:
                    val = - (math.sin(w2 * n) - math.sin(w1 * n)) / (math.pi * n)
                hd.append(val)
            else:
                raise ValueError("Unknown Filter Type")

        # 4) Compute window w(n) using the selected window function
        w = []
        for i, n in enumerate(range(-M, M + 1)):
            wval = window_func(n, N)  # #CHANGE: pass N, not M
            w.append(wval)

        # 5) Multiply hd(n) * w(n) => h(n)
        h = [hd_i * w_i for hd_i, w_i in zip(hd, w)]

        # Construct the final filter as a Signal. Indices from -M..M
        indices = list(range(-M, M + 1))
        return Signal(indices, h)

    @staticmethod
    def filter_signal(signal, fir_filter, method='time'):
        """
        Filter the input 'signal' with 'fir_filter' in either 'time' or 'freq'.
        If 'time', do standard convolution.
        If 'freq', do frequency multiplication => IDFT( DFT(x) * DFT(h) ).
        """
        if method == 'time':
            return SignalOperations.convolve(signal, fir_filter)
        else:
            # freq-domain method
            N = len(signal.samples) + len(fir_filter.samples) - 1
            # zero-pad to length N
            x_padded = Signal(signal.indices, signal.samples + [0]*(N - len(signal.samples)))
            h_padded = Signal(fir_filter.indices, fir_filter.samples + [0]*(N - len(fir_filter.samples)))
            X_real, X_imag = SignalOperations.compute_dft(x_padded)
            H_real, H_imag = SignalOperations.compute_dft(h_padded)

            # Multiply in frequency
            Y_real = []
            Y_imag = []
            for rX, iX, rH, iH in zip(X_real, X_imag, H_real, H_imag):
                # (rX + j iX)(rH + j iH) = (rX*rH - iX*iH) + j(rX*iH + iX*rH)
                yr = (rX * rH) - (iX * iH)
                yi = (rX * iH) + (iX * rH)
                Y_real.append(yr)
                Y_imag.append(yi)

            # IDFT
            y_samples = SignalOperations.compute_idft(Y_real, Y_imag)
            # Indices => start_index = sum of start indices
            start_index = signal.indices[0] + fir_filter.indices[0]
            indices = list(range(start_index, start_index + N))
            return Signal(indices, y_samples)

# Signal Processing App with GUI
class SignalProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Signal Processing App")
        self.signals = {}  # Dictionary to store loaded/generated signals

        self.create_widgets()

    def create_widgets(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill='both')

        task1_frame = ttk.Frame(notebook)
        task2_frame = ttk.Frame(notebook)
        task3_frame = ttk.Frame(notebook)
        task4_frame = ttk.Frame(notebook)
        task5_frame = ttk.Frame(notebook)
        taskA_frame = ttk.Frame(notebook)
        taskB_frame = ttk.Frame(notebook)
        bonus_frame = ttk.Frame(notebook)

        notebook.add(task1_frame, text='Task 1 - Signal Operations')
        notebook.add(task2_frame, text='Task 2 - Signal Generation')
        notebook.add(task3_frame, text='Task 3 - Quantization')
        notebook.add(task4_frame, text='Task 4 - Advanced Operations')
        notebook.add(task5_frame, text='Task 5 - Fourier Transform')
        # New Tabs
        notebook.add(taskA_frame, text='Task A - Correlation')
        notebook.add(taskB_frame, text='Task B - Filtering')
        # New Bonus Task Tab
        notebook.add(bonus_frame, text='Bonus Task - Detection in Noise')

        self.create_task1_widgets(task1_frame)
        self.create_task2_widgets(task2_frame)
        self.create_task3_widgets(task3_frame)
        self.create_task4_widgets(task4_frame)
        self.create_task5_widgets(task5_frame)
        self.create_taskA_widgets(taskA_frame)
        self.create_taskB_widgets(taskB_frame)
        self.create_bonus_widgets(bonus_frame)


    # Task 1 Widgets
    def create_task1_widgets(self, frame):
        # Frame for signal operations
        operations_frame = ttk.LabelFrame(frame, text="Signal Operations")
        operations_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Load Signal Buttons
        load_btn1 = ttk.Button(operations_frame, text="Load Signal 1", command=lambda: self.load_signal(1))
        load_btn1.grid(row=0, column=0, padx=5, pady=5)

        load_btn2 = ttk.Button(operations_frame, text="Load Signal 2", command=lambda: self.load_signal(2))
        load_btn2.grid(row=0, column=1, padx=5, pady=5)

        # Operation Buttons
        add_btn = ttk.Button(operations_frame, text="Add Signals", command=self.add_signals)
        add_btn.grid(row=1, column=0, padx=5, pady=5)

        subtract_btn = ttk.Button(operations_frame, text="Subtract Signals", command=self.subtract_signals)
        subtract_btn.grid(row=1, column=1, padx=5, pady=5)

        multiply_btn = ttk.Button(operations_frame, text="Multiply Signal 1 by Constant", command=self.multiply_signal)
        multiply_btn.grid(row=2, column=0, padx=5, pady=5)

        shift_btn = ttk.Button(operations_frame, text="Shift Signal 1", command=self.shift_signal)
        shift_btn.grid(row=2, column=1, padx=5, pady=5)

        fold_btn = ttk.Button(operations_frame, text="Fold Signal 1", command=self.fold_signal)
        fold_btn.grid(row=3, column=0, padx=5, pady=5)

        # Visualization Options
        viz_frame = ttk.LabelFrame(frame, text="Visualization Options")
        viz_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.representation = tk.StringVar(value="Discrete")
        cont_rb = ttk.Radiobutton(viz_frame, text="Continuous", variable=self.representation, value="Continuous")
        cont_rb.grid(row=0, column=0, padx=5, pady=5)
        disc_rb = ttk.Radiobutton(viz_frame, text="Discrete", variable=self.representation, value="Discrete")
        disc_rb.grid(row=0, column=1, padx=5, pady=5)

        plot_btn = ttk.Button(viz_frame, text="Plot Signals", command=self.plot_signals)
        plot_btn.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        # Frame for Matplotlib plot
        plot_frame = ttk.Frame(frame)
        plot_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10)

        self.task1_figure, self.task1_ax = plt.subplots(figsize=(8,6))
        self.task1_canvas = FigureCanvasTkAgg(self.task1_figure, master=plot_frame)
        self.task1_canvas.draw()
        self.task1_canvas.get_tk_widget().pack()

    # Task 2 Widgets
    def create_task2_widgets(self, frame):
        # Input Frame
        input_frame = ttk.LabelFrame(frame, text="Signal Parameters")
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Wave Type Selection
        self.wave_type = tk.StringVar(value='sine')
        wave_type_frame = ttk.Frame(input_frame)
        wave_type_frame.grid(row=0, column=0, columnspan=2, pady=5)
        ttk.Label(wave_type_frame, text="Select Wave Type:").grid(row=0, column=0, padx=5)
        sine_rb = ttk.Radiobutton(wave_type_frame, text="Sine", variable=self.wave_type, value='sine')
        sine_rb.grid(row=0, column=1, padx=5)
        cosine_rb = ttk.Radiobutton(wave_type_frame, text="Cosine", variable=self.wave_type, value='cosine')
        cosine_rb.grid(row=0, column=2, padx=5)

        ttk.Label(input_frame, text="Amplitude (A):").grid(row=1, column=0, padx=5, pady=5)
        self.amplitude_entry = ttk.Entry(input_frame)
        self.amplitude_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Phase Shift (θ in radians):").grid(row=2, column=0, padx=5, pady=5)
        self.phase_entry = ttk.Entry(input_frame)
        self.phase_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Analog Frequency (Hz):").grid(row=3, column=0, padx=5, pady=5)
        self.freq_entry = ttk.Entry(input_frame)
        self.freq_entry.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Sampling Frequency (Hz):").grid(row=4, column=0, padx=5, pady=5)
        self.sampling_freq_entry = ttk.Entry(input_frame)
        self.sampling_freq_entry.grid(row=4, column=1, padx=5, pady=5)

        generate_btn = ttk.Button(input_frame, text="Generate Signal", command=self.generate_task2_signal)
        generate_btn.grid(row=5, column=0, columnspan=2, pady=10)

        # Visualization Options
        viz_frame = ttk.LabelFrame(frame, text="Visualization Options")
        viz_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.task2_representation = tk.StringVar(value="Discrete")
        cont_rb = ttk.Radiobutton(viz_frame, text="Continuous", variable=self.task2_representation, value="Continuous")
        cont_rb.grid(row=0, column=0, padx=5, pady=5)
        disc_rb = ttk.Radiobutton(viz_frame, text="Discrete", variable=self.task2_representation, value="Discrete")
        disc_rb.grid(row=0, column=1, padx=5, pady=5)

        # Plot Frame
        plot_frame = ttk.Frame(frame)
        plot_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10)

        self.task2_figure, self.task2_ax = plt.subplots(figsize=(8,6))
        self.task2_canvas = FigureCanvasTkAgg(self.task2_figure, master=plot_frame)
        self.task2_canvas.draw()
        self.task2_canvas.get_tk_widget().pack()

    def generate_task2_signal(self):
        amplitude_str = self.amplitude_entry.get()
        phase_str = self.phase_entry.get()
        freq_str = self.freq_entry.get()
        sampling_freq_str = self.sampling_freq_entry.get()
        wave_type = self.wave_type.get()  # Get the selected wave type

        try:
            amplitude = float(amplitude_str)
            phase = float(phase_str)
            freq = float(freq_str)
            sampling_freq = float(sampling_freq_str)

            if sampling_freq < 2 * freq:
                messagebox.showerror("Error", "Sampling frequency must be at least twice the analog frequency (Nyquist rate).")
                return

            # Generate signal
            if wave_type == 'sine':
                generated_signal = Signal.generate_sine(amplitude, phase, freq, sampling_freq)
            else:
                generated_signal = Signal.generate_cosine(amplitude, phase, freq, sampling_freq)

            # Plot signal
            self.task2_ax.clear()
            representation = self.task2_representation.get()
            time_values = generated_signal.time_values
            samples = generated_signal.samples

            if representation == "Continuous":
                self.task2_ax.plot(time_values, samples, label=f'{wave_type.capitalize()} Signal')
            else:
                try:
                    self.task2_ax.stem(time_values, samples, label=f'{wave_type.capitalize()} Signal', use_line_collection=True)
                except TypeError:
                    self.task2_ax.stem(time_values, samples, label=f'{wave_type.capitalize()} Signal')

            self.task2_ax.set_xlabel('Time (s)')
            self.task2_ax.set_ylabel('Amplitude')
            self.task2_ax.set_title(f"{wave_type.capitalize()} Signal")
            self.task2_ax.legend()
            self.task2_ax.grid(True)
            self.task2_canvas.draw()

            # Store the generated signal
            name = f"Generated_{wave_type.capitalize()}_{len(self.signals)+1}"
            self.signals[name] = generated_signal
            messagebox.showinfo("Success", f"{wave_type.capitalize()} signal generated successfully as '{name}'.")

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values.")

    # Task 3 Widgets
    def create_task3_widgets(self, frame):
        # Input Frame
        input_frame = ttk.LabelFrame(frame, text="Quantization Parameters")
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        load_signal_btn = ttk.Button(input_frame, text="Select Signal", command=self.load_quantization_signal)
        load_signal_btn.grid(row=0, column=0, padx=5, pady=5)

        ttk.Label(input_frame, text="Number of Levels:").grid(row=1, column=0, padx=5, pady=5)
        self.levels_entry = ttk.Entry(input_frame)
        self.levels_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Number of Bits:").grid(row=2, column=0, padx=5, pady=5)
        self.bits_entry = ttk.Entry(input_frame)
        self.bits_entry.grid(row=2, column=1, padx=5, pady=5)

        quantize_btn = ttk.Button(input_frame, text="Quantize Signal", command=self.quantize_signal)
        quantize_btn.grid(row=3, column=0, columnspan=2, pady=10)

        # Plot Frame
        plot_frame = ttk.Frame(frame)
        plot_frame.grid(row=0, column=1, padx=10, pady=10)

        self.task3_figure, self.task3_axes = plt.subplots(2, 1, figsize=(8,8))
        self.task3_canvas = FigureCanvasTkAgg(self.task3_figure, master=plot_frame)
        self.task3_canvas.draw()
        self.task3_canvas.get_tk_widget().pack()

    def load_quantization_signal(self):
        # Allow user to select from existing signals
        signal_names = list(self.signals.keys())
        if signal_names:
            def select_signal():
                selected = signal_listbox.curselection()
                if selected:
                    index = selected[0]
                    signal_name = signal_names[index]
                    self.quantization_signal = self.signals[signal_name]
                    messagebox.showinfo("Success", f"Signal '{signal_name}' selected for quantization.")
                    select_window.destroy()
                else:
                    messagebox.showwarning("Warning", "Please select a signal.")
            
            select_window = tk.Toplevel(self.root)
            select_window.title("Select Signal for Quantization")
            ttk.Label(select_window, text="Select Signal:").pack(padx=5, pady=5)
            signal_listbox = tk.Listbox(select_window)
            signal_listbox.pack(padx=5, pady=5)
            for name in signal_names:
                signal_listbox.insert(tk.END, name)
            ttk.Button(select_window, text="Select", command=select_signal).pack(pady=5)

        else:
            messagebox.showwarning("Warning", "No signals available. Please generate or load a signal first.")

    def quantize_signal(self):
        if not hasattr(self, 'quantization_signal'):
            messagebox.showwarning("Warning", "Please select a signal for quantization.")
            return

        levels_str = self.levels_entry.get()
        bits_str = self.bits_entry.get()

        try:
            if levels_str:
                levels = int(levels_str)
            elif bits_str:
                bits = int(bits_str)
                levels = 2 ** bits
            else:
                messagebox.showerror("Error", "Please enter number of levels or number of bits.")
                return

            # Perform quantization
            signal = self.quantization_signal
            samples = np.array(signal.samples)
            min_val = np.min(samples)
            max_val = np.max(samples)

            # Compute quantization step size
            q_step = (max_val - min_val) / levels

            # Quantize samples
            quantized_samples = np.floor((samples - min_val) / q_step) * q_step + min_val + q_step / 2

            # Quantization error
            error = samples - quantized_samples

            # Encode quantized samples
            encoded_samples = ((quantized_samples - min_val) / q_step).astype(int)

            # Plot original and quantized signal
            self.task3_axes[0].clear()
            self.task3_axes[0].plot(signal.indices, samples, label='Original Signal')
            self.task3_axes[0].step(signal.indices, quantized_samples, where='mid', label='Quantized Signal')
            self.task3_axes[0].set_xlabel('Sample Index')
            self.task3_axes[0].set_ylabel('Amplitude')
            self.task3_axes[0].set_title('Original and Quantized Signal')
            self.task3_axes[0].legend()
            self.task3_axes[0].grid(True)

            # Plot quantization error
            self.task3_axes[1].clear()
            try:
                self.task3_axes[1].stem(signal.indices, error, label='Quantization Error', use_line_collection=True)
            except TypeError:
                self.task3_axes[1].stem(signal.indices, error, label='Quantization Error')
            self.task3_axes[1].set_xlabel('Sample Index')
            self.task3_axes[1].set_ylabel('Error')
            self.task3_axes[1].set_title('Quantization Error')
            self.task3_axes[1].legend()
            self.task3_axes[1].grid(True)

            self.task3_canvas.draw()

            # Display encoded signal
            encoded_str = ' '.join(map(str, encoded_samples))
            messagebox.showinfo("Encoded Signal", f"Encoded Levels:\n{encoded_str}")

        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer values for levels or bits")
    
    # Task 4 Widgets
    def create_task4_widgets(self, frame):
        # Frame for operation selection
        operations_frame = ttk.LabelFrame(frame, text="Select Operation")
        operations_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Operation Selection
        self.task4_operation = tk.StringVar(value='moving_average')
        ttk.Radiobutton(operations_frame, text="Moving Average", variable=self.task4_operation, value='moving_average').grid(row=0, column=0, padx=5, pady=5)
        ttk.Radiobutton(operations_frame, text="First Derivative", variable=self.task4_operation, value='first_derivative').grid(row=0, column=1, padx=5, pady=5)
        ttk.Radiobutton(operations_frame, text="Second Derivative", variable=self.task4_operation, value='second_derivative').grid(row=0, column=2, padx=5, pady=5)
        ttk.Radiobutton(operations_frame, text="Convolution", variable=self.task4_operation, value='convolution').grid(row=0, column=3, padx=5, pady=5)

        # Load Signal Buttons
        ttk.Button(operations_frame, text="Select Signal", command=self.load_task4_signal).grid(row=1, column=0, padx=5, pady=5)

        self.window_size_entry = None  # Placeholder for window size entry

        # Additional parameter input
        def update_parameters(*args):
            # Clear previous parameter inputs
            for widget in operations_frame.grid_slaves(row=2):
                widget.destroy()
            if self.task4_operation.get() == 'moving_average':
                ttk.Label(operations_frame, text="Window Size:").grid(row=2, column=0, padx=5, pady=5)
                self.window_size_entry = ttk.Entry(operations_frame)
                self.window_size_entry.grid(row=2, column=1, padx=5, pady=5)
            elif self.task4_operation.get() == 'convolution':
                ttk.Button(operations_frame, text="Select Second Signal", command=self.load_task4_signal2).grid(row=2, column=0, padx=5, pady=5)

        self.task4_operation.trace('w', update_parameters)

        # Compute Button
        ttk.Button(operations_frame, text="Compute", command=self.compute_task4_operation).grid(row=3, column=0, columnspan=4, pady=10)

        # Plot Frame
        plot_frame = ttk.Frame(frame)
        plot_frame.grid(row=1, column=0, padx=10, pady=10)

        self.task4_figure, self.task4_ax = plt.subplots(figsize=(8,6))
        self.task4_canvas = FigureCanvasTkAgg(self.task4_figure, master=plot_frame)
        self.task4_canvas.draw()
        self.task4_canvas.get_tk_widget().pack()

    def load_task4_signal(self):
        # Allow user to select from existing signals
        signal_names = list(self.signals.keys())
        if signal_names:
            def select_signal():
                selected = signal_listbox.curselection()
                if selected:
                    index = selected[0]
                    signal_name = signal_names[index]
                    self.task4_signal = self.signals[signal_name]
                    messagebox.showinfo("Success", f"Signal '{signal_name}' selected.")
                    select_window.destroy()
                else:
                    messagebox.showwarning("Warning", "Please select a signal.")
            
            select_window = tk.Toplevel(self.root)
            select_window.title("Select Signal")
            ttk.Label(select_window, text="Select Signal:").pack(padx=5, pady=5)
            signal_listbox = tk.Listbox(select_window)
            signal_listbox.pack(padx=5, pady=5)
            for name in signal_names:
                signal_listbox.insert(tk.END, name)
            ttk.Button(select_window, text="Select", command=select_signal).pack(pady=5)

        else:
            messagebox.showwarning("Warning", "No signals available. Please generate or load a signal first.")

    def load_task4_signal2(self):
        # Allow user to select the second signal for convolution
        signal_names = list(self.signals.keys())
        if signal_names:
            def select_signal():
                selected = signal_listbox.curselection()
                if selected:
                    index = selected[0]
                    signal_name = signal_names[index]
                    self.task4_signal2 = self.signals[signal_name]
                    messagebox.showinfo("Success", f"Signal '{signal_name}' selected as second signal.")
                    select_window.destroy()
                else:
                    messagebox.showwarning("Warning", "Please select a signal.")
            
            select_window = tk.Toplevel(self.root)
            select_window.title("Select Second Signal")
            ttk.Label(select_window, text="Select Second Signal:").pack(padx=5, pady=5)
            signal_listbox = tk.Listbox(select_window)
            signal_listbox.pack(padx=5, pady=5)
            for name in signal_names:
                signal_listbox.insert(tk.END, name)
            ttk.Button(select_window, text="Select", command=select_signal).pack(pady=5)

        else:
            messagebox.showwarning("Warning", "No signals available. Please generate or load a signal first.")

    def compute_task4_operation(self):
        operation = self.task4_operation.get()
        if not hasattr(self, 'task4_signal'):
            messagebox.showwarning("Warning", "Please select a signal.")
            return

        signal = self.task4_signal

        if operation == 'moving_average':
            if not self.window_size_entry:
                messagebox.showerror("Error", "Window size entry not found.")
                return
            window_size_str = self.window_size_entry.get()
            try:
                window_size = int(window_size_str)
                if window_size < 1:
                    messagebox.showerror("Error", "Window size must be at least 1.")
                    return
                if window_size > len(signal.samples):
                    messagebox.showerror("Error", "Window size cannot be larger than the number of samples.")
                    return
                result_signal = SignalOperations.moving_average(signal, window_size)
                name = f"Moving_Average_{window_size}_{len(self.signals)+1}"
                self.signals[name] = result_signal
                messagebox.showinfo("Success", f"Moving average computed successfully as '{name}'.")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid integer for window size.")
                return
        elif operation == 'first_derivative':
            result_signal = SignalOperations.first_derivative(signal)
            name = f"First_Derivative_{len(self.signals)+1}"
            self.signals[name] = result_signal
            messagebox.showinfo("Success", f"First derivative computed successfully as '{name}'.")
        elif operation == 'second_derivative':
            result_signal = SignalOperations.second_derivative(signal)
            name = f"Second_Derivative_{len(self.signals)+1}"
            self.signals[name] = result_signal
            messagebox.showinfo("Success", f"Second derivative computed successfully as '{name}'.")
        elif operation == 'convolution':
            if not hasattr(self, 'task4_signal2'):
                messagebox.showwarning("Warning", "Please select the second signal for convolution.")
                return
            signal2 = self.task4_signal2
            result_signal = SignalOperations.convolve(signal, signal2)
            name = f"Convolution_{len(self.signals)+1}"
            self.signals[name] = result_signal
            messagebox.showinfo("Success", f"Convolution computed successfully as '{name}'.")
        else:
            messagebox.showerror("Error", "Unknown operation.")
            return

        # Plot result
        self.task4_ax.clear()
        indices = result_signal.indices
        samples = result_signal.samples
        self.task4_ax.plot(indices, samples, label=name)
        self.task4_ax.set_xlabel('Sample Index')
        self.task4_ax.set_ylabel('Amplitude')
        self.task4_ax.set_title(f"Result of {operation.replace('_', ' ').capitalize()}")
        self.task4_ax.legend()
        self.task4_ax.grid(True)
        self.task4_canvas.draw()

    # Task 5 Widgets
    def create_task5_widgets(self, frame):
        # Input Frame
        input_frame = ttk.LabelFrame(frame, text="Fourier Transform")
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Load Signal Button
        ttk.Button(input_frame, text="Select Signal", command=self.load_task5_signal).grid(row=0, column=0, padx=5, pady=5)

        # Sampling Frequency Entry
        ttk.Label(input_frame, text="Sampling Frequency (Hz):").grid(row=1, column=0, padx=5, pady=5)
        self.sampling_freq_entry_task5 = ttk.Entry(input_frame)
        self.sampling_freq_entry_task5.grid(row=1, column=1, padx=5, pady=5)

        # Compute DFT Button
        ttk.Button(input_frame, text="Compute DFT", command=self.compute_dft).grid(row=2, column=0, columnspan=2, pady=10)

        # Compute IDFT Button
        ttk.Button(input_frame, text="Compute IDFT", command=self.compute_idft).grid(row=3, column=0, columnspan=2, pady=10)

        # Plot Frame for Amplitude and Phase
        plot_frame = ttk.Frame(frame)
        plot_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10)

        self.task5_figure, (self.amplitude_ax, self.phase_ax) = plt.subplots(2, 1, figsize=(8, 8))
        self.task5_canvas = FigureCanvasTkAgg(self.task5_figure, master=plot_frame)
        self.task5_canvas.draw()
        self.task5_canvas.get_tk_widget().pack()

        # Plot Frame for Reconstructed Signal
        plot_frame_recon = ttk.Frame(frame)
        plot_frame_recon.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        self.task5_recon_figure, self.recon_ax = plt.subplots(figsize=(8, 4))
        self.task5_recon_canvas = FigureCanvasTkAgg(self.task5_recon_figure, master=plot_frame_recon)
        self.task5_recon_canvas.draw()
        self.task5_recon_canvas.get_tk_widget().pack()

    def load_task5_signal(self):
        # Allow user to select from existing signals
        signal_names = list(self.signals.keys())
        if signal_names:
            def select_signal():
                selected = signal_listbox.curselection()
                if selected:
                    index = selected[0]
                    signal_name = signal_names[index]
                    self.task5_signal = self.signals[signal_name]
                    messagebox.showinfo("Success", f"Signal '{signal_name}' selected for DFT.")
                    select_window.destroy()
                else:
                    messagebox.showwarning("Warning", "Please select a signal.")
            
            select_window = tk.Toplevel(self.root)
            select_window.title("Select Signal for DFT")
            ttk.Label(select_window, text="Select Signal:").pack(padx=5, pady=5)
            signal_listbox = tk.Listbox(select_window)
            signal_listbox.pack(padx=5, pady=5)
            for name in signal_names:
                signal_listbox.insert(tk.END, name)
            ttk.Button(select_window, text="Select", command=select_signal).pack(pady=5)
        else:
            messagebox.showwarning("Warning", "No signals available. Please generate or load a signal first.")

    def compute_dft(self):
        if not hasattr(self, 'task5_signal'):
            messagebox.showwarning("Warning", "Please select a signal for DFT.")
            return

        sampling_freq_str = self.sampling_freq_entry_task5.get()
        try:
            sampling_freq = float(sampling_freq_str)
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid sampling frequency.")
            return

        signal = self.task5_signal
        N = len(signal.samples)
        X_real, X_imag = SignalOperations.compute_dft(signal)

        # Compute amplitude and phase spectra
        amplitudes = [math.sqrt(r**2 + im**2) for r, im in zip(X_real, X_imag)]
        phases = [math.atan2(im, r) for r, im in zip(X_real, X_imag)]

        # Frequency bins
        freqs = [sampling_freq * k / N for k in range(N)]

        # Plot amplitude spectrum
        self.amplitude_ax.clear()
        try:
            self.amplitude_ax.stem(freqs, amplitudes, use_line_collection=True)
        except TypeError:
            self.amplitude_ax.stem(freqs, amplitudes)
        self.amplitude_ax.set_xlabel('Frequency (Hz)')
        self.amplitude_ax.set_ylabel('Amplitude')
        self.amplitude_ax.set_title('Amplitude Spectrum')
        self.amplitude_ax.grid(True)

        # Plot phase spectrum
        self.phase_ax.clear()
        try:
            self.phase_ax.stem(freqs, phases, use_line_collection=True)
        except TypeError:
            self.phase_ax.stem(freqs, phases)
        self.phase_ax.set_xlabel('Frequency (Hz)')
        self.phase_ax.set_ylabel('Phase (Radians)')
        self.phase_ax.set_title('Phase Spectrum')
        self.phase_ax.grid(True)

        self.task5_canvas.draw()

        # Store DFT components for IDFT
        self.X_real = X_real
        self.X_imag = X_imag
        self.N = N
        self.sampling_freq = sampling_freq


    def compute_idft(self):
        if not hasattr(self, 'X_real') or not hasattr(self, 'X_imag'):
            messagebox.showwarning("Warning", "Please compute DFT first.")
            return

        reconstructed_samples = SignalOperations.compute_idft(self.X_real, self.X_imag)

        # Plot reconstructed signal
        self.recon_ax.clear()
        time_axis = [n / self.sampling_freq for n in range(self.N)]
        self.recon_ax.plot(time_axis, reconstructed_samples, label='Reconstructed Signal')
        self.recon_ax.set_xlabel('Time (s)')
        self.recon_ax.set_ylabel('Amplitude')
        self.recon_ax.set_title('Reconstructed Signal via IDFT')
        self.recon_ax.legend()
        self.recon_ax.grid(True)
        self.task5_recon_canvas.draw()

        # Store reconstructed signal
        name = f"Reconstructed_Signal_{len(self.signals)+1}"
        indices = list(range(self.N))
        self.signals[name] = Signal(indices, reconstructed_samples)
        messagebox.showinfo("Success", f"Signal reconstructed successfully as '{name}'.")
    

    # =======================
    # Task A: Correlation
    # =======================
    def create_taskA_widgets(self, frame):
        operations_frame = ttk.LabelFrame(frame, text="Correlation Tasks")
        operations_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Buttons to load signals
        ttk.Button(operations_frame, text="Load Correlation Signal 1", command=lambda: self.load_signal('Corr1')).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(operations_frame, text="Load Correlation Signal 2", command=lambda: self.load_signal('Corr2')).grid(row=0, column=1, padx=5, pady=5)

        corr_btn = ttk.Button(operations_frame, text="Compute Correlation", command=self.compute_correlation)
        corr_btn.grid(row=1, column=0, padx=5, pady=5)

        delay_btn = ttk.Button(operations_frame, text="Estimate Time Delay", command=self.estimate_time_delay)
        delay_btn.grid(row=1, column=1, padx=5, pady=5)

        classify_btn = ttk.Button(operations_frame, text="Classify (A or B)", command=self.classify_signal)
        classify_btn.grid(row=2, column=0, columnspan=2, pady=5)

        # Plot Frame
        plot_frame = ttk.Frame(frame)
        plot_frame.grid(row=0, column=1, padx=10, pady=10)

        self.taskA_figure, self.taskA_ax = plt.subplots(figsize=(8,6))
        self.taskA_canvas = FigureCanvasTkAgg(self.taskA_figure, master=plot_frame)
        self.taskA_canvas.draw()
        self.taskA_canvas.get_tk_widget().pack()

    def compute_correlation(self):
        if 'Corr1' not in self.signals or 'Corr2' not in self.signals:
            messagebox.showwarning("Warning", "Please load both correlation signals.")
            return
        signal1 = self.signals['Corr1']
        signal2 = self.signals['Corr2']

        corr_signal = SignalOperations.correlate(signal1, signal2)
        name = f"Correlation_{len(self.signals)+1}"
        self.signals[name] = corr_signal
        messagebox.showinfo("Success", f"Correlation computed successfully as '{name}'.")

        # Plot
        self.taskA_ax.clear()
        self.taskA_ax.plot(corr_signal.indices, corr_signal.samples, label='Correlation')
        self.taskA_ax.set_xlabel('Lag')
        self.taskA_ax.set_ylabel('Correlation Amplitude')
        self.taskA_ax.set_title("Correlation of two signals")
        self.taskA_ax.legend()
        self.taskA_ax.grid(True)
        self.taskA_canvas.draw()

    def estimate_time_delay(self):
        # We assume we have a correlation signal saved under some name (e.g., "Correlation_#")
        # For simplicity, let's pick the last correlation we computed
        corr_keys = [k for k in self.signals.keys() if "Correlation_" in k]
        if not corr_keys:
            messagebox.showwarning("Warning", "No correlation signal found. Please compute correlation first.")
            return
        corr_signal = self.signals[corr_keys[-1]]
        delay = SignalOperations.time_delay_from_correlation(corr_signal)
        messagebox.showinfo("Time Delay", f"Estimated time delay (lag) is: {delay}")

    def classify_signal(self):
        # Example usage: we assume we have "CorrSignal" to classify with "ClassA_template" and "ClassB_template"
        if 'Corr1' not in self.signals or 'ClassA_template' not in self.signals or 'ClassB_template' not in self.signals:
            messagebox.showwarning("Warning", "Please load all required signals: Corr1, ClassA_template, ClassB_template.")
            return
        to_classify = self.signals['Corr1']
        classA_template = self.signals['ClassA_template']
        classB_template = self.signals['ClassB_template']

        c = SignalOperations.classify_signal_by_correlation(to_classify, classA_template, classB_template)
        messagebox.showinfo("Classification", f"Signal classified as: {c}")

    # =======================
    # Task B: Filtering
    # =======================
    def create_taskB_widgets(self, frame):
        operations_frame = ttk.LabelFrame(frame, text="Filtering Tasks")
        operations_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Buttons to load input signal
        ttk.Button(operations_frame, text="Load Signal for Filtering", command=lambda: self.load_signal('FilterInput')).grid(row=0, column=0, padx=5, pady=5)

        # Provide a way to load filter specs from a file
        ttk.Button(operations_frame, text="Load Filter Specs", command=self.load_filter_specs).grid(row=0, column=1, padx=5, pady=5)

        # Button to compute FIR filter
        compute_filter_btn = ttk.Button(operations_frame, text="Compute FIR Filter", command=self.compute_fir_filter)
        compute_filter_btn.grid(row=1, column=0, padx=5, pady=5)

        # Button to filter signal
        filter_signal_btn = ttk.Button(operations_frame, text="Apply Filter", command=self.apply_filter)
        filter_signal_btn.grid(row=1, column=1, padx=5, pady=5)

        # Choice of method
        self.filter_method = tk.StringVar(value="time")
        ttk.Radiobutton(operations_frame, text="Time Convolution", variable=self.filter_method, value="time").grid(row=2, column=0, padx=5, pady=5)
        ttk.Radiobutton(operations_frame, text="Freq Multiplication", variable=self.filter_method, value="freq").grid(row=2, column=1, padx=5, pady=5)

        # Plot Frame
        plot_frame = ttk.Frame(frame)
        plot_frame.grid(row=0, column=1, padx=10, pady=10)

        self.taskB_figure, self.taskB_ax = plt.subplots(figsize=(8,6))
        self.taskB_canvas = FigureCanvasTkAgg(self.taskB_figure, master=plot_frame)
        self.taskB_canvas.draw()
        self.taskB_canvas.get_tk_widget().pack()

        # Store filter specs
        self.filter_specs = {}

    def load_filter_specs(self):
        file_path = filedialog.askopenfilename(title="Select Filter Specification File", filetypes=(("Text Files", "*.txt"),))
        if file_path:
            specs = {}
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if '=' in line:
                        key, val = line.split('=')
                        key = key.strip()
                        val = val.strip()
                        specs[key] = val
            self.filter_specs = specs
            messagebox.showinfo("Success", "Filter specifications loaded.")

    def compute_fir_filter(self):
        if not self.filter_specs:
            messagebox.showwarning("Warning", "Please load filter specs first.")
            return
        fir_signal = SignalOperations.design_fir_filter(self.filter_specs)
        name = f"FIR_Filter_{len(self.signals)+1}"
        self.signals[name] = fir_signal
        messagebox.showinfo("Success", f"FIR filter computed successfully as '{name}'.")
        # Plot the filter
        self.taskB_ax.clear()
        self.taskB_ax.stem(fir_signal.indices, fir_signal.samples, label='FIR Filter Coeffs', use_line_collection=True)
        self.taskB_ax.set_xlabel('n')
        self.taskB_ax.set_ylabel('h(n)')
        self.taskB_ax.set_title("FIR Filter Coefficients")
        self.taskB_ax.legend()
        self.taskB_ax.grid(True)
        self.taskB_canvas.draw()

    def apply_filter(self):
        # We assume we have 'FilterInput' signal and a 'FIR_Filter_#' signal
        filter_keys = [k for k in self.signals.keys() if "FIR_Filter_" in k]
        if 'FilterInput' not in self.signals or not filter_keys:
            messagebox.showwarning("Warning", "Please load input signal and compute FIR filter first.")
            return
        input_signal = self.signals['FilterInput']
        fir_filter = self.signals[filter_keys[-1]]
        method = self.filter_method.get()

        filtered_signal = SignalOperations.filter_signal(input_signal, fir_filter, method=method)
        name = f"Filtered_Signal_{len(self.signals)+1}"
        self.signals[name] = filtered_signal
        messagebox.showinfo("Success", f"Signal filtered by FIR in {method} domain as '{name}'.")

        # Plot
        self.taskB_ax.clear()
        self.taskB_ax.plot(filtered_signal.indices, filtered_signal.samples, label='Filtered Signal')
        self.taskB_ax.set_xlabel('n')
        self.taskB_ax.set_ylabel('Amplitude')
        self.taskB_ax.set_title("Filtered Signal")
        self.taskB_ax.legend()
        self.taskB_ax.grid(True)
        self.taskB_canvas.draw()

    # Existing methods for Task 1
    def load_signal(self, signal_number):
        """
        Overloaded so we can pass a string like 'Corr1' or 'FilterInput' 
        or an integer 1/2 for the older code.
        """
        file_path = filedialog.askopenfilename(
            title="Select Signal File", 
            filetypes=(("Text Files", "*.txt"),)
        )
        if file_path:
            try:
                signal = Signal.from_file(file_path)
                # If signal_number is an int, do the old behavior
                if isinstance(signal_number, int):
                    self.signals[f"Signal{signal_number}"] = signal
                    messagebox.showinfo("Success", f"Signal {signal_number} loaded successfully.")
                else:
                    self.signals[signal_number] = signal
                    messagebox.showinfo("Success", f"Signal '{signal_number}' loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load signal: {e}")

    def add_signals(self):
        if "Signal1" in self.signals and "Signal2" in self.signals:
            try:
                added_signal = SignalOperations.add(self.signals["Signal1"], self.signals["Signal2"])
                self.signals["Added_Signal"] = added_signal
                messagebox.showinfo("Success", "Signals added successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Addition failed: {e}")
        else:
            messagebox.showwarning("Warning", "Please load both Signal 1 and Signal 2.")

    def subtract_signals(self):
        if "Signal1" in self.signals and "Signal2" in self.signals:
            try:
                subtracted_signal = SignalOperations.subtract(self.signals["Signal1"], self.signals["Signal2"])
                self.signals["Subtracted_Signal"] = subtracted_signal
                messagebox.showinfo("Success", "Signals subtracted successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Subtraction failed: {e}")
        else:
            messagebox.showwarning("Warning", "Please load both Signal 1 and Signal 2.")

    def multiply_signal(self):
        if "Signal1" in self.signals:
            def perform_multiplication():
                try:
                    const = float(const_entry.get())
                    multiplied_signal = SignalOperations.multiply(self.signals["Signal1"], const)
                    name = f"Multiplied_Signal_{const}"
                    self.signals[name] = multiplied_signal
                    messagebox.showinfo("Success", f"Signal 1 multiplied by {const} successfully as '{name}'.")
                    mul_window.destroy()
                except ValueError:
                    messagebox.showerror("Error", "Please enter a valid constant.")

            mul_window = tk.Toplevel(self.root)
            mul_window.title("Multiply Signal 1 by Constant")
            ttk.Label(mul_window, text="Constant:").grid(row=0, column=0, padx=5, pady=5)
            const_entry = ttk.Entry(mul_window)
            const_entry.grid(row=0, column=1, padx=5, pady=5)
            ttk.Button(mul_window, text="Multiply", command=perform_multiplication).grid(row=1, column=0, columnspan=2, pady=5)
        else:
            messagebox.showwarning("Warning", "Please load Signal 1.")

    def shift_signal(self):
        if "Signal1" in self.signals:
            def perform_shift():
                try:
                    k = int(shift_entry.get())
                    shifted_signal = SignalOperations.shift(self.signals["Signal1"], k)
                    name = f"Shifted_Signal_{k}"
                    self.signals[name] = shifted_signal
                    messagebox.showinfo("Success", f"Signal 1 shifted by {k} successfully as '{name}'.")
                    shift_window.destroy()
                except ValueError:
                    messagebox.showerror("Error", "Please enter a valid integer for shift.")

            shift_window = tk.Toplevel(self.root)
            shift_window.title("Shift Signal 1")
            ttk.Label(shift_window, text="Shift by (k):").grid(row=0, column=0, padx=5, pady=5)
            shift_entry = ttk.Entry(shift_window)
            shift_entry.grid(row=0, column=1, padx=5, pady=5)
            ttk.Button(shift_window, text="Shift", command=perform_shift).grid(row=1, column=0, columnspan=2, pady=5)
        else:
            messagebox.showwarning("Warning", "Please load Signal 1.")

    def fold_signal(self):
        if "Signal1" in self.signals:
            try:
                folded_signal = SignalOperations.fold(self.signals["Signal1"])
                name = "Folded_Signal"
                self.signals[name] = folded_signal
                messagebox.showinfo("Success", f"Signal 1 folded successfully as '{name}'.")
            except Exception as e:
                messagebox.showerror("Error", f"Folding failed: {e}")
        else:
            messagebox.showwarning("Warning", "Please load Signal 1.")

    def plot_signals(self):
        self.task1_ax.clear()
        representation = self.representation.get()
        plot_continuous = representation == "Continuous"

        # List of signals to plot
        signals_to_plot = []
        for key in self.signals:
            signals_to_plot.append((key, self.signals[key]))

        if not signals_to_plot:
            messagebox.showwarning("Warning", "No signals to plot.")
            return

        for name, signal in signals_to_plot:
            t = signal.indices
            s = signal.samples
            if plot_continuous:
                self.task1_ax.plot(t, s, label=name)
            else:
                try:
                    self.task1_ax.stem(t, s, label=name, use_line_collection=True)
                except TypeError:
                    # Fallback if use_line_collection is not supported
                    self.task1_ax.stem(t, s, label=name)

        self.task1_ax.set_xlabel("n")
        self.task1_ax.set_ylabel("Amplitude")
        self.task1_ax.set_title("Signal Visualization")
        self.task1_ax.legend()
        self.task1_ax.grid(True)
        self.task1_canvas.draw()

    # Bonus Widget
    def create_bonus_widgets(self, frame):
        # Input Frame
        input_frame = ttk.LabelFrame(frame, text="Sine Wave Parameters")
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Amplitude Entry
        ttk.Label(input_frame, text="Amplitude (A):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.bonus_amplitude_entry = ttk.Entry(input_frame)
        self.bonus_amplitude_entry.grid(row=0, column=1, padx=5, pady=5)

        # Frequency Entry
        ttk.Label(input_frame, text="Frequency (Hz):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.bonus_frequency_entry = ttk.Entry(input_frame)
        self.bonus_frequency_entry.grid(row=1, column=1, padx=5, pady=5)

        # Phase Shift Entry
        ttk.Label(input_frame, text="Phase Shift (radians):").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        self.bonus_phase_entry = ttk.Entry(input_frame)
        self.bonus_phase_entry.grid(row=2, column=1, padx=5, pady=5)

        # Number of Samples Entry
        ttk.Label(input_frame, text="Number of Samples:").grid(row=3, column=0, padx=5, pady=5, sticky='e')
        self.bonus_samples_entry = ttk.Entry(input_frame)
        self.bonus_samples_entry.grid(row=3, column=1, padx=5, pady=5)
        self.bonus_samples_entry.insert(0, "100")  # Default to 100 samples

        # Sampling Frequency Entry
        ttk.Label(input_frame, text="Sampling Frequency (Hz):").grid(row=4, column=0, padx=5, pady=5, sticky='e')
        self.bonus_sampling_freq_entry = ttk.Entry(input_frame)
        self.bonus_sampling_freq_entry.grid(row=4, column=1, padx=5, pady=5)
        self.bonus_sampling_freq_entry.insert(0, "1000")  # Default to 1000 Hz

        # Generate Sine Button
        generate_sine_btn = ttk.Button(input_frame, text="Generate Sine Wave", command=self.generate_bonus_sine)
        generate_sine_btn.grid(row=5, column=0, columnspan=2, pady=10)

        # Operations Frame
        operations_frame = ttk.LabelFrame(frame, text="Operations")
        operations_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Autocorrelation Button
        autocorr_sine_btn = ttk.Button(operations_frame, text="Autocorrelate Sine Wave", command=self.autocorrelate_sine)
        autocorr_sine_btn.grid(row=0, column=0, padx=5, pady=5)

        # Generate Noise Button
        generate_noise_btn = ttk.Button(operations_frame, text="Generate AWGN Noise", command=self.generate_bonus_noise)
        generate_noise_btn.grid(row=0, column=1, padx=5, pady=5)

        # Add Noise Button
        add_noise_btn = ttk.Button(operations_frame, text="Add Noise to Sine", command=self.add_noise_to_sine)
        add_noise_btn.grid(row=1, column=0, padx=5, pady=5)

        # Autocorrelation of Corrupted Signal Button
        autocorr_corrupted_btn = ttk.Button(operations_frame, text="Autocorrelate Corrupted Signal", command=self.autocorrelate_corrupted)
        autocorr_corrupted_btn.grid(row=1, column=1, padx=5, pady=5)

        # Plot Frame
        plot_frame = ttk.LabelFrame(frame, text="Plots")
        plot_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")

        # Create a grid of subplots: 2 rows x 3 columns
        self.bonus_figure, self.bonus_axes = plt.subplots(2, 3, figsize=(18, 10))
        self.bonus_canvas = FigureCanvasTkAgg(self.bonus_figure, master=plot_frame)
        self.bonus_canvas.draw()
        self.bonus_canvas.get_tk_widget().pack()

        # Initialize storage for signals
        self.bonus_signals = {}

    def generate_bonus_sine(self):
        try:
            amplitude = float(self.bonus_amplitude_entry.get())
            frequency = float(self.bonus_frequency_entry.get())
            phase_shift = float(self.bonus_phase_entry.get())
            num_samples = int(self.bonus_samples_entry.get())
            sampling_freq = float(self.bonus_sampling_freq_entry.get())

            if sampling_freq < 2 * frequency:
                messagebox.showerror("Error", "Sampling frequency must be at least twice the frequency of the sine wave (Nyquist rate).")
                return

            # Generate sine wave
            sine_signal = Signal.generate_sine(amplitude, phase_shift, frequency, sampling_freq, num_samples)
            name = f"Sine_A{amplitude}_F{frequency}_P{phase_shift}"
            self.bonus_signals['sine'] = sine_signal
            self.signals[name] = sine_signal  # Also store in main signals dictionary

            # Plot sine wave
            ax = self.bonus_axes[0, 0]
            ax.clear()
            ax.plot(sine_signal.time_values, sine_signal.samples, label='Sine Wave', color='blue')
            ax.set_title("Sine Wave")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.legend()
            ax.grid(True)

            # Clear the autocorrelation plot
            self.bonus_axes[0, 1].clear()
            self.bonus_axes[0, 1].set_title("Autocorrelation of Sine Wave")
            self.bonus_axes[0, 1].set_xlabel("Lag")
            self.bonus_axes[0, 1].set_ylabel("Correlation")
            self.bonus_axes[0, 1].grid(True)

            # Clear noise plot
            self.bonus_axes[0, 2].clear()
            self.bonus_axes[0, 2].set_title("AWGN Noise")
            self.bonus_axes[0, 2].set_xlabel("Sample Index")
            self.bonus_axes[0, 2].set_ylabel("Amplitude")
            self.bonus_axes[0, 2].grid(True)

            # Clear corrupted signal plot
            self.bonus_axes[1, 0].clear()
            self.bonus_axes[1, 0].set_title("Corrupted Sine Wave with AWGN")
            self.bonus_axes[1, 0].set_xlabel("Sample Index")
            self.bonus_axes[1, 0].set_ylabel("Amplitude")
            self.bonus_axes[1, 0].grid(True)

            # Clear autocorrelation of corrupted signal plot
            self.bonus_axes[1, 1].clear()
            self.bonus_axes[1, 1].set_title("Autocorrelation of Corrupted Signal")
            self.bonus_axes[1, 1].set_xlabel("Lag")
            self.bonus_axes[1, 1].set_ylabel("Correlation")
            self.bonus_axes[1, 1].grid(True)

            self.bonus_canvas.draw()

            messagebox.showinfo("Success", f"Sine wave '{name}' generated and plotted.")

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values.")


    def autocorrelate_sine(self):
        if 'sine' not in self.bonus_signals:
            messagebox.showwarning("Warning", "Please generate the sine wave first.")
            return

        sine_signal = self.bonus_signals['sine']
        autocorr_signal = SignalOperations.correlate(sine_signal, sine_signal)
        self.bonus_signals['autocorr_sine'] = autocorr_signal
        self.signals['Autocorr_Sine'] = autocorr_signal  # Store in main signals

        # Plot autocorrelation
        ax = self.bonus_axes[0, 1]
        ax.clear()
        ax.plot(autocorr_signal.indices, autocorr_signal.samples, label='Autocorrelation', color='purple')
        ax.set_title("Autocorrelation of Sine Wave")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Correlation")
        ax.legend()
        ax.grid(True)

        self.bonus_canvas.draw()

        messagebox.showinfo("Success", "Autocorrelation of sine wave computed and plotted.")

    def generate_bonus_noise(self):
        try:
            if 'sine' not in self.bonus_signals:
                messagebox.showwarning("Warning", "Please generate the sine wave first.")
                return

            sine_signal = self.bonus_signals['sine']
            num_samples = len(sine_signal.samples)

            # Generate AWGN with zero mean and standard deviation
            noise_std = 0.5  # Fixed standard deviation; can be made adjustable if desired
            noise_samples = np.random.normal(0, noise_std, num_samples)
            noise_signal = Signal(list(range(num_samples)), noise_samples.tolist())
            name = f"AWGN_Std{noise_std}"
            self.bonus_signals['noise'] = noise_signal
            self.signals[name] = noise_signal  # Store in main signals

            # Plot noise
            ax = self.bonus_axes[0, 2]
            ax.clear()
            ax.plot(noise_signal.indices, noise_signal.samples, label='AWGN Noise', color='orange')
            ax.set_title("AWGN Noise")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Amplitude")
            ax.legend()
            ax.grid(True)

            self.bonus_canvas.draw()

            messagebox.showinfo("Success", f"AWGN noise '{name}' generated and plotted.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate noise: {e}")

    def add_noise_to_sine(self):
        if 'sine' not in self.bonus_signals or 'noise' not in self.bonus_signals:
            messagebox.showwarning("Warning", "Please generate both sine wave and noise first.")
            return

        sine_signal = self.bonus_signals['sine']
        noise_signal = self.bonus_signals['noise']

        # Ensure both signals have the same number of samples
        if len(sine_signal.samples) != len(noise_signal.samples):
            messagebox.showerror("Error", "Sine wave and noise signals must have the same number of samples.")
            return

        # Add noise to sine
        corrupted_samples = np.array(sine_signal.samples) + np.array(noise_signal.samples)
        corrupted_signal = Signal(sine_signal.indices, corrupted_samples.tolist())
        name = "Corrupted_Sine_Noise"
        self.bonus_signals['corrupted'] = corrupted_signal
        self.signals[name] = corrupted_signal  # Store in main signals

        # Plot corrupted signal
        ax = self.bonus_axes[1, 0]
        ax.clear()
        ax.plot(corrupted_signal.indices, corrupted_signal.samples, label='Sine + Noise', color='green')
        ax.set_title("Corrupted Sine Wave with AWGN")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)

        self.bonus_canvas.draw()

        messagebox.showinfo("Success", "Noise added to sine wave and plotted.")



    def autocorrelate_corrupted(self):
        if 'corrupted' not in self.bonus_signals:
            messagebox.showwarning("Warning", "Please add noise to the sine wave first.")
            return

        corrupted_signal = self.bonus_signals['corrupted']
        autocorr_corrupted = SignalOperations.correlate(corrupted_signal, corrupted_signal)
        self.bonus_signals['autocorr_corrupted'] = autocorr_corrupted
        self.signals['Autocorr_Corrupted'] = autocorr_corrupted  # Store in main signals

        # Plot autocorrelation of corrupted signal
        ax = self.bonus_axes[1, 1]
        ax.clear()
        ax.plot(autocorr_corrupted.indices, autocorr_corrupted.samples, label='Autocorrelation', color='red')
        ax.set_title("Autocorrelation of Corrupted Signal")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Correlation")
        ax.legend()
        ax.grid(True)

        self.bonus_canvas.draw()

        messagebox.showinfo("Success", "Autocorrelation of corrupted signal computed and plotted.")

    



def compare_signals(signal1, signal2, tolerance=1e-3):
    if signal1.indices != signal2.indices:
        return False
    for s1, s2 in zip(signal1.samples, signal2.samples):
        if abs(s1 - s2) > tolerance:
            return False
    return True

def Compare_Signals(file_name, Your_indices, Your_samples):      
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break

    print("Current Output Test file is: ")
    print(file_name)
    print("\n")

    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Test case failed, your signal has different length from the expected one")
        return

    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            print("Test case failed, your signal has different indices from the expected one") 
            return

    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal has different values from the expected one") 
            return

    print("Test case passed successfully")

# Compare functions for DFT amplitudes and phases
def SignalComapreAmplitude(SignalInput=[], SignalOutput=[], tolerance=1e-3):
    if len(SignalInput) != len(SignalOutput):
        return False
    for i in range(len(SignalInput)):
        if abs(SignalInput[i] - SignalOutput[i]) > tolerance:
            return False
    return True


def SignalComaprePhaseShift(SignalInput=[], SignalOutput=[], tolerance=1e-3):
    if len(SignalInput) != len(SignalOutput):
        return False
    for i in range(len(SignalInput)):
        # Normalize phases to the range [-π, π]
        A = ((SignalInput[i] + math.pi) % (2 * math.pi)) - math.pi
        B = ((SignalOutput[i] + math.pi) % (2 * math.pi)) - math.pi
        if abs(A - B) > tolerance:
            return False
    return True


def read_dft_output(file_path):
    amplitudes = []
    phases = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        N = int(lines[2].strip())  # Number of samples
        for line in lines[3:3+N]:
            parts = line.strip().split()
            amplitude = float(parts[0])
            phase = float(parts[1].replace('f', ''))
            amplitudes.append(amplitude)
            phases.append(phase)
    return amplitudes, phases

# Test Functions
def test_dft_idft():
    input_signal = Signal.from_file('input_Signal_DFT.txt')
    expected_amplitudes, expected_phases = read_dft_output('Output_Signal_DFT_A,Phase.txt')

    X_real, X_imag = SignalOperations.compute_dft(input_signal)

    # Compute amplitude and phase
    amplitudes = [math.sqrt(r**2 + im**2) for r, im in zip(X_real, X_imag)]
    phases = [math.atan2(im, r) for r, im in zip(X_real, X_imag)]


    # Compare amplitudes and phases
    amplitude_match = SignalComapreAmplitude(amplitudes, expected_amplitudes)
    phase_match = SignalComaprePhaseShift(phases, expected_phases)

    if amplitude_match and phase_match:
        print("DFT Test Passed")
    else:
        print("DFT Test Failed")

    # Test IDFT
    reconstructed_samples = SignalOperations.compute_idft(X_real, X_imag)
    reconstructed_signal = Signal(input_signal.indices, reconstructed_samples)

    # Compare reconstructed signal with original
    if compare_signals(input_signal, reconstructed_signal):
        print("IDFT Test Passed")
    else:
        print("IDFT Test Failed")


if __name__ == "__main__":
    # Run tests
    # test_dft_idft()
    # Start the GUI
    root = tk.Tk()
    app = SignalProcessingApp(root)
    root.mainloop()
