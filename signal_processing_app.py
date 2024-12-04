import numpy as np
import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Signal Class
class Signal:
    def __init__(self, indices=None, samples=None):
        self.indices = indices if indices is not None else []
        self.samples = samples if samples is not None else []

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
        return cls(indices, samples.tolist())

    @classmethod
    def generate_cosine(cls, amplitude, phase_shift, frequency, sampling_freq):
        N = 50  # Fixed number of samples
        t = np.arange(N) / sampling_freq  # Time vector
        samples = amplitude * np.cos(2 * np.pi * frequency * t + phase_shift)
        indices = list(range(N))
        return cls(indices, samples.tolist())

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

    


# Signal Processing App with GUI
class SignalProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Signal Processing App")
        self.signals = {}  # Dictionary to store loaded/generated signals

        self.create_widgets()

    def create_widgets(self):
        # Create a Notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill='both')

        # Create frames for each tab
        task1_frame = ttk.Frame(notebook)
        task2_frame = ttk.Frame(notebook)
        task3_frame = ttk.Frame(notebook)
        task4_frame = ttk.Frame(notebook)

        # Add tabs to notebook
        notebook.add(task1_frame, text='Task 1 - Signal Operations')
        notebook.add(task2_frame, text='Task 2 - Signal Generation')
        notebook.add(task3_frame, text='Task 3 - Quantization')
        notebook.add(task4_frame, text='Task 4 - Advanced Operations')

        # Create widgets for each task
        self.create_task1_widgets(task1_frame)
        self.create_task2_widgets(task2_frame)
        self.create_task3_widgets(task3_frame)
        self.create_task4_widgets(task4_frame)

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
            indices = generated_signal.indices
            samples = generated_signal.samples

            if representation == "Continuous":
                self.task2_ax.plot(indices, samples, label=f'{wave_type.capitalize()} Signal')
            else:
                try:
                    self.task2_ax.stem(indices, samples, label=f'{wave_type.capitalize()} Signal', use_line_collection=True)
                except TypeError:
                    self.task2_ax.stem(indices, samples, label=f'{wave_type.capitalize()} Signal')

            self.task2_ax.set_xlabel('Sample Index')
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
            messagebox.showerror("Error", "Please enter valid integer values for levels or bits.")
    
    # Task 4: Moving Average Widgets
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


    # Existing methods for Task 1
    def load_signal(self, signal_number):
        file_path = filedialog.askopenfilename(title="Select Signal File", filetypes=(("Text Files", "*.txt"),))
        if file_path:
            try:
                signal = Signal.from_file(file_path)
                self.signals[f"Signal{signal_number}"] = signal
                messagebox.showinfo("Success", f"Signal {signal_number} loaded successfully.")
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

def compare_signals(signal1, signal2, tolerance=1e-3):
    if signal1.indices != signal2.indices:
        return False
    for s1, s2 in zip(signal1.samples, signal2.samples):
        if abs(s1 - s2) > tolerance:
            return False
    return True

# Test Functions
def test_convolution():
    signal1 = Signal.from_file('Signal 1.txt')
    signal2 = Signal.from_file('Signal 2.txt')
    expected_output = Signal.from_file('Conv_output.txt')

    result_signal = SignalOperations.convolve(signal1, signal2)
    if compare_signals(result_signal, expected_output):
        print("Convolution Test Passed")
    else:
        print("Convolution Test Failed")

def test_derivatives():
    input_signal = Signal.from_file('Derivative_input.txt')
    expected_first_derivative = Signal.from_file('1st_derivative_out.txt')
    expected_second_derivative = Signal.from_file('2nd_derivative_out.txt')

    first_derivative = SignalOperations.first_derivative(input_signal)
    second_derivative = SignalOperations.second_derivative(input_signal)

    if compare_signals(first_derivative, expected_first_derivative):
        print("First Derivative Test Passed")
    else:
        print("First Derivative Test Failed")

    if compare_signals(second_derivative, expected_second_derivative):
        print("Second Derivative Test Passed")
    else:
        print("Second Derivative Test Failed")

def test_moving_average():
    input_signal = Signal.from_file('MovingAvg_input.txt')

    # Test 1: Window size = 3
    expected_output1 = Signal.from_file('MovingAvg_out1.txt')
    moving_avg1 = SignalOperations.moving_average(input_signal, window_size=3)
    if compare_signals(moving_avg1, expected_output1):
        print("Moving Average Test 1 Passed")
    else:
        print("Moving Average Test 1 Failed")

    # Test 2: Window size = 5
    expected_output2 = Signal.from_file('MovingAvg_out2.txt')
    moving_avg2 = SignalOperations.moving_average(input_signal, window_size=5)
    if compare_signals(moving_avg2, expected_output2):
        print("Moving Average Test 2 Passed")
    else:
        print("Moving Average Test 2 Failed")



if __name__ == "__main__":
    # test_convolution()
    # test_derivatives()
    # test_moving_average()
    root = tk.Tk()
    app = SignalProcessingApp(root)
    root.mainloop()
