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
    def generate_sine(cls, amplitude, phase_shift, frequency, sampling_freq, duration):
        t = np.arange(0, duration, 1/sampling_freq)
        samples = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)
        indices = list(range(len(samples)))
        return cls(indices, samples.tolist())

    @classmethod
    def generate_cosine(cls, amplitude, phase_shift, frequency, sampling_freq, duration):
        t = np.arange(0, duration, 1/sampling_freq)
        samples = amplitude * np.cos(2 * np.pi * frequency * t + phase_shift)
        indices = list(range(len(samples)))
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



# Task 1 Tests
def ReadSignalFile(file_name):
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
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
    return expected_indices,expected_samples



def AddSignalSamplesAreEqual(userFirstSignal,userSecondSignal,Your_indices,Your_samples):
    if(userFirstSignal=='Signal1.txt' and userSecondSignal=='Signal2.txt'):
        file_name="add.txt"  # write here the path of the add output file
    expected_indices,expected_samples=ReadSignalFile(file_name)          
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Addition Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Addition Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Addition Test case failed, your signal have different values from the expected one") 
            return
    print("Addition Test case passed successfully")



def SubSignalSamplesAreEqual(userFirstSignal,userSecondSignal,Your_indices,Your_samples):
    if(userFirstSignal=='Signal1.txt' and userSecondSignal=='Signal2.txt'):
        file_name="subtract.txt" # write here the path of the subtract output file
        
    expected_indices,expected_samples=ReadSignalFile(file_name)   
    
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Subtraction Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Subtraction Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Subtraction Test case failed, your signal have different values from the expected one") 
            return
    print("Subtraction Test case passed successfully")



def MultiplySignalByConst(User_Const,Your_indices,Your_samples):
    if(User_Const==5):
        file_name="mul5.txt"  # write here the path of the mul5 output file
        
    expected_indices,expected_samples=ReadSignalFile(file_name)      
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Multiply by "+str(User_Const)+ " Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Multiply by "+str(User_Const)+" Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Multiply by "+str(User_Const)+" Test case failed, your signal have different values from the expected one") 
            return
    print("Multiply by "+str(User_Const)+" Test case passed successfully")



def ShiftSignalByConst(Shift_value,Your_indices,Your_samples):
    if(Shift_value==3):  #x(n+k)
        file_name="advance3.txt" # write here the path of delay3 output file
    elif(Shift_value==-3): #x(n-k)
        file_name="delay3.txt" # write here the path of advance3 output file
        
    expected_indices,expected_samples=ReadSignalFile(file_name)      
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Shift by "+str(Shift_value)+" Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Shift by "+str(Shift_value)+" Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Shift by "+str(Shift_value)+" Test case failed, your signal have different values from the expected one") 
            return
    print("Shift by "+str(Shift_value)+" Test case passed successfully")



def Folding(Your_indices,Your_samples):
    file_name = "folding.txt"  # write here the path of the folding output file
    expected_indices,expected_samples=ReadSignalFile(file_name)      
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Folding Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Folding Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Folding Test case failed, your signal have different values from the expected one") 
            return
    print("Folding Test case passed successfully")



# Signal Processing App with GUI
class SignalProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Signal Processing App")
        self.signals = {}  # Dictionary to store loaded/generated signals

        self.create_menu()
        self.create_widgets()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Signal Generation Menu
        gen_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Signal Generation", menu=gen_menu)
        gen_menu.add_command(label="Sine Wave", command=self.generate_sine_wave)
        gen_menu.add_command(label="Cosine Wave", command=self.generate_cosine_wave)

    def create_widgets(self):
        # Frame for signal operations
        operations_frame = ttk.LabelFrame(self.root, text="Signal Operations")
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
        viz_frame = ttk.LabelFrame(self.root, text="Visualization Options")
        viz_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.representation = tk.StringVar(value="Discrete")
        cont_rb = ttk.Radiobutton(viz_frame, text="Continuous", variable=self.representation, value="Continuous")
        cont_rb.grid(row=0, column=0, padx=5, pady=5)
        disc_rb = ttk.Radiobutton(viz_frame, text="Discrete", variable=self.representation, value="Discrete")
        disc_rb.grid(row=0, column=1, padx=5, pady=5)

        plot_btn = ttk.Button(viz_frame, text="Plot Signals", command=self.plot_signals)
        plot_btn.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        # Frame for Matplotlib plot
        plot_frame = ttk.Frame(self.root)
        plot_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10)

        self.figure, self.ax = plt.subplots(figsize=(8,6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

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
        self.ax.clear()
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
                self.ax.plot(t, s, label=name)
            else:
                try:
                    self.ax.stem(t, s, label=name, use_line_collection=True)
                except TypeError:
                    # Fallback if use_line_collection is not supported
                    self.ax.stem(t, s, label=name)

        self.ax.set_xlabel("n")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title("Signal Visualization")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def generate_sine_wave(self):
        self.open_signal_generation_window('sine')

    def generate_cosine_wave(self):
        self.open_signal_generation_window('cosine')

    def open_signal_generation_window(self, wave_type):
        def perform_generation():
            try:
                amplitude = float(amplitude_entry.get())
                phase = float(phase_entry.get())
                freq = float(freq_entry.get())
                sampling_freq = float(sampling_freq_entry.get())
                duration = float(duration_entry.get())

                # Check Sampling Theorem
                if sampling_freq < 2 * freq:
                    messagebox.showerror("Error", "Sampling frequency must be at least twice the analog frequency (Nyquist rate).")
                    return

                if wave_type == 'sine':
                    generated_signal = Signal.generate_sine(amplitude, phase, freq, sampling_freq, duration)
                else:
                    generated_signal = Signal.generate_cosine(amplitude, phase, freq, sampling_freq, duration)

                # Assign a unique name
                name = f"Generated_{wave_type.capitalize()}_{len(self.signals)+1}"
                self.signals[name] = generated_signal
                messagebox.showinfo("Success", f"{wave_type.capitalize()} wave generated successfully as '{name}'.")
                gen_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numerical values.")

        gen_window = tk.Toplevel(self.root)
        gen_window.title(f"Generate {wave_type.capitalize()} Wave")

        ttk.Label(gen_window, text="Amplitude (A):").grid(row=0, column=0, padx=5, pady=5)
        amplitude_entry = ttk.Entry(gen_window)
        amplitude_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(gen_window, text="Phase Shift (θ in radians):").grid(row=1, column=0, padx=5, pady=5)
        phase_entry = ttk.Entry(gen_window)
        phase_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(gen_window, text="Analog Frequency (Hz):").grid(row=2, column=0, padx=5, pady=5)
        freq_entry = ttk.Entry(gen_window)
        freq_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(gen_window, text="Sampling Frequency (Hz):").grid(row=3, column=0, padx=5, pady=5)
        sampling_freq_entry = ttk.Entry(gen_window)
        sampling_freq_entry.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(gen_window, text="Duration (seconds):").grid(row=4, column=0, padx=5, pady=5)
        duration_entry = ttk.Entry(gen_window)
        duration_entry.grid(row=4, column=1, padx=5, pady=5)

        ttk.Button(gen_window, text="Generate", command=perform_generation).grid(row=5, column=0, columnspan=2, pady=10)

# Main function for Task 1 testing
def main_task1():
    try:
        # Read Signal1 and Signal2
        signal1 = Signal.from_file('Signal1.txt')
        signal2 = Signal.from_file('Signal2.txt')
    except Exception as e:
        print(f"Error reading signal files: {e}")
        return

    # Perform Addition
    print("Testing Addition:")
    added_signal = SignalOperations.add(signal1, signal2)
    AddSignalSamplesAreEqual("Signal1.txt", "Signal2.txt", added_signal.indices, added_signal.samples)

    # Perform Subtraction
    print("\nTesting Subtraction:")
    subtracted_signal = SignalOperations.subtract(signal1, signal2)
    SubSignalSamplesAreEqual("Signal1.txt", "Signal2.txt", subtracted_signal.indices, subtracted_signal.samples)

    # Perform Multiplication by 5
    print("\nTesting Multiplication by 5:")
    multiplied_signal = SignalOperations.multiply(signal1, 5)
    MultiplySignalByConst(5, multiplied_signal.indices, multiplied_signal.samples)

    # Perform Shift by +3 (Advance)
    print("\nTesting Shift by +3 (Advance):")
    shifted_signal = SignalOperations.shift(signal1, 3)
    ShiftSignalByConst(3, shifted_signal.indices, shifted_signal.samples)

    # Perform Shift by +3 (Advance)
    print("\nTesting Shift by -3 (Delay):")
    shifted_signal = SignalOperations.shift(signal1, 3)
    ShiftSignalByConst(3, shifted_signal.indices, shifted_signal.samples)

    # Perform Folding
    print("\nTesting Folding:")
    folded_signal = SignalOperations.fold(signal1)
    Folding(folded_signal.indices, folded_signal.samples)


if __name__ == "__main__":
    # Uncomment the following line to run Task 1 tests
    # main_task1()

    # Run the GUI application for Task 2
    root = tk.Tk()
    app = SignalProcessingApp(root)
    root.mainloop()
