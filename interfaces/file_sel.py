import tkinter as tk
from tkinter import filedialog

# Function to select two files separately
def select_files_separately():
    # Create a Tkinter root window (hidden)
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    # First file selection
    first_file = filedialog.askopenfilename(
        title="Select the first file",
        filetypes=(("Text files", "*.csv"), ("All files", "*.*"))
        # filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
    )

    if first_file:
        print(f"First file selected: {first_file}")
    else:
        print("No first file selected.")
        return

    # Second file selection
    second_file = filedialog.askopenfilename(
        title="Select the second file",
        filetypes=(("Text files", "*.csv"), ("All files", "*.*"))
    )

    if second_file:
        print(f"Second file selected: {second_file}")
    else:
        print("No second file selected.")
        return

# Call the function
file1 = select_files_separately()
print(file1)