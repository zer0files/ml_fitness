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
        filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
    )

    if first_file:
        print(f"First file selected: {first_file}")
    else:
        print("No first file selected.")
        return first_file

    # Second file selection
    second_file = filedialog.askopenfilename(
        title="Select the second file",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
    )

    if second_file:
        print(f"Second file selected: {second_file}")
    else:
        print("No second file selected.")
        return

# Call the function
select_files_separately()