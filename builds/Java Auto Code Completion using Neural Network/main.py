import json
import os
import random
import subprocess
import tkinter as tk
from tkinter import font
from tkinter.scrolledtext import ScrolledText
import re
import torch
from model import RNNModule  # Assuming RNNModule is defined in model.py

created_java_files = []  # List to keep track of created .java files


def generate_code(keyword, code_templates):
    keyword_lower = keyword.lower()

    matched_templates = []
    for template in code_templates:
        patterns_lower = [pattern.lower() for pattern in template["patterns"]]

        if any(pattern_lower == keyword_lower for pattern_lower in patterns_lower):
            matched_templates.append(template)

    if matched_templates:
        generated_data = []
        for template in matched_templates:
            code_text = template["responses"][0]
            explanations = template.get("explanations", ["No explanations available."])
            explanation = random.choice(explanations)
            generated_data.append((code_text, explanation))
        return generated_data
    else:
        return None


def save_code_files(code_text):
    class_or_interface_pattern = r'''
        (                               # Start of capturing group
            (public\s+)?                # Optional 'public' modifier
            (class|interface)\s+\w+     # 'class' or 'interface' keyword followed by a name
            (\s+extends\s+\w+)?         # Optional 'extends' clause
            (\s+implements\s+[\w,\s]+)? # Optional 'implements' clause
            \s*{                        # Opening brace of the class or interface
            (?:                         # Non-capturing group to match the class or interface body
                [^{}]*                  # Match anything except braces
                |                       # OR
                \{                      # Opening brace of a nested block
                (?:                     # Non-capturing group to match nested content
                    [^{}]*              # Match anything except braces
                    |                   # OR
                    \{[^{}]*\}          # Nested block with balanced braces
                )*                      # Repeat as necessary
                \}                      # Closing brace of a nested block
            )*                          # Repeat as necessary
        \}                              # Closing brace of the class or interface
        )                               # End of capturing group
    '''

    class_or_interface_pattern = re.compile(class_or_interface_pattern, re.VERBOSE)

    class_matches = class_or_interface_pattern.findall(code_text)
    print("Matches:", class_matches)

    class_names = []  # List to collect class or interface names

    for match in class_matches:
        class_code = match[0]  # Extract the full class or interface code from the match tuple
        class_name_match = re.search(r'(class|interface)\s+(\w+)', class_code)
        if class_name_match:
            class_type = class_name_match.group(1)
            class_name = class_name_match.group(2)
            filename = f'{class_name}.java'  # Corrected file naming to save as .java files
            print("Saving file:", filename)  # Debug statement to print the filename being saved
            with open(filename, 'w') as f:
                f.write(class_code)
            created_java_files.append(filename)  # Add to the global list
            class_names.append(class_name)  # Collect the class or interface name

    return class_names  # Return the list of class or interface names

def center_window(root, width, height):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")


def run_program(class_name):
    global last_generated_class_name, text_widget  # Use the global variable

    try:
        # Remove '{' symbol if present in class name
        class_name = class_name.replace("{", "")

        # Compile the generated Java code to create .class file
        subprocess.run(["javac", f"{class_name}.java"])  # Compile the generated Java code

        # Run the Java program
        result = subprocess.run(["java", class_name], capture_output=True, text=True)
        text_widget.insert(tk.END, f"Monik.A.I.: Program Output:\n{result.stdout}\n")

        # Remove the generated .java and .class files after running
        os.remove(f'{class_name}.java')
        os.remove(f'{class_name}.class')
    except Exception as e:
        text_widget.insert(tk.END, f"Monik.A.I.: Error running the program. {e}\n")


def send_message(user_input):
    global last_generated_class_name, text_widget, intents, input_entry, code_templates, root
    if not user_input:
        return

    text_widget.config(state=tk.NORMAL)
    text_widget.insert(tk.END, "You: " + user_input + "\n", "user")

    if user_input.lower() in ["run program", "run the program"]:
        if last_generated_class_name:
            run_program(last_generated_class_name)
        else:
            text_widget.insert(tk.END, f"Monik.A.I.: No program to run. Generate code first.\n", "bot")
    else:
        generated_data = generate_code(user_input, code_templates)
        if generated_data:

            for code_text, explanation in generated_data:
                class_names = save_code_files(code_text)

                if class_names:
                    last_generated_class_name = class_names[-1]  # Set the last class as the main class to run
                    response_prefix = random.choice([
                        "Sure! Here's an example of a code for that:\n\n",
                        "Here's an example of the code:\n\n",
                        "Here's some code for that:\n\n"
                    ])
                    response = f"{response_prefix}{code_text}\nExplanation: {explanation}\n\n"
                    text_widget.insert(tk.END, "Monik.A.I.: " + response, "bot")
                else:
                    text_widget.insert(tk.END, "Monik.A.I.: Failed to generate code.\n", "bot")
        else:
            matched_intent = None
            for intent in intents:
                for pattern in intent["patterns"]:
                    if pattern.lower() in user_input.lower():
                        matched_intent = intent
                        break
                if matched_intent:
                    break

            if matched_intent:
                response = random.choice(matched_intent['responses'])
                text_widget.insert(tk.END, "Monik.A.I.: " + response + "\n", "bot")
                if matched_intent['tag'] == 'farewell':
                    root.destroy()
            else:
                text_widget.insert(tk.END, "Monik.A.I.: I'm sorry. I do not understand.\n", "bot")

    text_widget.insert(tk.END, "\n")
    text_widget.config(state=tk.DISABLED)
    text_widget.see(tk.END)
    input_entry.delete(0, tk.END)


def on_closing():
    global created_java_files
    for filename in created_java_files:
        if os.path.exists(filename):
            os.remove(filename)
        class_file = filename.replace('.java', '.class')
        if os.path.exists(class_file):
            os.remove(class_file)
    root.destroy()


def main():
    global text_widget, intents, input_entry, code_templates, root, last_generated_class_name

    last_generated_class_name = None  # Initialize the global variable

    # Load data and initialize the chatbot
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('mix.json', 'r') as f:
        data = json.load(f)
        intents = data["intents"]
        code_templates = data["code"]

    # Load and initialize the model
    FILE = "data.pth"
    model_data = torch.load(FILE)

    print("Keys in model_data:", model_data.keys())  # Print keys to inspect the structure

    output_size_model = model_data['fc.bias'].shape[0]  # Get the output size from the model data
    tags = ['tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'tag6', 'tag7', 'tag8',
            'tag9']  # Define tags based on your requirement

    # Print the shapes of the weight and bias tensors to determine the output size
    print("Shape of fc.weight:", model_data['fc.weight'].shape)
    print("Shape of fc.bias:", model_data['fc.bias'].shape)

    # Extract correct sizes from model_data
    hidden_size = model_data['batch_norm.weight'].shape[0]  # Size from the batch_norm layer
    output_size = model_data['fc.bias'].shape[0]  # Size from the fc layer
    input_size = 128  # Assuming input_size as a constant

    model_state = model_data

    model = RNNModule(input_size, hidden_size, output_size).to(device)

    # Load the model's state dictionary with strict=False
    model.load_state_dict(model_state, strict=False)
    model.eval()

    root = tk.Tk()
    root.title("Monik.A.I. Chatbot")

    window_width = 900
    window_height = 600
    center_window(root, window_width, window_height)

    text_widget = ScrolledText(root, wrap=tk.WORD, font=font.Font(family="Arial", size=12))
    text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    input_frame = tk.Frame(root)
    input_frame.pack(fill=tk.X, padx=10, pady=5)

    input_entry = tk.Entry(input_frame, font=font.Font(family="Arial", size=12))
    input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    input_entry.bind("<Return>", lambda event: send_message(input_entry.get()))

    send_button = tk.Button(input_frame, text="Send", font=font.Font(family="Arial", size=12),
                            command=lambda: send_message(input_entry.get()))
    send_button.pack(side=tk.RIGHT)

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
