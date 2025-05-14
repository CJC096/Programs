from tkinter import *
from tkinter import ttk
from googletrans import Translator, LANGUAGES

def Translate():
    translator = Translator()
    text = Input_text.get("1.0", END)
    translated = asyncio.run(translator.translate(text=text, dest=dest_lang.get()))
    Output_text.delete("1.0", END)
    Output_text.insert(END, translated.text)


root = Tk()
root.geometry('1100x320')
root.resizable(False, False)
root.iconbitmap('logo.ico')
root['bg'] = 'teal'
root.title('Language Translator')

# Title label
Label(root, text="Language Translator", font=("Times New Roman", 20, "bold"), bg='teal', fg='black').pack(pady=10)

# "Enter Text" Label and Text Widget
Label(root, text="Enter Text", font=("Times New Roman", 14, "bold"), bg='teal', fg= 'black').place(x=225, y=90)
Input_text = Text(root, wrap=WORD)
Input_text.place(x=85, y=130, width=400, height=100)


# "Output Text" Label and Text Widget
Label(root, text="Output Text", font=("Times New Roman", 14, "bold"), bg='teal', fg='black').place(x=780, y=90)
Output_text = Text(root, wrap=WORD)
Output_text.place(x=645, y=130, width=400, height=100)

language = list(LANGUAGES.values())

dest_lang= ttk.Combobox(root, values= language, width=22)
dest_lang.place(x=200, y=235)
dest_lang.set('choose language')

def Translate():
    translator = Translator()
    text = Input_text.get("1.0", END)
    translated = translator.translate(text=text, dest=dest_lang.get())
    Output_text.delete(1.0, END)
    Output_text.insert(END, translated.text)

trans_btn = Button(root, text='Translate', font=("Times New Roman", 14, "bold"), pady = 5, command= Translate, bg= 'orange', activebackground='green')
trans_btn.place(x=520, y=180)

root.mainloop()
