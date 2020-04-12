import json
from threading import Thread

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from ml import MyClassifier, PREDICTIONS


CONFIG = {
    "clf": MyClassifier(logging=True),
    "input_data": "",
    "translated": False,
    "training_thread": None,
    "stopped": True
}

COLORS = [
    [0.6, "red"],
    [0.7, "orange"],
    [0.85, "yellow"],
    [1, "green"]
]

translate_dict = json.load(open("..\\train_data\\__t.json"))


def get_color(value, negative=False):
    for th, color in COLORS:
        if th > (1 - value if negative else value):
            return color


def load_data():
    t = Thread(target=CONFIG["clf"].load_data)
    t.start()
    t.join()
    train.config(state=ACTIVE)
    

def train_model():
    def thread_done():
        if not CONFIG["stopped"] and CONFIG["training_thread"].is_alive():
            root.after(150, thread_done)
        else:
            if not CONFIG["stopped"]:
                predict.config(state=ACTIVE)
                editmenu.entryconfigure("Metrics", state=ACTIVE)
            CONFIG["stopped"] = True
            train.config(text="Start training")

    if CONFIG["stopped"]:
        CONFIG["stopped"] = False
        predict.config(state=DISABLED)
        editmenu.entryconfigure("Metrics", state=DISABLED)
        train.config(text="Stop training")
        
        t = Thread(target=CONFIG["clf"].adfa_train)
        CONFIG["training_thread"] = t
        root.after(1, thread_done)
        t.start()
    else:
        CONFIG["stopped"] = True


def predict():
    X = CONFIG.get("input_data", "").strip()
    if not X:
        messagebox.showerror("Error", "No input data provided")
    res = CONFIG["clf"].predict([X], predict_one=True)

    color = "red" if res else "green"
    prediction.config(text=PREDICTIONS[res], bg=color)


# Open TSPLIB file
def open_file():
    filename = filedialog.askopenfilename(
        initialdir="..\\tests", title="Select file", filetypes=(("ADFA-LD files", "*.txt"), ("all files", "*.*"))
    )
    if not filename:
        return
    with open(filename) as f:
        trace = f.read()

    try:
        convert = list(map(int, trace.split()))
    except Exception as e:
        messagebox.showerror("Error", "Bad input data")
        return

    CONFIG["input_data"] = trace
    result.delete('1.0', END)
    result.insert(INSERT, trace)

def translate(event):
    if CONFIG["translated"]:
        return translate_back()
    
    CONFIG["translated"] = True
    trace = CONFIG.get("input_data")
    if not trace:
        return
    
    translated = ", ".join(map(lambda x: translate_dict.get(x, "UNKNOWN"), trace.split()))
    result.delete('1.0', END)
    result.insert(INSERT, translated)
    result.config(state=DISABLED)
    

def translate_back():
    CONFIG["translated"] = False
    trace = CONFIG.get("input_data")
    if not trace:
        return

    result.config(state=NORMAL)
    result.delete('1.0', END)
    result.insert(INSERT, trace)
    
    
def settings():
    def click():
        try:
            random_state = int(rs.get())
            CONFIG["clf"].rs = random_state
            new.destroy()
        except:
            messagebox.showerror("Error", "Random state must be an integer")
            return
        
    rs = IntVar(value=CONFIG["clf"].rs)

    new = Toplevel()
    new.title("Settings")
    new.focus_force()
    new.resizable(width=False, height=False)
    new.grab_set()
    
    random_state_label = Label(new, text="Random state:")
    random_state = Entry(new, textvariable=rs)
    btn = Button(new, text = "Apply", command=click)
    
    random_state_label.pack()
    random_state.pack()
    btn.pack(pady=10, padx=50)

def metrics():
    binary_accuracy = CONFIG["clf"].metrics["binary_accuracy"]
    multilabel_accuracy = CONFIG["clf"].metrics["multilabel_accuracy"]
    tp, fp, fn, tn = CONFIG["clf"].metrics["confusion_matrix"]

    new = Toplevel()
    new.geometry("185x200")
    new.title("Metrics")
    new.focus_force()
    new.resizable(width=False, height=False)
    new.grab_set()

    Label(new, text="Binary classifier: ", font='Helvetica 12 bold').grid(row=0, column=0, sticky="W")
    Label(new, text=f"Accuracy: {binary_accuracy:.3f}", fg=get_color(binary_accuracy)).grid(row=1, column=0, sticky="W")
    conf_matrix = Frame(new, width=300, height=200, relief=SUNKEN)
    conf_matrix.grid(row=2, column=0, sticky="W")
    
    Label(conf_matrix, text=f"TP: {tp}", fg=get_color(tp/(tp+fn+1))).grid(row=2, column=0, sticky="W")
    Label(conf_matrix, text=f"FP: {fp}", fg=get_color(fp/(tp+fn+1), negative=True)).grid(row=2, column=1, padx=50)
    Label(conf_matrix, text=f"FN: {fn}", fg=get_color(fn/(tp+fn+1), negative=True)).grid(row=3, column=0)
    Label(conf_matrix, text=f"TN: {tn}", fg=get_color(tn/(fp+tn+1))).grid(row=3, column=1)

    Label(new, text="Multilabel classifier: ", font='Helvetica 12 bold').grid(row=4, column=0, sticky="W")
    Label(new, text=f"Accuracy: {multilabel_accuracy:.3f}", fg=get_color(multilabel_accuracy)).grid(row=5, column=0, sticky="W")
    Button(new, text="ROC-CURVES", command=CONFIG["clf"].draw_roc_curves, width=25).grid(row=6, column=0, pady=10)
    
# Main frame
root = Tk()
root.title("ADFA-LD Classifier")
root.geometry("300x130")
root.resizable(width=False, height=False)

#GUI Buttons
frame = Frame(width=249, height=300, bd=1, relief=SUNKEN)
frame.pack(padx=5, pady=5, anchor="center")

load = Button(frame, text = "Load train data", command=load_data)
load.grid(row=0, column=0)

train = Button(frame, text = "Train model", state=DISABLED, command=train_model)
train.grid(row=0, column=1)

predict = Button(frame, text = "Predict", state=DISABLED, command=predict)
predict.grid(row=0, column=2)
#

# Input data
frame = Frame(width=300, height=200, relief=SUNKEN)
frame.pack(side="left")

Label(frame, text = "Input:").grid(row=0, column=0, sticky="NW")
result = Text(frame, height=3, width=30)
result.grid(row=0, column=1)
#

ttk.Separator(frame, orient="horizontal").grid(row=1, column=0, pady=3)

# Result ouput
Label(frame, text="Result:").grid(row=2, column=0, sticky="W")
prediction = Label(frame, text="NO ATTACK", bg="green")
prediction.grid(row=2, column=1, sticky="W")
#

# MENU    
menubar = Menu(root)

# Create a pulldown menu, and add it to the menu bar
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Load trace", command=open_file)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.destroy)
menubar.add_cascade(label="File", menu=filemenu)

# Create more pulldown menus
editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label="Settings", command=settings)
editmenu.add_command(label="Metrics", command=metrics, state=DISABLED)
menubar.add_cascade(label="Model", menu=editmenu,)

# Display the menu
root.config(menu=menubar)
frame.bind("<Double-Button-1>", translate)
##
root.mainloop()
