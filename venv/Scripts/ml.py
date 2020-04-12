from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, plot_confusion_matrix
from sklearn.linear_model import *
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


PREDICTIONS = {
    0: "NO ATTACK",
    1: "ADD SECOND ADMIN",
    2: "HYDRA FTP BRUTEFORCE",
    3: "HYDRA SSH BRUTEFORCE",
    4: "JAVA METERPRETER",
    5: "METERPRETER",
    6: "WEB_SHELL"
}


class MyClassifier():
    def __init__(self, rs=42, logging=False):
        self.binary_classifier = LogisticRegression(penalty="l2", random_state=rs, max_iter=3000)
        self.attack_classifier = LogisticRegression(penalty="l2", random_state=rs, max_iter=2000)
        self.attack_vector = None
        self.scaler = None
        self.rs = rs
        
        self.metrics = {}
        self.logging = logging
        
    def load_data(self, callback=None):
        self.train_data = pd.read_csv("..\\train_data\\train_data.csv")
        self.validation_data = pd.read_csv("..\\train_data\\validation_data.csv")
        self.attack_data = self.train_data[self.train_data.iloc[:, 2:].sum(axis=1) == 1]

        if callback:
            callback()

    def get_X_y(self, df):
        trace = df["trace"].apply(lambda x: list(map(int, x.split())))
        X = self.transform_X(trace)
        y = np.array(df.iloc[:, 2:])
        return X, y

    def get_attack_X_y(self, df):
        traces = df["trace"].apply(lambda x: x.split())
        X = self.transform_attack(traces)
        y = np.array(df.iloc[:, 2:])
        return X, y

    def transform_X(self, traces):
        X = []
        for arr in traces:
            temp = [0] * 340
            for i in arr:
                if i > 340:
                    continue
                temp[i-1] += 1
            X.append(temp)
        return np.array(X)
    
    def prepare_vector(self, X):
        d = {}
        features = set()
        ind = 0
        for arr in X:
            for size in range(2, 6):
                for i in range(0, len(arr) - size):
                    sub = arr[i: i+size]
                    key = "-".join(sub)
                    if key in features:
                        if key not in d:
                            d[key] = ind
                            ind += 1
                    else:
                        features.add(key)                               
        return d

    def transform_attack(self, X):
        res = []
        for arr in X:
            temp = [0]*len(self.attack_vector) + [350]
            for size in range(2, 6):
                for i in range(0, len(arr) - size):
                    sub = arr[i: i+size]
                    key = "-".join(map(str, sub))
                    if key in self.attack_vector:
                        temp[self.attack_vector[key]] += 1
            temp = np.array(temp, dtype="float64")
            res.append(temp)
        
        return np.array(res)
        
    def adfa_train(self, callback=None):
        self.binary_train()
        self.attack_train()
        if callback:
            callback()

    def attack_train(self):
        if self.logging:
            print("\nAttack training in progress")
        traces = self.attack_data["trace"].apply(lambda x: x.split())
        self.attack_vector = self.prepare_vector(traces)

        X, y = self.get_attack_X_y(self.attack_data)
        X_train, X_test, y_train, y_test = [], [], [], []
        for i in range(6):
            ind = (y.argmax(axis=1) == i)
            arr_x = X[ind]
            arr_y = y[ind]
            np.random.shuffle(arr_x)
            X_train += list(arr_x[::2])
            y_train += list(arr_y[::2])
            X_test += list(arr_x[1::2])
            y_test += list(arr_y[1::2])

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        self.metrics["y_test"] = y_test.T
        y_train = y_train.argmax(axis=1)
        y_test = y_test.argmax(axis=1)
        
        self.attack_classifier.fit(X_train, y_train)

        self.metrics["probas"] = self.attack_classifier.predict_proba(X_test).T
        self.metrics["map_x"] = X_test
        self.metrics["map_y"] = y_test
        
        y_pred = self.attack_classifier.predict(X_test)
        if self.logging:
            print("\nAttack_classifier accuracy:", accuracy_score(y_test, y_pred))

        self.metrics["multilabel_accuracy"] = accuracy_score(y_test, y_pred)

    def draw_map(self):
        disp = plot_confusion_matrix(self.attack_classifier, self.metrics["map_x"],
                                     self.metrics["map_y"], cmap=plt.cm.Blues, normalize=None)
        plt.show()
        
    def draw_roc_curves(self):
        for i in range(6):
            lr_fpr, lr_tpr, _ = roc_curve(self.metrics["y_test"][i], self.metrics["probas"][i])
            plt.plot(lr_fpr, lr_tpr, marker='.', label=PREDICTIONS[i+1])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.show()
        
    def binary_train(self):
        if self.logging:
            print("Binary training in progress")
            
        X, y = self.get_X_y(self.train_data)
        y = y.sum(axis=1)
        X_attack, y_attack = self.get_X_y(self.attack_data)
        X_val, y_val = self.get_X_y(self.validation_data)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=self.rs)
        self.binary_classifier.fit(X_train, y_train)

        pred_val = self.binary_classifier.predict(X_val)
        pred_a = self.binary_classifier.predict(X_attack)
        y_pred = self.binary_classifier.predict(X_test)
        
        pred_val = pred_val
        pred_a = pred_a
        y_pred = y_pred
        
        if self.logging:
            print("Binary overall test accuracy:", accuracy_score(y_test, y_pred))
            print("Binary: attack only:", accuracy_score([1 for i in range(len(pred_a))], pred_a))
            print("Binary: validation only:", accuracy_score([0 for i in range(len(pred_val))], pred_val))

        self.metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).ravel()
        self.metrics["binary_accuracy"] = accuracy_score(y_test, y_pred)


    def predict(self, X, predict_one=False):
        if isinstance(X, str):
            X = np.array([list(map(int, X.split()))])
            
        X_bin = self.transform_X(X)
        bp = self.binary_predict(X_bin)
        
        if predict_one and not bp[0]:
            if self.logging:
                print("No attack")
            return 0

        X_atk = self.transform_attack(X)
        attack_predict = self.attack_predict(X_atk) + 1

        if predict_one:
            return attack_predict[0]
        return attack_predict

    def binary_predict(self, X):
        return self.binary_classifier.predict(X)

    def attack_predict(self, X):
        return self.attack_classifier.predict(X)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("No tests passed!")
        exit()

    mc = MyClassifier()
    mc.load_data()
    print("Training model")
    mc.adfa_train()
    print("Training complete")
    
    for test in sys.argv[1:]:
        try:
            print(f"Loading {test}...")
            with open(test) as fs:
                trace = fs.read().strip()

            pred = PREDICTIONS.get(mc.predict(trace, predict_one=True), "-")
            print("VERDICT:", pred)
        except Exception as e:
            print(e)
            print()
    input()


                
