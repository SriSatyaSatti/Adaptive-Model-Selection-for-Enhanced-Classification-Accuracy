from flask import Flask, render_template, request, redirect, url_for, flash, session
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import traceback

app = Flask(__name__)
app.secret_key = "satya"

# Dictionary to map classifiers to their corresponding functions
classifiers = {
    "Naive Bayes":GaussianNB(),
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Dictionary to map datasets to their corresponding load functions
datasets = {
    "Iris": load_iris,
    "Breast Cancer": load_breast_cancer,
    "Digits": load_digits,
    "Wine": load_wine
}


@app.route('/')
def home():
    return render_template("index.html", classifiers=classifiers.keys(), datasets=datasets.keys())


@app.route('/validate', methods=["POST"])
def validate():
    classifier_name = request.form['classifier']
    dataset_name = request.form['dataset']

    if classifier_name == "Use Multiple Classifiers":
        return render_template("multi_classifiers.html", dataset=dataset_name, classifiers=classifiers.keys())

    if classifier_name == "Logistic Regression" and dataset_name not in ["Breast Cancer"]:
        flash("Logistic Regression is suitable only for binary classification. Please choose Breast Cancer dataset.")
        return redirect(url_for('home'))

    return render_template("test_size.html", classifier=classifier_name, dataset=dataset_name)


@app.route('/train', methods=["POST"])
def train():
    classifier_name = request.form['classifier']
    dataset_name = request.form['dataset']
    test_size = float(request.form['test_size'])

    # Load dataset
    dataset = datasets[dataset_name]()
    x = dataset.data
    y = dataset.target

    # Split dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42)

    if classifier_name == "Use Multiple Classifiers":
        selected_classifiers = request.form.getlist('classifiers')
        results = []

        for idx, clf_name in enumerate(selected_classifiers, start=1):
            try:
                classifier = classifiers[clf_name]
                classifier.fit(x_train, y_train)
                y_pred = classifier.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                results.append((idx, clf_name, y_pred, accuracy))
            except Exception as e:
                traceback.print_exc()
                flash(f"Error occurred while training {clf_name}: {str(e)}")
                return redirect(url_for('home'))

        return render_template("multi_results.html", dataset=dataset_name, results=results)
    else:
        # Code for single classifier
        try:
            classifier = classifiers[classifier_name]
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)

            # Store the classifier name in session
            session['classifier_name'] = classifier_name
            session['dataset_name'] = dataset_name
            session['x_test'] = x_test.tolist()
            session['y_test'] = y_test.tolist()
            session['y_pred'] = y_pred.tolist()

            return render_template("result.html", classifier=classifier_name, dataset=dataset_name, y_pred=y_pred)
        except Exception as e:
            traceback.print_exc()
            flash(f"Error occurred while training {classifier_name}: {str(e)}")
            return redirect(url_for('home'))


@app.route('/accuracy', methods=["GET"])
def accuracy():
    try:
        y_test = np.array(session.get('y_test'))
        y_pred = np.array(session.get('y_pred'))
        accuracy = accuracy_score(y_test, y_pred)
        return render_template("accuracy.html", accuracy=accuracy)
    except Exception as e:
        flash(f"Error occurred: {str(e)}")
        return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)