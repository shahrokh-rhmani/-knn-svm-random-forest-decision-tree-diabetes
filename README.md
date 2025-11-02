# run project:
### 1. Create a virtual environment in project root not src/ (windows):
```
python -m venv venv
.\venv\Scripts\activate
```

### or
### Create a virtual environment in project root not src/ (linux)
```
python3 -m venv venv
source venv/bin/activate
```

### 2. Install requirements:
```
pip install -r requirements.txt
```

# 3. 
```
cd src
```
### windows:
### knn:
```
python .\knn-diabetes.py
```

### svm:
```
python .\svm-diabetes.py
```

### random-forest:
```
python .\random-forest-diabetes.py
```

### decision-tree-diabetes:
```
python .\decision-tree-diabetes.py
```
