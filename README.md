#  Machine Learning Project

This is a simple machine learning project that uses TensorFlow.

---

##  Authors

- **Abdelrahman Mostafa Mohamed**
- **Ali Ahmed Elio Mahmoud**

---



##  Requirements

Before running the project, install the required Python packages.

###  1. Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
```

Activate the virtual environment:

- On **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- On **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

---

###  2. Install Dependencies

Create a file named `requirements.txt` in the project folder with the following content:

```text
tensorflow
```

Then run:

```bash
pip install -r requirements.txt
```

>  If you only need TensorFlow, you can also install it directly with:
> ```bash
> pip install tensorflow
> ```

---

###  3. Run the Project

Once the environment is ready and dependencies are installed, run the script:

```bash
python main.py
```

---

##  Troubleshooting

- If you see this error:
  ```
  ModuleNotFoundError: No module named 'tensorflow'
  ```
  → It means TensorFlow is not installed. Run:
  ```bash
  pip install tensorflow
  ```

- Use `python --version` to make sure you're using Python 3.8 or newer.

---

##  Project Structure (Example)

```
ML-Project/
├── main.py
├── requirements.txt
└── README.md
```
