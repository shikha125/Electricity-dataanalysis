
# ⚡ Electricity Data Analysis and Visualization (Django Project)

This project is a web-based application for analyzing electricity consumption data using **Python**, **Django**, and **data visualization libraries**. Users can upload CSV datasets, clean the data, visualize trends, and apply basic machine learning algorithms — all through an interactive web interface.

---

## 🚀 Features

- ✅ Upload electricity consumption datasets (CSV format)
- ✅ Data cleaning and preprocessing
- ✅ Visualize data with interactive charts (Matplotlib / Seaborn)
- ✅ Analyze consumption patterns over time
- ✅ Perform basic algorithmic analysis (e.g., correlation, mean/max/min usage)
- ✅ User-friendly interface with dropdowns and controls

---

## 📁 Project Structure

```
electricity/
│
├── manage.py
├── electricity/              # Main Django project
│   ├── settings.py
│   └── urls.py
│
├── app/                      # Your core analysis app
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── templates/
│   └── static/
│
├── media/                    # Uploaded datasets
└── templates/                # HTML UI templates
```

---

## ⚙️ Tech Stack

- 🐍 Python 3.x
- 🌐 Django 4.x
- 📊 Matplotlib / Seaborn
- 📂 Pandas / NumPy
- 🌍 HTML, CSS, Bootstrap
- ☁️ SQLite / (or MySQL for production)

---

## 📈 Algorithms & Data Analysis

- Descriptive statistics (mean, median, standard deviation)
- Correlation analysis between different electricity usage variables
- Peak demand time detection
- Data cleaning (missing value handling, outlier removal)

---

## 🧑‍💻 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/shikha125/Electricity-dataanalysis.git
   cd Electricity-dataanalysis
   ```

2. **Create virtual environment & install dependencies**
   ```bash
   python -m venv env
   source env/bin/activate    # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the server**
   ```bash
   python manage.py runserver
   ```

4. **Visit**: [http://localhost:8000](http://localhost:8000)

---
5. Screenshots:
   ![Screenshot 2025-06-09 233411](https://github.com/user-attachments/assets/b89cc774-e0b9-46a8-b35b-2eb4894ed7c7)


## 🙋‍♀️ Author

**Shikha**  
📧 [shikhashah1103@gmail.com]
