# Summer-Training Project -- Covid-19 Data Analysis & Prediction PandeMic System

✨✨The objective of this capstone project is to analyze Covid -19 data, perform EDA and develop predictive models using Linear Regression and Random Forest Regressor to forecast Covid-19 cases among different countries/regions.
A complete exploratory and predictive analytics project focused on Covid-19 global data. This project leverages real datasets, performs in-depth visualizations, and builds regression models to forecast future Covid-19 cases across different countries.

---

## 📌 Project Overview

This project was developed during a capstone internship and summer training at Diginique Techlabs, IIT Roorkee. It includes:

- Data analysis of confirmed, recovered, and death cases
- Country-wise trend comparison
- Predictive modeling using:
  - Linear Regression
  - Random Forest Regressor
  - and many more ML models were implemented for training purpose

---

## 🧰 Technologies Used

- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Plotly
- Google Colab
- VS Code Editor

---

## 📦 Installation & Setup

Clone this repository:

```bash
git clone https://github.com/svedant2103/Diginique_TechlabsIITRoorkee-Covid-19-Data-Analysis-Project.git
cd Diginique_TechlabsIITRoorkee-Covid-19-Data-Analysis-Project


Install required Python libraries:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn plotly scikit-learn
📊 Dataset
Johns Hopkins University Covid-19 dataset


CSV files containing:

time_series_covid19_confirmed_global.csv

time_series_covid19_deaths_global.csv

time_series_covid19_recovered_global.csv

📈 Exploratory Data Analysis
✅ Key Steps:
python
Copy
Edit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

confirmed_df = pd.read_csv('time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('time_series_covid19_deaths_global.csv')
recovered_df = pd.read_csv('time_series_covid19_recovered_global.csv')


🔍 Visualizations:
1. Line plots showing growth trends

2. Heatmaps comparing cases across countries

3. Animated plots using Plotly

4. Bar plots for country-wise total cases



🌎 Example:
python
Copy
Edit
# Sum by country
global_confirmed = confirmed_df.groupby("Country/Region").sum().iloc[:, -1]
top_10 = global_confirmed.sort_values(ascending=False).head(10)
top_10.plot(kind='barh', color='skyblue')



🤖 Machine Learning Models

1️⃣ Linear Regression
python
Copy
Edit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = ... # days since start
y = ... # total confirmed cases
model = LinearRegression()
model.fit(X_train, y_train)


2️⃣ Random Forest Regressor
python
Copy
Edit
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


📊 Evaluation Metrics:
python
Copy
Edit
from sklearn.metrics import mean_squared_error, r2_score

print("R2 Score:", r2_score(y_test, predictions))
print("RMSE:", mean_squared_error(y_test, predictions, squared=False))


🔮 Predictions
1. Models trained on country-specific case data

2. Forecast future daily cases using learned trends

3. Graphs compare actual vs predicted values



📁 File Structure
bash
Copy
Edit
├── Pandemix_Covid_19Data_Analysis&Prediction.ipynb
├── Pandemix_Covid_19Data_Analysis&Prediction.py
├── README.md
├── /datasets
│   ├── time_series_covid19_confirmed_global.csv
│   ├── time_series_covid19_deaths_global.csv
│   └── time_series_covid19_recovered_global.csv
▶️ Run in Colab
Open the notebook in Colab:

🔗 Colab Notebook



🧠 Future Scope

1. Include vaccination datasets

2. Use Prophet for time series prediction

3. Add live Covid-19 APIs

4. Create a web dashboard using Streamlit



👨‍💻 Author
Vedant Singh
🎓 B.Tech IT, III Year – IIIT Sonepat
📧 svedant2103@gmail.com
🔗 LinkedIn



📄 License
This project was built during the Diginique Techlabs Internship (IIT Roorkee) and is open for academic and educational use only.
---

Would you like me to:
- Upload this `README.md` to your repo?
- Include a `requirements.txt` based on your code?
- Auto-generate plots as `.png` for offline use?
