# Insurance Price Predictor

A application that predicts insurance premium costs based on user details such as age, gender, BMI, region, and smoking status.  
It uses a trained Machine Learning regression model to return an estimated insurance cost instantly.

🔗 **Live Demo:** [https://insurance-price-predictor-6ncz.onrender.com/](https://insurance-price-predictor-6ncz.onrender.com/)

---

## 📌 Description
This application allows users to enter basic health and demographic information and get an estimated insurance premium.  
The prediction is powered by a pre-trained ML model deployed alongside the backend API.  
MongoDB stores user data and request logs, while React provides a responsive frontend interface.

---

## ⚙ How It Works
1. **User Input:** The user fills out a form with age, gender, BMI, number of children, smoking status, and region.
2. **API Request:** The frontend sends this data via a POST request to the Express.js backend.
3. **Prediction:** The backend loads a trained ML model (`model.pkl`) and uses the input to predict the insurance charge.
4. **Response:** The API sends the prediction back to the frontend.
5. **Display:** The frontend shows the result instantly to the user.

---

## ✨ Features
- User-friendly form for data entry
- Real-time prediction
- MongoDB integration for storing data/logs
- ML model integration for accurate cost estimation
- Deployed on Render (backend & frontend together)

---

## 🛠 Tech Stack
- **Frontend:** React
- **Backend:** Node.js, Express.js
- **Database:** MongoDB
- **ML Model:** Trained regression model (`model.pkl`)
- **Deployment:** Render.com

---

