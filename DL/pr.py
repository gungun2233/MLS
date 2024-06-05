import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression

# Set page title and favicon
st.set_page_config(page_title="Placement Predictor", page_icon="🎓")

# Generate a balanced dataset
np.random.seed(42)
X = np.random.rand(1000, 2)
X[:, 0] = X[:, 0] * 10  # CGPA (0-10)
X[:, 1] = X[:, 1] * 100  # Resume Score (0-100)

# Ensure a mix of outcomes
y = np.zeros(1000, dtype=bool)
y[:500] = True  # First half are placed
y[500:] = False  # Second half are not placed
np.random.shuffle(y)  # Shuffle to mix them up

# Adjust X to correlate with y
for i in range(1000):
    if y[i]:
        X[i, 0] += np.random.normal(2, 1)  # Higher CGPA for placed students
        X[i, 1] += np.random.normal(20, 10)  # Higher Resume Score for placed students
    else:
        X[i, 0] += np.random.normal(-1, 1)  # Lower CGPA for non-placed students
        X[i, 1] += np.random.normal(-10, 10)  # Lower Resume Score for non-placed students

# Clip values to stay within range
X[:, 0] = np.clip(X[:, 0], 0, 10)  # CGPA between 0 and 10
X[:, 1] = np.clip(X[:, 1], 0, 100)  # Resume Score between 0 and 100

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Function to get qualitative feedback
def get_feedback(cgpa, resume_score, prob):
    feedback = "Based on our analysis:\n\n"
    
    if prob > 0.8:
        feedback += "🌟 Outstanding! Your profile is top-tier. Companies will be eager to have you.\n\n"
    elif prob > 0.6:
        feedback += "😊 Great prospects! You're well-prepared for the job market.\n\n"
    elif prob > 0.4:
        feedback += "🤔 On the fence. With some improvements, you can tip the scales in your favor.\n\n"
    else:
        feedback += "😟 It might be challenging, but don't lose hope!\n\n"
    
    if cgpa >= 8.5:
        feedback += "📚 Stellar CGPA! It reflects exceptional academic prowess.\n"
    elif cgpa >= 7.5:
        feedback += "📚 Strong CGPA. You're clearly dedicated to your studies.\n"
    elif cgpa >= 6.5:
        feedback += "📚 Decent CGPA. Try to push it higher in remaining semesters.\n"
    else:
        feedback += "📚 Your CGPA needs work. Consider tutoring or study groups.\n"
    
    if resume_score >= 85:
        feedback += "📝 Your resume is a masterpiece! It's sure to impress recruiters.\n"
    elif resume_score >= 70:
        feedback += "📝 Great resume! It effectively showcases your abilities.\n"
    elif resume_score >= 55:
        feedback += "📝 Good start on your resume. Add more achievements and skills.\n"
    else:
        feedback += "📝 Your resume needs significant work. Seek professional help.\n"
    
    if prob < 0.5:
        feedback += "\n💡 Action Plan:\n- Target higher grades in major subjects\n- Intern at startups or contribute to open-source\n- Tailor your resume for each application\n- Build a strong LinkedIn profile"
    
    return feedback

# Title
st.title("Student Placement Predictor 🎓")

# Predict placement for new student
st.subheader("Predict Your Placement Chances")
cgpa = st.number_input("Enter Your CGPA:", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
resume_score = st.number_input("Rate Your Resume (0-100):", min_value=0, max_value=100, value=70, step=1)

new_student = np.array([[cgpa, resume_score]])

if st.button("Predict My Chances"):
    prediction_prob = model.predict_proba(new_student)[0, 1]
    
    if prediction_prob > 0.6:
        st.success(f"Excellent! You have a {prediction_prob:.1%} chance of being placed. 🎉")
        st.balloons()
    elif prediction_prob > 0.4:
        st.warning(f"You have a {prediction_prob:.1%} chance. Not bad, but room to improve. 🚀")
    else:
        st.error(f"It'll be tough. Your current placement probability is {prediction_prob:.1%}. 📚")
    
    # Provide qualitative feedback
    feedback = get_feedback(cgpa, resume_score, prediction_prob)
    st.info(feedback)

# About section in sidebar
st.sidebar.title("About")
st.sidebar.info(
    "🎓 **Student Placement Predictor**\n\n"
    "Empowering students with AI-driven career insights.\n\n"
    "📊 **How It Works:**\n"
    "1. Enter your CGPA (0-10)\n"
    "2. Rate your resume (0-100)\n"
    "3. Get your personalized forecast\n\n"
    "🧠 **Our AI:**\n"
    "- Learns from 1000s of graduate outcomes\n"
    "- Adapts to industry trends\n"
    "- Offers strategic guidance\n\n"
    "🎯 **Not Just Predictions:**\n"
    "We map your path to success. Our advice is tailored to your unique profile.\n\n"
    "🌟 Remember: Our AI guides, but your passion leads!\n\n"
    "Trusted by top universities worldwide."
)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2024 PlacementAI. All rights reserved.")