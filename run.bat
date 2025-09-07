@echo off
cd /d "c:/Users/sanat/OneDrive/Desktop/Project1_fall_sem25_26"
start "Streamlit" cmd /k "streamlit run frontend.py"
start "Main" cmd /k "python main.py"
