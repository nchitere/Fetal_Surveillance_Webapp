import streamlit as st
from page1 import show_page as page1
from page2 import show_page as page2


# Create a sidebar with navigation links
page = st.sidebar.selectbox("Choose a page", ["Page 1", "Page 2"])

# Define the content for each page
if page == "Page 1":
    st.title("The working for the Machine Learnin Project")
    st.write("The Fetal Monitoring Project.")

elif page == "Page 1":
    page1()  # Call the function from page1.py








