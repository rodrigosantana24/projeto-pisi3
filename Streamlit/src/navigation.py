def show_navigation():
    import streamlit as st

    st.sidebar.title("Navigation")
    options = ["Exploratory Analysis", "Classification", "Clustering"]
    choice = st.sidebar.radio("Select a tab:", options)

    return choice