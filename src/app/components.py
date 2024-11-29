import streamlit as st
def github_button(url, logo_url="https://github.githubassets.com/favicons/favicon-dark.png"):
    button_style = f"""
        <style>
        .github-button {{
            display: inline-flex;
            align-items: center;
            padding: 2px;
            border: none;
            background-color: transparent;
            cursor: pointer;
            text-decoration: none;
            text-align: center;
        }}
        .github-button img {{
            height: 40px; /* Ajustez la taille de l'image ici */
        }}
        .github-button:hover {{
            background-color: #0E1117; /* Couleur de survol si désirée */
        }}
        </style>
    """
    button_html = f"""
        <a href="{url}" class="github-button" target="_blank">
            <img src="{logo_url}" alt="GitHub Logo"/>
        </a>
    """
    st.markdown(button_style, unsafe_allow_html=True)
    st.markdown(button_html, unsafe_allow_html=True)