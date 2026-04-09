import streamlit as st
from supabase import create_client

# COLE AQUI EXATAMENTE
SUPABASE_URL = "https://nyopjahyrdybnddehvky.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZxb2xwZ3lmY2tpYWtna29tdnpwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzU2NzgwNTcsImV4cCI6MjA5MTI1NDA1N30.wfsxnzcDS0RUwXTmsOPpG-oW8iUQofayj8U2rD0ZM-U"

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

st.set_page_config(page_title="Teste Login Supabase", layout="centered")
st.title("Teste de Login Supabase")

email = st.text_input("Email")
senha = st.text_input("Senha", type="password")

if st.button("Entrar"):
    try:
        resp = supabase.auth.sign_in_with_password({
            "email": email,
            "password": senha
        })
        st.success("Login realizado com sucesso")
        st.write(resp)
    except Exception as e:
        st.error(f"Erro real: {e}")