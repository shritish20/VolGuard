import streamlit as st
from py5paisa import FivePaisaClient

@st.cache_resource(show_spinner=False)
def load_credentials():
    return st.secrets["5paisa"]

def login_to_5paisa():
    creds = load_credentials()

    st.sidebar.subheader("🔐 5Paisa Secure Login")
    use_live = st.sidebar.checkbox("Enable Live 5Paisa Mode", value=False)

    if not use_live:
        st.sidebar.info("Live mode disabled.")
        return None

    totp = st.sidebar.text_input("Enter TOTP Code", type="password", max_chars=6)
    login_button = st.sidebar.button("Login to 5Paisa")

    if login_button and totp:
        try:
            client = FivePaisaClient(cred={
                "APP_NAME": creds["app_name"],
                "APP_SOURCE": creds["app_source"],
                "USER_ID": creds["user_id"],
                "PASSWORD": creds["password"],
                "USER_KEY": creds["user_key"],
                "ENCRYPTION_KEY": creds["encryption_key"]
            })

            # Attempt login
            client.get_totp_session(creds["client_code"], totp, creds["pin"])

            # Validate with a real data pull (Nifty spot)
            test_req = [{
                "Exch": "N",
                "ExchType": "C",
                "ScripCode": 999920000,
                "Symbol": "NIFTY"
            }]
            response = client.fetch_market_feed(test_req)

            if response and "Data" in response and response["Data"][0].get("LastRate"):
                st.sidebar.success("✅ Logged in to 5Paisa!")
                return client
            else:
                st.sidebar.error("❌ Login failed: Session invalid or expired.")
                return None

        except Exception as e:
            st.sidebar.error(f"Login error: {e}")
            return None

    elif login_button and not totp:
        st.sidebar.warning("Please enter your 6-digit TOTP.")
        return None

    return None
