import pandas as pd
import re

class SmartBhaiGPT:
    def __init__(self, responses_file="responses.csv"):
        # Load response templates
        try:
            self.responses = pd.read_csv(responses_file)
        except FileNotFoundError:
            raise FileNotFoundError("Bhai, responses.csv nahi mila! Check kar project folder mein.")
    
    def fetch_app_data(self, context_needed):
        """
        Fetch real-time data (e.g., IV, gamma) from your VolGuard Pro app.
        Replace this with your actual data pipeline (e.g., generate_features or 5paisa API).
        """
        try:
            # Placeholder: Replace with your actual data fetching logic
            # Example: from your_app_module import generate_features
            # features = generate_features()
            # data = {
            #     "iv": features.get("iv", 30.0),
            #     "gamma": features.get("gamma", 0.05),
            #     "margin": features.get("margin_usage", 85.0),
            #     "delta": features.get("delta", 0.4),
            #     "vix": features.get("vix", 25.0)
            # }
            
            # Mock data for testing
            data = {
                "iv": 30.0,  # Replace with real IV from your app
                "gamma": 0.05,
                "margin": 85.0,
                "delta": 0.4,
                "vix": 25.0
            }
        except Exception as e:
            # Fallback if data fetch fails
            print(f"Error fetching data: {e}")
            data = {"iv": "N/A", "gamma": "N/A", "margin": "N/A", "delta": "N/A", "vix": "N/A"}
        
        # Return only the needed context
        return {key: data.get(key, "N/A") for key in context_needed.split(",")}
    
    def generate_response(self, user_query):
        """
        Match user query to a response template and fill with app data.
        """
        user_query = user_query.lower().strip()
        
        # Find matching response
        for _, row in self.responses.iterrows():
            pattern = row["query_pattern"]
            if re.search(pattern, user_query):
                # Fetch required context (e.g., IV, gamma)
                context = self.fetch_app_data(row["context_needed"])
                
                # Fill response template
                try:
                    response = row["response_template"].format(**context)
                    return response
                except KeyError:
                    return "Bhai, data thoda off lag raha hai. Try again! Do your own research!"
        
        # Fallback response for unmatched queries
        return "Bhai, yeh query thodi alag hai. Kya bol raha hai, thoda clearly bata? 😜 Do your own research!"

# Test the class
if __name__ == "__main__":
    gpt = SmartBhaiGPT()
    test_queries = [
        "What is IV?",
        "Check my straddle at 21000",
        "Should I hedge?",
        "Random query"
    ]
    for query in test_queries:
        print(f"Query: {query}")
        print(f"Response: {gpt.generate_response(query)}\n")
