import pandas as pd
import os
import json


class PesticideRecommender:
    def __init__(self):
        self.pesticide_db = None
        self.load_pesticide_database()

    def load_pesticide_database(self):
        """Load and process pesticide database from Pestopia dataset"""

        pestopia_path = "data/pestopia"
        csv_files = []

        if os.path.exists(pestopia_path):
            for root, dirs, files in os.walk(pestopia_path):
                for file in files:
                    if file.endswith(".csv"):
                        csv_files.append(os.path.join(root, file))

        if csv_files:
            # Load the first CSV file found
            try:
                self.pesticide_db = pd.read_csv(csv_files[0])
                print(f"✅ Loaded pesticide database from: {csv_files[0]}")
                print(f"Columns: {list(self.pesticide_db.columns)}")
            except Exception as e:
                print(f"⚠️ Error loading database: {e}")
                self.create_default_database()
        else:
            print("⚠️ No CSV file found, creating default database")
            self.create_default_database()

    def create_default_database(self):
        """Create a default pesticide database if none found"""

        default_data = {
            "pest_name": [
                "aphid", "aphid", "aphid",
                "thrips", "thrips", "thrips",
                "whitefly", "whitefly", "whitefly",
                "caterpillar", "caterpillar", "caterpillar",
                "beetle", "beetle", "beetle",
                "mite", "mite", "mite",
                "leafhopper", "leafhopper", "leafhopper",
                "scale", "scale", "scale",
                "borer", "borer", "borer",
                "weevil", "weevil", "weevil",
                "armyworm", "armyworm", "bollworm", "bollworm"
            ],
            "pesticide_name": [
                "Imidacloprid", "Thiamethoxam", "Acetamiprid",
                "Spinosad", "Abamectin", "Fipronil",
                "Spiromesifen", "Pyriproxyfen", "Buprofezin",
                "Bacillus thuringiensis", "Chlorantraniliprole", "Indoxacarb",
                "Carbaryl", "Permethrin", "Lambda-cyhalothrin",
                "Abamectin", "Hexythiazox", "Propargite",
                "Imidacloprid", "Deltamethrin", "Cypermethrin",
                "Spirotetramat", "Buprofezin", "Dinotefuran",
                "Chlorantraniliprole", "Fipronil", "Cartap hydrochloride",
                "Imidacloprid", "Chlorpyrifos", "Thiamethoxam",
                "Spinetoram", "Emamectin benzoate", "Flubendiamide", "Chlorantraniliprole"
            ],
            "application_rate": [
                "0.5ml/L", "0.3g/L", "0.6ml/L",
                "0.45ml/L", "1.9ml/L", "2ml/L",
                "1.5ml/L", "1ml/L", "2ml/L",
                "2g/L", "0.6ml/L", "2ml/L",
                "2g/L", "2ml/L", "1ml/L",
                "1.9ml/L", "1.5ml/L", "2ml/L",
                "0.5ml/L", "1ml/L", "1ml/L",
                "1ml/L", "2ml/L", "0.6ml/L",
                "0.6ml/L", "2ml/L", "2g/L",
                "0.5ml/L", "2ml/L", "0.3g/L",
                "0.45ml/L", "0.5g/L", "2.5ml/L", "0.6ml/L"
            ],
            "effectiveness": [
                "High", "High", "Medium",
                "High", "Medium", "High",
                "High", "Medium", "Medium",
                "High", "High", "High",
                "Medium", "High", "High",
                "High", "Medium", "Medium",
                "High", "High", "High",
                "Medium", "Medium", "High",
                "High", "High", "Medium",
                "High", "Medium", "High",
                "High", "High", "High", "High"
            ]
        }

        self.pesticide_db = pd.DataFrame(default_data)

        os.makedirs("data", exist_ok=True)
        self.pesticide_db.to_csv("data/default_pesticide_db.csv", index=False)

        print("✅ Created default pesticide database")
        print(f"Shape: {self.pesticide_db.shape}")

    def get_recommendations(self, pest_name):
        """Get pesticide recommendations for a specific pest"""

        if self.pesticide_db is None:
            return []

        pest_name_clean = pest_name.lower().strip()

        # Handle both possible column names
        pest_col = None
        pesticide_col = None
        for col in self.pesticide_db.columns:
            if col.lower() in ["pest name", "pest_name"]:
                pest_col = col
            if col.lower() in ["most commonly used pesticides", "pesticide_name"]:
                pesticide_col = col
        if pest_col is None or pesticide_col is None:
            print("❌ Could not find required columns in pesticide database.")
            return []

        # Direct match
        matches = self.pesticide_db[
            self.pesticide_db[pest_col].str.lower().str.contains(pest_name_clean, na=False)
        ]

        # If no direct matches, try partial
        if matches.empty:
            for pest in self.pesticide_db[pest_col].unique():
                if pest_name_clean in str(pest).lower() or str(pest).lower() in pest_name_clean:
                    matches = self.pesticide_db[self.pesticide_db[pest_col] == pest]
                    break

        # Convert to list of dicts, handling both CSV and default DB columns
        recommendations = []
        for _, row in matches.iterrows():
            rec = {}
            # Always include pest name and pesticide info
            rec["pest_name"] = row[pest_col]
            rec["pesticide"] = row[pesticide_col]
            # Optionally include extra info if present
            if "application_rate" in row:
                rec["application_rate"] = row["application_rate"]
            if "effectiveness" in row:
                rec["effectiveness"] = row["effectiveness"]
            recommendations.append(rec)
        return recommendations

    def get_all_pests(self):
        """Get list of all pests in database"""
        if self.pesticide_db is not None:
            return self.pesticide_db["pest_name"].unique().tolist()
        return []


# 🔹 Test block
if __name__ == "__main__":
    recommender = PesticideRecommender()
    print("\n🧪 Testing pesticide recommendations:")

    test_pests = ["aphid", "thrips", "caterpillar"]
    for pest in test_pests:
        recs = recommender.get_recommendations(pest)
        print(f"\n{pest.upper()}:")
        for rec in recs:
            # Print all available info for each recommendation
            info = []
            for k, v in rec.items():
                info.append(f"{k}: {v}")
            print("  - " + ", ".join(info))
