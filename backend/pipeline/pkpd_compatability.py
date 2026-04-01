class PKPDCompatibility:
    def assess_alignment(self, drug_a_props, drug_b_props):
        t12_a = drug_a_props.get("half_life", 8)
        t12_b = drug_b_props.get("half_life", 8)
        
        # Ratio of half-lives: 1.0 is perfect, > 3.0 is a "Compliance Nightmare"
        ratio = max(t12_a, t12_b) / min(t12_a, t12_b)
        
        # Check for Transporter Competition (P-gp)
        # If both are substrates, they compete for entry (Bad for tumors/brain)
        both_pgp = drug_a_props.get("pgp_substrate") and drug_b_props.get("pgp_substrate")
        
        return {
            "half_life_ratio": round(ratio, 2),
            "fdc_feasibility": "HIGH" if ratio < 2.0 else "LOW",
            "transporter_clash": both_pgp,
            "pk_score": 1.0 / ratio # Penalty for misalignment
        }