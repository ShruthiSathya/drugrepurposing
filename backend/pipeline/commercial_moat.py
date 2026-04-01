class CommercialMoat:
    def evaluate_ip_potential(self, combo_result):
        # If drugs are from completely different therapeutic areas but show 
        # high synergy, it is "Highly Non-Obvious" (Strong Patent Case)
        is_novel_pairing = combo_result['mechanism_a'] != combo_result['mechanism_b']
        
        # Dosage sparing is a "New Chemical Entity" (NCE) equivalent for IP
        has_dosage_sparing = combo_result.get("dosage_sparing") > 0.4
        
        moat_score = 0.0
        if is_novel_pairing: moat_score += 0.4
        if has_dosage_sparing: moat_score += 0.6 # Pharma loves safety wins
        
        return {
            "patent_strength": moat_score,
            "strategic_value": "ACQUISITION_TARGET" if moat_score > 0.8 else "NICHED"
        }