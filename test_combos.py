import asyncio
import json
from backend.pipeline.production_pipeline import ProductionPipeline

async def test_drug_combinations():
    # Initialize the main production pipeline
    # This will load your Scorer, SynergyEngine, and Filters
    pipeline = ProductionPipeline()
    
    disease = "pulmonary arterial hypertension"
    print(f"--- Testing Combo Logic for: {disease} ---")

    try:
        # generate_treatment_plan is the entry point for combo scoring
        # It utilizes combo_scorer.py and synergy_engine.py
        plan = await pipeline.generate_treatment_plan(
            disease_name=disease,
            max_regimens=5,        # Limit results for clear testing
            include_triples=True,  # Test both pairs and triplets
            fetch_ppi=True,
            use_tissue=True
        )

        # 1. Check if the Orange/Purple Book filters worked
        print(f"\n[Filter Check]")
        print(f"Total drugs in pool: {plan['header']['drug_pool']['final_pool']}")
        
        # 2. Verify the Ranked Regimens
        print(f"\n[Ranked Regimens (Top 3)]")
        for i, regimen in enumerate(plan["ranked_regimens"][:3], 1):
            regimen_name = regimen["regimen"]  # Use 'regimen' instead of 'drugs'
            orr = regimen.get("orr_estimate", "N/A")
            score = regimen.get("combo_score", 0)
            
            print(f"{i}. {regimen_name}")
            print(f"   - Predicted ORR: {orr:.1%}")
            print(f"   - Combined Score: {score:.4f}")
            
            # 3. Verify Synergy/Toxicity Engine integration
            if "synergy_notes" in regimen:
                print(f"   - Synergy Insight: {regimen['synergy_notes']}")

        # 4. Check the Biotech/Wet Lab briefs for consistency
        print(f"\n[Briefs Check]")
        print(f"Targets identified for validation: {len(plan['wet_lab_brief']['priority_targets'])}")

    except Exception as e:
        print(f"Error during combo testing: {e}")
    finally:
        await pipeline.close()

if __name__ == "__main__":
    asyncio.run(test_drug_combinations())