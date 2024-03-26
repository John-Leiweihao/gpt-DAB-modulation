def recommend_modulation(weights):
    user_selected_indicators = list(weights.keys())
    # Step 1: Calculate the indicator values for each modulation mode
    modulations = {
        "SPS": {"Soft switching Range": 3, "Current Stress": 2, "Easiness to implement": 5},
        "DPS": {"Soft switching Range": 4, "Current Stress": 3, "Easiness to implement": 4},
        "EPS": {"Soft switching Range": 4, "Current Stress": 4, "Easiness to implement": 3},
        "TPS": {"Soft switching Range": 5, "Current Stress": 4, "Easiness to implement": 2},
        "Five-Degree": {"Soft switching Range": 5, "Current Stress": 5, "Easiness to implement": 1},
    }

    # Define the calculation formulas for additional indicators
    additional_indicators_formulas = {
        "Conduction loss": lambda x: 0.4 * x["Current Stress"] + 0.6 * x["Soft switching Range"],
        "Copper loss": lambda x: 0.4 * x["Current Stress"] + 0.6 * x["Soft switching Range"],
        "Core Loss": lambda x: 2.5 + 0.5 * x["Current Stress"],
        "Switch loss": lambda x: 1.5 + 0.7 * x["Soft switching Range"],
        "Efficiency": lambda x: 0.4 * x["Soft switching Range"] + 0.6 * x["Current Stress"],
        "Circulating current": lambda x: 0.5 * x["Current Stress"] + 0.5 * x["Soft switching Range"],
        "Reactive power": lambda x: 0.3 * x["Current Stress"] + 0.7 * x["Soft switching Range"],
        "Thermal performance": lambda x: 0.6 * x["Soft switching Range"] + 0.4 * x["Current Stress"],
       # "Implementation Cost": lambda x: 0.9 * x["Easiness to implement"] + 0.1 * x["Current Stress"],
        "Control complexity": lambda x: 0.9 * x["Easiness to implement"] + 0.1 * x["Soft switching Range"]
    }

    # Calculate the additional indicator values for each modulation mode
    for modulation, indicators in modulations.items():
        for indicator, formula in additional_indicators_formulas.items():
            indicators[indicator] = formula(indicators)

    # Step 2: User's selected indicators
    N = len(user_selected_indicators)
    K = len(modulations["SPS"]) - N

    # Step 3: Calculate the final score for each modulation mode
    for modulation, indicators in modulations.items():
        user_selected_score = sum(indicators[indicator] * weights[indicator] for indicator in user_selected_indicators)
        user_unselected_score = sum(indicators[indicator] for indicator in indicators if indicator not in user_selected_indicators)
        final_score = user_selected_score + (0.2 / K * user_unselected_score)
        modulations[modulation]["Final Score"] = final_score

    # Find the modulation mode with the highest final score
    best_modulation = max(modulations, key=lambda m: modulations[m]["Final Score"])

    return best_modulation

# Example usage
#best_modulation = recommend_modulation([ "Easiness to implement"])
#print(f"The recommended modulation is {best_modulation}" )
