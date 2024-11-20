from creative_tools import CreativeSpaceFinder

def explore_creative_space():
    # Initialize creative tools
    finder = CreativeSpaceFinder()
    
    # Define seed concepts for a creative project
    seed_concepts = [
        "sustainability", "technology", "nature", "innovation",
        "tradition", "progress", "harmony", "disruption",
        "organic", "artificial", "balance", "transformation",
        "simplicity", "complexity", "evolution", "stability"
    ]
    
    # Initialize the creative space
    finder.initialize_creative_space(seed_concepts)
    
    # Example 1: Find conceptual opposites
    print("\nExploring conceptual opposites of 'sustainability':")
    opposites = finder.find_conceptual_opposites("sustainability")
    for concept, distance in opposites:
        print(f"- {concept} (distance: {distance:.2f})")
    
    # Example 2: Find creative midpoints
    print("\nExploring the space between 'technology' and 'nature':")
    midpoints = finder.find_creative_midpoint("technology", "nature")
    print("Bridging concepts:")
    for concept, distance in midpoints:
        print(f"- {concept} (distance from midpoint: {distance:.2f})")
    
    # Example 3: Explore the broader creative space
    print("\nExploring the creative space between 'technology' and 'nature':")
    space_concepts = finder.find_midpoint_space("technology", "nature")
    print("Concepts in the creative space:")
    for concept, dist1, dist2 in space_concepts:
        print(f"- {concept} (distance to technology: {dist1:.2f}, distance to nature: {dist2:.2f})")
    
    # Example 4: Complete space analysis
    print("\nComplete analysis of the space between 'technology' and 'nature':")
    analysis = finder.analyze_creative_space("technology", "nature")
    
    print("\nDirect midpoints:")
    for concept, distance in analysis['midpoints']:
        print(f"- {concept} (distance: {distance:.2f})")
    
    print("\nSpace concepts:")
    for concept, dist1, dist2 in analysis['space_concepts']:
        print(f"- {concept} (distances: {dist1:.2f}, {dist2:.2f})")
    
    if analysis['contrasts']:
        print("\nContrasting concepts:")
        for concept in analysis['contrasts']:
            print(f"- {concept}")
    
    # Example 5: Find creative tensions
    print("\nExploring creative tensions around 'harmony':")
    tensions = finder.find_creative_tension("harmony")
    for concept1, concept2, distance in tensions:
        print(f"- Tension between '{concept1}' and '{concept2}' (distance: {distance:.2f})")
    
    # Example 6: Generate a creative prompt
    print("\nGenerating creative prompt for 'transformation':")
    prompt = finder.generate_creative_prompt("transformation")
    print(prompt)

if __name__ == "__main__":
    explore_creative_space()