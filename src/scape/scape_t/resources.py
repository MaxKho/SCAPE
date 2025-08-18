"""
resources.py
------------

Centralised resources for SCAPE-T, including text normalisation patterns, 
default hyperparameters, and the abstractness dataset.

This module defines built‑in resources (e.g., abbreviation expansions) and
exposes them alongside common preprocessing constants.

Exposed constants:
- ABBREVIATIONS : list of (pattern, replacement) regex pairs, loaded from Abbreviations.json
- DELIMITERS    : set of sentence boundary characters
- STOPWORDS     : set of semantically uninformative function words to ignore in analysis
- PUNCT         : translation table for stripping punctuation (excluding apostrophes)
- DEFAULT_PARAMS: dict of default SCAPE-T hyperparameters

Search order for JSON data:
1. SCAPE_DATA_DIR environment variable (explicit override)
2. Packaged data inside scape.scape_t/data (if installed with package data)
3. Project-root/src/data (when working in src/ layout)
4. Project-root/data (legacy layout without src/)
5. Current working directory and cwd/data (for Colab or ad-hoc runs)

Usage example:
    from scape.scape_t.resources import ABBREVIATIONS, DEFAULT_PARAMS
    print(ABBREVIATIONS[:3])
    print(DEFAULT_PARAMS["alpha"])
"""

import string

ABBREVIATIONS = [["&c\\s*\\.","and so forth "],
                 ["&c\\s*\\.\\s*([A-Z])", "and so forth . \\1"],
                 ["([.,;:?!])[.,;:?!]+", "\\1"],
                 ["Dr\\.", "Doctor "],
                 ["Jr\\.", "Junior "],
                 ["Mr\\.", "Mr "],
                 ["Mrs\\.", "Mrs "],
                 ["Ms\\.", "Ms "],
                 ["Prof\\.", "Professor "],
                 ["Sr\\.", "Senior "],
                 ["St\\.", "Saint "],
                 ["\\b([ivxIVX]+)\\.", "\\1 "],
                 ["\\bP\\.", "Page "],
                 ["\\bVol\\.", "volume "],
                 [ "\\bcf\\.", "cf "],
                 ["\\bch\\.", "chapter "],
                 ["\\bchap\\.", "chapter "],
                 ["\\be\\.\\s*g\\.", "for example "],
                 ["\\bed\\.", "edition "],
                 ["\\bedd\\.", "editions "],
                 ["\\bi\\.\\s*e\\.", "that is, "],
                 ["\\bl\\.", "line "],
                 ["\\bp\\.", "page "],
                 ["\\bpp\\.", "pages "],
                 ["\\bpref\\.", "preface "],
                 ["\\bviz\\.", "namely "],
                 ["\\bvol\\.", "volume "],
                 ["etc\\s*\\.", "and so forth"],
                 ["etc\\s*\\.\\s*([A-Z])", "and so forth . \\1"]]
DELIMITERS = {".", "!", "?"}
STOPWORDS = {"a", "an", "the", "that", "and", "very", "will", "just"}
PUNCT = str.maketrans("", "", string.punctuation.replace("'", ""))
DEFAULT_PARAMS = {"alpha": 0.35, "beta": 3.25, "gamma": 0.4, "delta": 10.0,
                  "zeta": 0.55, "kappa": 0.1, "eta": 1.0, "tau": 0.9,
                  "mu": 6.75, "sigma": 1.3}
ABSTRACTNESS = {
    # Class 1: Direct References to Physical Objects, Actions, or Properties (Balanced Variance in Length)
    "Rock": 1, "Waterfall": 1, "Fire": 1, "Tree": 1, "Wind": 1, "Mountain": 1, "River delta": 1, "Sand": 1, "Metal": 1, "Smoke": 1,
    "Cloud": 1, "Thunder": 1, "Snow": 1, "Skin": 1, "Eyeball": 1, "Tongue": 1, "Hand": 1, "Joint": 1, "Femur bone": 1, "Bone marrow": 1,
    "Pebble": 1, "Grass blade": 1, "Leaf": 1, "Branch": 1, "Flower": 1, "Seedling": 1, "Feather": 1, "Fur": 1, "Snake": 1, "Talon": 1,
    "Tooth": 1, "Beak": 1, "Shell": 1, "Wave": 1, "Sunlight": 1, "Shadow": 1, "Echo": 1, "Sound": 1, "Scent trail": 1,
    "Flavor": 1, "Laughter": 1, "Crying": 1, "Sweat": 1, "Blood": 1, "Tear": 1, "Pulse": 1, "Breath": 1,
    "Whisper": 1, "Sneeze": 1, "Cough": 1, "Yawn": 1, "Blink": 1, "Step": 1, "Jump": 1, "Stride": 1,
    "Walk": 1, "Kick": 1, "Clap": 1, "Hug": 1, "Smile": 1, "Frown": 1, "Bite": 1, "Chew": 1,
    "Swallow": 1, "Echo sound": 1, "Laughter sound": 1, "Crying noise": 1, "Sweat drop": 1, "Blood vessel": 1,
    "Whisper tone": 1, "Sneeze burst": 1,

    # Class 2: Concepts with Direct Physical Embodiment (Balanced Variance in Length)
    "House": 2, "Bridge": 2, "Road": 2, "Lamp": 2, "Door": 2, "Chair": 2, "Table": 2, "School building": 2,
    "Campus": 2, "Hospital ward": 2, "Cathedral": 2, "Prison": 2, "Factory floor": 2, "City": 2,
    "Nation": 2, "Coin": 2, "Money": 2, "Book": 2, "Painting": 2, "Statue": 2, "Computer": 2,
    "Phone": 2, "Letter": 2, "Ticket": 2, "Passport": 2, "Flag": 2, "Uniform": 2, "Observation tower": 2,
    "Library": 2, "Museum": 2, "Market": 2, "Warehouse": 2, "Ship": 2, "Airplane": 2, "Train": 2,
    "Subway": 2, "Bus": 2, "Car": 2, "Motorcycle": 2, "Helmet": 2, "Backpack": 2, "Tent": 2, "Farm": 2,
    "Barn": 2, "Windmill": 2, "Fountain": 2, "Theater": 2, "Cinema": 2, "Restaurant": 2,
    "Cargo ship": 2, "Passenger airplane": 2, "Electric train": 2, "Metro station": 2, "City bus": 2, "Sports car": 2,
    "Racing motorcycle": 2, "Safety helmet": 2, "Camping backpack": 2, "Canvas tent": 2,

    # Class 3: Concepts Indirectly Embodied in the Physical World or Defined by Physical Processes (Balanced Variance in Length)
    "Empire": 3, "Rivalry": 3, "Battle": 3, "Commerce": 3, "Coinage": 3, "Market": 3, "Government": 3,
    "Law": 3, "Election": 3, "Dictatorship": 3, "Economy": 3, "Industry": 3, "Farming": 3,
    "Diplomacy": 3, "Education": 3, "Crime": 3, "Taxation": 3, "Innovation": 3, "Broadcasting": 3,
    "Migration": 3, "Exploration": 3, "Bureaucracy": 3, "Transport": 3, "Tradition": 3, "Revolution": 3,
    "Colonial rule": 3, "Monarchy": 3, "Republic": 3, "Infrastructure": 3, "Piracy": 3, "Security": 3,
    "Abolition": 3, "Feudalism": 3, "Discovery": 3, "Urbanization": 3, "Corporate strategy": 3,
    "Manufacturing": 3, "Market crash": 3, "Workers' strike": 3, "Public protest": 3, "Corruption scandal": 3,
    "Engineering patent": 3, "News broadcast": 3, "Human relocation": 3, "Space travel": 3, "Civic administration": 3,
    "Shipping logistics": 3, "Family tradition": 3, "Civil uprising": 3, "Colonial rule": 3, "Royal monarchy": 3,
    "Federal republic": 3,

    # Class 4: Abstract Concepts with No Clear Relationship to the Physical World (Balanced Variance in Length)
    "Justice": 4, "Morality": 4, "Free will": 4, "Truth": 4, "Knowledge": 4, "Wisdom": 4,
    "Love": 4, "Honor": 4, "Beauty": 4, "Happiness": 4, "Suffering": 4, "Thought": 4,
    "Idea": 4, "Concept": 4, "Ethics": 4, "Logic": 4, "Mathematics": 4, "Philosophy": 4,
    "Theory": 4, "Inspiration": 4, "Consciousness": 4, "Intuition": 4, "Creativity": 4,
    "Intelligence": 4, "Fate": 4, "Faith": 4, "Enlightenment": 4, "Destiny": 4, "Virtue": 4,
    "Sin": 4, "Guilt": 4, "Hope": 4, "Regret": 4, "Metaphor": 4, "Symbolism": 4,
    "Imagination": 4, "Ego": 4, "Nostalgia": 4, "Despair": 4, "Spirituality": 4,
    "Aesthetic": 4, "Charisma": 4, "Idealism": 4, "Ambition": 4, "Introspection": 4,
    "Philosophical idea": 4, "Logical reasoning": 4, "Moral dilemma": 4, "Symbolic gesture": 4,
    "Mathematical proof": 4, "Transcendent thought": 4, "Scientific theory": 4,
    "Artistic inspiration": 4, "Mental awareness": 4, "Intuitive feeling": 4, "Creative impulse": 4
}
