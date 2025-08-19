from scape.scape_t.core import scape_t

texts = ["The individualistic yet universalist nature of existentialist analysis paradoxically betrays the inherent contradictions within this mode of thought.",
         "Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making.",
         "In the beginning of years, when the world was so new and all, and the Animals were just beginning to work for Man, there was a Camel, and he lived in the middle of a Howling Desert because he did not want to work"]

scores = scape_t(texts)  # returns {text: score}
for t, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
    print(f"SENTENCE: {t}\nSCORE: {s:7.3f}")
