from datasets import Dataset

def load_preference_data(data_path):
   """Load IMDB review pairs with clear preference examples"""
   
   # Example preference pairs
   preference_data = {
       "prompt": [
           "Write a movie review about pacing and storytelling",
           "Describe the film's character development",
           "Analyze the movie's emotional impact",
           "Discuss the film's technical aspects",
           "Review the movie's dialogue quality",
           "Evaluate the film's ending"
       ],
       "chosen": [
           "The film maintains perfect pacing throughout, with each scene building naturally to the next. The story unfolds organically without any rushed moments.",
           "Characters show remarkable depth and growth. The protagonist's journey from reluctant hero to confident leader feels earned through meaningful interactions.",
           "The movie masterfully builds emotional resonance through subtle moments and powerful performances, avoiding melodrama while hitting emotional peaks.",
           "Cinematography enhances storytelling with thoughtful framing and lighting. Each technical choice serves the narrative.",
           "Dialogue flows naturally and reveals character depth while advancing the plot. Every conversation feels purposeful and authentic.", 
           "The conclusion ties together all narrative threads satisfyingly while leaving just enough open for interpretation."
       ],
       "rejected": [
           "Movie was slow sometimes and fast others. Plot jumped around too much.",
           "Main character suddenly changes personality halfway through. Supporting cast feels flat.",
           "It's sad when it needs to be sad and happy when it needs to be happy. Pretty basic.",
           "They used lots of camera angles and effects. Some scenes were dark, others bright.",
           "People talked a lot. Some lines were good, others weren't so good.",
           "It just ends. Things happen and then it's over."
       ]
   }
   
   return Dataset.from_dict(preference_data)