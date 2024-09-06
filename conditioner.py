import os
import re
import torch

# TODO 
# encode should maybe take 'prompts', and return a num_prompts * concept * 768 tensor (would need padding to ensure all are same length?)
# should maybe save vectors as safetensors?
# pre-created concepts could potentially have n vectors, so each concept could have n vectors, default to 1
# would need some way of encoding positional information if used with SD3 or Flux, though cross-attention blocks of unets would seemingly not need it

class Conditioner:
    def __init__(self, vector_dim=768):
        self.vector_dim = vector_dim
        self.conditioning_vectors: Dict[str, torch.Tensor] = {}
    
    def _create_vector(self, mean=0.0, std=1e-6):
        return torch.randn(self.vector_dim) * std + mean

    def get_vector(self, concept):
        if concept not in self.conditioning_vectors:
            self.conditioning_vectors[concept] = self._create_vector()
        return self.conditioning_vectors[concept]
    
    # pre-create vectors which match multiple words in a prompt, such as 'John Smith', 'Space Command Uniform', etc (might add option for number of vectors with each concept, default to 1)
    def create_concepts(self, concepts):
        for concept in concepts:
            self.get_vector(concept)
    
    def encode(self, prompt):
        # sort concepts by number of spaces to match longer combinations of words first
        sorted_concepts = sorted(self.conditioning_vectors.keys(), key=lambda k: len(k.split()), reverse=True)
        
        # prompt will be split into a series of (text, is_processed) segments, where each 'text' is either a known existing conditioning key, or a word which does not yet have a conditioning vector
        text_segments = [ (prompt, False) ]
        
        finished = False
        while not finished:
            finished = True
            
            for index, (segment, is_processed) in enumerate(text_segments):
                if not is_processed:
                    del text_segments[index]
                    
                    for concept in sorted_concepts:
                        key_pattern = r'\b' + re.escape(concept) + r'\b'
                        match = re.search(key_pattern, segment)
                        
                        if match:
                            before = segment[:match.start()].strip()
                            after = segment[match.end():].strip()
                            
                            if before:
                                text_segments.insert(index, (before, False))
                                index += 1
                            
                            text_segments.insert(index, (concept, True))
                            index += 1
                            
                            if after:
                                text_segments.insert(index, (after, False))
                            
                            finished = False
                            break
                    
                    # if finished is still true, there were no existing concept matches in the segment, so split the segment into individual words, which will each get a new blank vector
                    if finished:
                        words = re.findall(r'\w+(?:[-_]\w+)*|\S', segment)
                        for word in words:
                            text_segments.insert(index, (word, True))
                            index += 1
                    
                    finished = False
                    break
        
        vectors = []
        for segment, _ in text_segments:
           vectors.append(self.get_vector(segment))
        
        return vectors
    
    def save(self, directory, step_count=None):
        if step_count:
            directory = os.path.join(directory, f"step_{step_count}")
        os.makedirs(directory, exist_ok=True)

        for concept, vector in self.conditioning_vectors.items():
            file_name = os.path.join(directory, f"{concept}.pt")
            torch.save(vector, file_name)

    def load(self, directory):
        for file_name in os.listdir(directory):
            if file_name.endswith('.pt'):
                concept = file_name[:-3]  # strip the '.pt' extension to get the concept name
                file_path = os.path.join(directory, file_name)
                vector = torch.load(file_path)
                self.conditioning_vectors[concept] = vector
    
    # concepts list is optional filter, and all other vectors will be set to opposite setting
    def set_requires_grad(self, requires_grad=True, concepts=None):
        for concept, vector in self.conditioning_vectors.items():
            match = (concept in concepts) if (concepts is not None) else True
            vector.requires_grad = requires_grad if match else not requires_grad

    def get_parameters(self, only_requires_grad=False):
        if only_requires_grad:
            return [vector for vector in self.conditioning_vectors.values() if vector.requires_grad]
        else:
            return self.conditioning_vectors.values()
