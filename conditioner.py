import os
import re
import torch
from sd1_unconditional_hack import get_sd1_unconditional

# TODO 
# should maybe save vectors as safetensors?
# pre-created concepts could potentially have n vectors, so each concept could have n vectors, default to 1
# would need some way of encoding positional information if used with SD3 or Flux, though cross-attention blocks of unets would seemingly not need it
# might need to support training an unconditional vector for SD1, whereas other models presumably all use zero unconditional (could also just cache the existing one)
# might want to pad out empty spaces with the unconditional tokens, and maybe add the BOS (which might in theory encode something like a pooled representation of the image when prompts are encoded)
# creating concepts after requires_grad is set would cause them to not be trainable

class Conditioner:
    def __init__(self, vector_dim=768, device='cpu', dtype=torch.float32):
        self.vector_dim = vector_dim
        self.conditioning_vectors: Dict[str, torch.Tensor] = {}
        self.device = device
        self.dtype = dtype
        self.unconditional = list(get_sd1_unconditional().to(device, dtype).unbind(0)) # turn the unconditional tensor with shape (77, 768) into a list with 77 tensor elements with shape (768)
    
    def _create_vector(self, mean=0.0, std=1e-6):
        vec = torch.randn(self.vector_dim) * std + mean
        return vec.to(device=self.device, dtype=self.dtype)

    def get_vector(self, concept):
        if concept not in self.conditioning_vectors:
            self.conditioning_vectors[concept] = self._create_vector()
        return self.conditioning_vectors[concept]
    
    # pre-create vectors which match multiple words in a prompt, such as 'John Smith', 'Space Command Uniform', etc (might add option for number of vectors with each concept, default to 1)
    def create_concepts(self, concepts):
        for concept in concepts:
            self.get_vector(concept)
    
    #def encode(self, prompts, pad_to_same_length=True):
    # prompts should never contain more than 75 concepts due to reliance on CLIP BOS and EOS
    # though perhaps length doesn't matter, and the encodings could just be expanded to match the longest, using repeats of the final unconditional where the unconditional will expand with repeats of the final tokens
    def encode(self, prompts):
        encodings = []
        longest_encoding = 0
        
        # sort concepts by number of spaces to match longer combinations of words first
        sorted_concepts = sorted(self.conditioning_vectors.keys(), key=lambda k: len(k.split()), reverse=True)
        
        for prompt in prompts:
            if prompt.strip() == '':
                #encoding = list(unconditional.to(device=self.device, dtype=self.dtype).unbind(0)) # turn the unconditional tensor with shape (77, 768) into a list with 77 tensor elements with shape (768)
                encoding = self.unconditional 
            else:
                '''# prompt will be split into a series of (text, is_processed) segments, where each 'text' is either a known existing conditioning key, or a word which does not yet have a conditioning vector
                text_segments = [ (prompt, False) ]
                
                finished = False
                while not finished:
                    finished = True
                    
                    for index, (segment, is_processed) in enumerate(text_segments):
                        if not is_processed:
                            del text_segments[index]
                            
                            for concept in sorted_concepts:
                                concept_pattern = r'\b' + re.escape(concept) + r'\b'
                                match = re.search(concept_pattern, segment)
                                
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
                
                encoding = []
                for segment, _ in text_segments:
                   encoding.append(self.get_vector(segment))'''
                
                # the above method is way too slow, for now just split on commas since that's how concept strings are being determined
                encoding = []
                encoding.append(self.unconditional[0]) # BOS
                concepts = [concept.strip() for concept in prompt.split(',')]
                for concept in concepts:
                    vector = self.get_vector(concept)
                    encoding.append(vector)
                
                if len(encoding) < len(self.unconditional):
                    encoding.extend(self.unconditional[len(encoding):len(self.unconditional)])
                #encoding.append(self.unconditional[-1]) # an EOS, seems useful to just have somewhere for some of the attention to go to mellow things?
                
            encodings.append(encoding)
            
            #if len(encoding) > longest_encoding:
            #    longest_encoding = len(encoding)
        
        # unsure if this is needed for a batch to work correctly
        '''if pad_to_same_length:
            for encoding in encodings:
                while len(encoding) < longest_encoding:
                    encoding.append(torch.zeros(self.vector_dim))'''
        '''if pad_to_length:
            for encoding in encodings:
                while len(encoding) < pad_to_length:
                    #encoding.append(torch.zeros(self.vector_dim).to(device=self.device, dtype=self.dtype))
                    encoding.append(self.unconditional[len(encoding)])
                if len(encoding) > pad_to_length:
                    del encoding[pad_to_length:]'''
        
        # convert list of lists to a tensor of shape (num_prompts, longest_encoding, vector_dim)
        batch_tensor = torch.stack([torch.stack(encoding) for encoding in encodings])
        
        return batch_tensor
    
    # concepts is optional filter list, and all other vectors will be set to opposite setting
    def set_requires_grad(self, requires_grad=True, concepts=None):
        for concept, vector in self.conditioning_vectors.items():
            match = True if (concepts is None) else (concept in concepts)
            vector.requires_grad_( requires_grad if match else (not requires_grad) )

    def get_parameters(self, only_requires_grad=False):
        if only_requires_grad:
            return [vector for vector in self.conditioning_vectors.values() if vector.requires_grad]
        else:
            return self.conditioning_vectors.values()
    
    # probably not the right way to do this
    def to(self, device, dtype):
        for concept, vector in self.conditioning_vectors.items():
            new_vector = vector.detach().to(device=device, dtype=dtype).requires_grad_(vector.requires_grad)
            self.conditioning_vectors[concept] = new_vector
        
        self.device = device
        self.dtype = dtype
        self.unconditional = [ x.to(device, dtype) for x in self.unconditional ]
    
    # potentially add argument to only save those with requires_grad
    def save(self, directory, step_count=None):
        #if step_count:
        #    directory = os.path.join(directory, f"step_{step_count}")
        os.makedirs(directory, exist_ok=True)

        for concept, vector in self.conditioning_vectors.items():
            file_name = os.path.join(directory, f"{concept}.pt")
            torch.save(vector, file_name)

    def load(self, directory):
        for file_name in os.listdir(directory):
            if file_name.endswith('.pt'):
                concept = file_name[:-3]  # strip the '.pt' extension to get the concept name
                file_path = os.path.join(directory, file_name)
                vector = torch.load(file_path).to(device=self.device, dtype=self.dtype)
                self.conditioning_vectors[concept] = vector
