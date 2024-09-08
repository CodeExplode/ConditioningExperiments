import os
import re
import torch

from conditioner_network import ConditioningAdjustmentNetwork
from sd1_unconditional_hack import get_sd1_unconditional

# TODO 
# should maybe save vectors as safetensors?
# pre-created concepts could potentially have n vectors, so each concept could have n vectors, default to 1
# would need some way of encoding positional information if used with SD3 or Flux, though cross-attention blocks of unets would seemingly not need it
# might need to support training an unconditional vector for SD1, whereas other models presumably all use zero unconditional (could also just cache the existing one)
# might want to pad out empty spaces with the unconditional tokens, and maybe add the BOS (which might in theory encode something like a pooled representation of the image when prompts are encoded)
# creating concepts after requires_grad is set would cause them to not be trainable, though that's fine since the optimizer wouldn't include them either, so need to pre-create
# should probably aim to keep vector norms around the same as original conditioning, maybe use unconditional as basis, maybe ignoring bos

class ConditionerPerLayer:
    def __init__(self, vector_dim=768, num_layers=7, device='cpu', dtype=torch.float32):
        self.vector_dim = vector_dim
        self.num_layers = num_layers
        self.conditioning_vectors: Dict[str, torch.Tensor] = {}
        self.device = device
        self.dtype = dtype
        self.unconditional = list(get_sd1_unconditional().to(device, dtype).unbind(0)) # turn the unconditional tensor with shape (77, 768) into a list with 77 tensor elements with shape (768)
        self.network = ConditioningAdjustmentNetwork(vector_dim, vector_dim).to(device=device, dtype=dtype)
    
    def _create_vectors(self, mean=0.0, std=1e-6):
        vecs = torch.randn(self.num_layers, self.vector_dim) * std + mean
        return vecs.to(device=self.device, dtype=self.dtype)

    def get_vectors(self, concept):
        if concept not in self.conditioning_vectors:
            self.conditioning_vectors[concept] = self._create_vectors()
        return self.conditioning_vectors[concept]
    
    def create_concepts(self, concepts):
        for concept in concepts:
            if len(concept.strip()) == 0:
                print(f'tried to create blank concept')
                continue
            self.get_vectors(concept)
    
    def encode(self, prompts):
        layer_encodings = []
        vectors = []
        
        for layer in range(self.num_layers):
            encodings = []
            masks = [] # positions of bos/eos vectors from unconditional inserted into the conditional, will be altered by a small network to fit the concept vectors
            
            for prompt in prompts:
                if prompt.strip() == '':
                    encoding = self.unconditional
                    mask = [0] * (len(self.unconditional)) # not training the unconditional
                else:               
                    # for now just split on commas since that's how concept strings are being determined
                    encoding = []
                    mask = []
                    
                    encoding.append(self.unconditional[0]) # BOS
                    mask.append(1) # BOS vector is alterable
                    
                    concepts = [concept.strip() for concept in prompt.split(',')]
                    for concept in concepts:
                        if len(concept.strip()) == 0:
                            print(f'empty concept in: {prompt}')
                            continue
                        
                        vector = self.get_vectors(concept)[layer]
                        encoding.append(vector)
                        mask.append(0)
                        
                        # apparently can't check for tensor comparison this way?
                        #if vector not in vectors:
                        vectors.append(vector)
                    
                    padding_length = len(self.unconditional) - len(encoding)
                    if padding_length > 0:
                        encoding.extend(self.unconditional[len(encoding):len(self.unconditional)])
                        mask.extend([1] * padding_length) # EOS vectors are alterable
                        
                    #encoding.append(self.unconditional[-1]) # an EOS, seems useful to just have somewhere for some of the attention to go to mellow things?
                    
                encodings.append(encoding)
                masks.append(mask)
            
            # convert list of lists to a tensor of shape (num_prompts, len(self.unconditional), vector_dim)
            batched_encoding = torch.stack([torch.stack(encoding) for encoding in encodings]).to(device=self.device, dtype=self.dtype)
            batched_mask = torch.tensor(masks, device=self.device, dtype=torch.float32)
            
            # adjust the vectors in the bos and eos positions
            # TODO - should maybe have some indication of layer? the vectors themselves would be the indication
            batched_encoding = self.network(batched_encoding, batched_mask)
            
            del batched_mask
            
            layer_encodings.append(batched_encoding)
        
        return layer_encodings, vectors
    
    # concepts is optional filter list, and all other vectors will be set to opposite setting
    def set_requires_grad(self, requires_grad=True):
        for concept, vectors in self.conditioning_vectors.items():
            vectors.requires_grad_( requires_grad )
            
        for param in self.network.parameters():
            param.requires_grad = requires_grad

    def get_parameters(self):
        conditioning_parameters = self.conditioning_vectors.values()
        network_parameters = self.network.parameters()
        
        return list(conditioning_parameters) + list(network_parameters)
    
    # might not be the best way to do this
    def to(self, device, dtype):
        for concept, vectors in self.conditioning_vectors.items():
            new_vectors = vectors.detach().to(device=device, dtype=dtype).requires_grad_(vectors.requires_grad)
            self.conditioning_vectors[concept] = new_vectors
        
        self.unconditional = [ x.to(device, dtype) for x in self.unconditional ]
        self.network.to(device=device, dtype=dtype)
        
        self.device = device
        self.dtype = dtype
    
    '''# a hacky way to force the conditioning vectors to have a similar magnitude to conditioning from CLIP, called during training, would ideally be done as part of the loss function to allow the optimizer to aim for it
    # only applied if the magnitude is larger than target, to give near-zero init vectors a chance to grow in the right direction
    # this value seems very large and is potentially incorrect, from the final hidden states of CLIP_L passed through the normalization layer, for 9000 test prompts, ignoring the first BOS vector which is always inserted manually here (should maybe have encoded with no special tokens to ignore padding too)
    def enforce_conditioning_magnitudes(self, vectors=[], target_magnitude=27.685843130140224):
        with torch.no_grad():
            for vector in vectors:
                current_magnitude = torch.norm(vector)
                
                if current_magnitude > target_magnitude:
                    scaling_factor = target_magnitude / current_magnitude
                    vector.mul_(scaling_factor) # scale in place'''
    
    def save(self, directory, step_count=None):
        os.makedirs(directory, exist_ok=True)

        for concept, vectors in self.conditioning_vectors.items():
            file_name = os.path.join(directory, f"{concept}.pt")
            torch.save(vectors, file_name)
        
        network_path = os.path.join(directory, "network.pt")
        torch.save(self.network.state_dict(), network_path)

    def load(self, directory):
        for file_name in os.listdir(directory):
            if file_name.endswith('.pt'):
                concept = file_name[:-3]  # strip the '.pt' extension to get the concept name
                file_path = os.path.join(directory, file_name)
                vectors = torch.load(file_path).to(device=self.device, dtype=self.dtype)
                self.conditioning_vectors[concept] = vectors
        #self.enforce_conditioning_magnitudes(self.conditioning_vectors.values())
        
        network_path = os.path.join(directory, "network.pt")
        if os.path.exists(network_path):
            self.network.load_state_dict(torch.load(network_path, map_location=self.device))
        self.network.to(device=self.device, dtype=self.dtype)