from typing import Optional, Iterable, Any, Union
import random

from monai.data.dataloader import DataLoader

class joint_dataloader:
    def __init__(self, dataloaders: list[DataLoader], names: Optional[list[str]], truncate_to_minimum: bool = False, random_iter_choice: bool = True):
        """
        Joint dataloader that will iterate through different dataloaders.
        This class is intended for Modality Agnostic methods or cross-modality learning where theoretically learning from a all modalities back-to-back as opposed to
        modality dedicated sub-epochs may prove optimal.

        This is an interable class that will either randomly or iteratively go through each provided dataloader to return data from it.

        ---
        Input arguments:

        * dataloaders: `list[Dataloader]` The dataloaders to be iterated through
        * names: `Optional[list[str]]` The names of each dataloader (i.e. the modalities involved)
        * truncate_to_minimum: `bool`, default = `False` In the case where the dataloaders are not completely equal in length, an optional flag to truncate each iteration to the same length.
        * random_iter_choice: `bool`, default = `True` Randomly chose the next dataloader to pull from or not.    
        """
        # Initializing input arguments
        self.dataloaders = dataloaders
        self.names = names if names else [i for i in range(len(dataloaders))]
        assert len(dataloaders) == len(self.names)
        self.random_iter_choice = random_iter_choice

        # Gathering parameters
        self.num_sets: int = len(self.dataloaders)
        self.lengths: list[int] = [len(dl) for dl in self.dataloaders]
        if truncate_to_minimum:
            minimum_length = min(self.lengths)
            self.lengths: list[int] = [minimum_length for _ in range(self.num_sets)]
        self.idxs: list[int] = [i for i in range(self.num_sets)]
        self.iidxs: list[int] = [0 for _ in range(self.num_sets)]

    def __len__(self):
        return sum(self.lengths)

    def __iter__(self):
        # Reset parameters to be used when iterating through the dataloaders
        self.c_idxs = self.idxs.copy()
        self.iidxs = [0 for _ in range(self.num_sets)]
        self.iterable_dataloaders = [iter(dl) for dl in self.dataloaders]
        self.static_choice = 0

        return self
    
    def __next__(self):
        # Check to see if there are remaining ids to pull from:
        if not self.c_idxs:
            raise StopIteration
        
        # Grab a modality to pull
        if self.random_iter_choice:
            # Pick a random modality from the current selection
            id = random.choice(self.c_idxs)
        else:
            self.static_choice = (self.static_choice + 1) % len(self.c_idxs)
            id = self.c_idxs[self.static_choice]

        # Update the iteration of that selection
        self.iidxs[id] += 1

        # Check to see if this is the last element of that selection
        if self.iidxs[id] == self.lengths[id]:
            # remove it if there are no more pulls to take
            self.c_idxs.pop(self.c_idxs.index(id))
        
        # Get and return the associated information
        name = self.names[id]
        item = next(self.iterable_dataloaders[id])

        return name, item
