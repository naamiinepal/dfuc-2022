import collections.abc
from typing import Callable, Optional, Sequence, Union
from monai.transforms import apply_transform

from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset

class Aug_UnAug_Dataset(_TorchDataset):
    """
    This is slightly modified implementation of monai's 'Dataset' class to retrurn both original and augmentated image on applying transformations
    """

    def __init__(self, data: Sequence, transform: Optional[Callable] = None, transform_aug: Optional[Callable] = None) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.

        """
        self.data = data
        self.transform = transform
        self.transform_aug = transform_aug

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i
    def _transform_load_only(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]
        return apply_transform(self.transform_aug, data_i)

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            return Subset(dataset=self, indices=index)
        return  self._transform_load_only(index), self._transform(index)

