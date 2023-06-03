from lightly.data import MultiViewDataModule, MultiViewCollateFunction
from PIL import Image

class CustomMultiViewCollateFunction(MultiViewCollateFunction):
    def __call__(self, batch):
        batch = list(filter(lambda x: isinstance(x[0], Image.Image), batch))
        if not batch:
            return super().__call__(batch)
        
        views = [[] for _ in range(len(batch[0][0]))]
        for sample in batch:
            for i, view in enumerate(sample[0]):
                views[i].append(view)
        
        batch = (views, batch[0][1])
        return super().__call__(batch)


