import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        shifted = [x-max(z) for x in z]
        total=np.sum([np.exp(x) for x in shifted])
        return [round(np.exp(x)/total, 4) for x in shifted]


        