from PIL import Image
import numpy as np
from numpy import asarray


image = Image.open('dog.jpg').convert('L')

data = asarray(image)
print(type(data))
# summarize shape
m, n = data.shape

U, s, V = np.linalg.svd(data, full_matrices=True)

print(s.shape)

for r in [5, 20, 100]:
    data_approx = np.dot(U[:, :r], np.dot(np.diag(s[:r]), V[:r, :]))
    compressed_image = Image.fromarray(data_approx)
    compressed_image.show()
