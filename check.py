import numpy as np

a = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
print("Conjugate of a: ")
print(str(np.conjugate(a)))
print("Transpose of a: ")
print(str(np.transpose(a)))
print("Hermitian of a: ")
print(str(np.conjugate(np.transpose(a))))
print("Product of a and its Hermitian: ")
print(str(np.dot(a, np.conjugate(np.transpose(a)))))
