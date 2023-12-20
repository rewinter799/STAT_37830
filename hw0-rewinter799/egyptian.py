"""
Egyptian algorithm
"""

def egyptian_multiplication(a, n):
    """
    Returns the product a * n.

    Assume n is a nonegative integer.
    """
    def isodd(n):
        """
        returns True if n is odd
        """
        return n & 0x1 == 1

    if n == 1:
        return a
    if n == 0:
        return 0

    if isodd(n):
        return egyptian_multiplication(a + a, n // 2) + a
    else:
        return egyptian_multiplication(a + a, n // 2)

if __name__ == '__main__':
    # this code runs when executed as a script
    for a in [1,2,3]:
        for n in [1,2,5,10]:
            print("{} * {} = {}".format(a, n, egyptian_multiplication(a,n)))


def power(a, n):
    """
    Computes the power a ** n.

    Argument a is any real number.
    Argument n is a nonegative integer
    """
    if n == 0:
        # Base Case: Anything to the 0th power is 1
        return 1
    if n == 1:
        # Base Case: Anything to the 1st power is itself
        return a
    
    if n % 2 == 1:
        # Odd n: take the square of a, with an extra multiplication at the end
        return power(a * a, n // 2) * a
    else:
        # Even n: take the square of a
        return power(a * a, n // 2)

print("\nExamples of the power function:")
print("3**3 = ", power(3, 3), sep = "")
print("4**4 = ", power(4, 4), sep = "")
print("5**3 = ", power(5, 3), sep = "")
