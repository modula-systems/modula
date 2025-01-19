class Error:

    def value(self, output, target):
        raise NotImplementedError

    def grad(self, output, target):
        raise NotImplementedError

    def __call__(self, output, target):
        return self.value(output, target)

class SquareError(Error):

    def __init__(self):
        super().__init__()

    def value(self, output, target):
        return ((output-target) ** 2).mean()

    def grad(self, output, target):
        return 2 * (output - target) / output.size

class XentError(Error):

    def __init__(self):
        super().__init__()

    def value(self, output, target):
        raise NotImplementedError

    def grad(self, output, target):
        raise NotImplementedError
