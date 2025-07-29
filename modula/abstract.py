import jax
import copy

class Module:
    def __init__(self):
        self.children = []

        self.atoms = None           # number of atoms: int
        self.bonds = None           # number of bonds: int
        self.smooth = None          # is this module smooth?: bool
        self.sensitivity = None     # input Lipschitz estimate: float > 0
        self.mass = None            # proportional contribution of module toward feature learning of any supermodule: float >= 0

    def __str__(self):
        string = self.__class__.__name__
        string += f"\n...consists of {self.atoms} atoms and {self.bonds} bonds"
        string += f"\n...{'smooth' if self.smooth else 'non-smooth'}"
        string += f"\n...input sensitivity is {self.sensitivity}"
        string += f"\n...contributes proportion {self.mass} to feature learning of any supermodule"
        return string

    def tare(self, absolute=1.0, relative=None):
        if relative is None:
            self.tare(relative = absolute / self.mass)
        else:
            self.mass *= relative
            for m in self.children:
                m.tare(relative = relative)

    def jit(self):
        self.forward = jax.jit(self.forward)
        self.project = jax.jit(self.project)
        self.dualize = jax.jit(self.dualize)

    def forward(self, x, w):
        # Input and weight list --> output
        raise NotImplementedError

    def initialize(self, key):
        # Return a weight list.
        raise NotImplementedError

    def project(self, w):
        # Return a weight list.
        raise NotImplementedError

    def dualize(self, grad_w, target_norm):
        # Weight gradient list and number --> normalized weight gradient list
        raise NotImplementedError

    def __matmul__(self, other):
        if isinstance(other, tuple):
            other = TupleModule(other)
        return CompositeModule(self, other)

    def __add__(self, other):
        return Add() @ TupleModule((self, other))

    def __mul__(self, other):
        assert other != 0, "cannot multiply a module by zero"
        return self @ Mul(other)

    def __rmul__(self, scalar):
        return Mul(scalar) @ self

    def __pow__(self, n):
        assert n >= 0 and n % 1 == 0, "nonnegative integer powers only"
        return copy.deepcopy(self) @ (self ** (n-1)) if n > 0 else Identity()

    def __call__(self, x, w):
        return self.forward(x, w)

class Atom(Module):
    def __init__(self):
        super().__init__()
        self.atoms = 1
        self.bonds = 0

class Bond(Module):
    def __init__(self):
        super().__init__()
        self.atoms = 0
        self.bonds = 1
        self.mass = 0

    def initialize(self, key):
        return []

    def project(self, w):
        return []

    def dualize(self, grad_w, target_norm=1.0):
        return []

class CompositeModule(Module):
    def __init__(self, m1, m0):
        super().__init__()
        self.children = (m0, m1)

        self.atoms       = m0.atoms + m1.atoms
        self.bonds       = m0.bonds + m1.bonds
        self.smooth      = m0.smooth and m1.smooth
        self.mass        = m0.mass + m1.mass
        self.sensitivity = m0.sensitivity * m1.sensitivity

    def forward(self, x, w):
        m0, m1 = self.children
        w0 = w[:m0.atoms]
        w1 = w[m0.atoms:]
        x0 = m0.forward(x, w0)
        x1 = m1.forward(x0, w1)
        return x1

    def initialize(self, key):
        m0, m1 = self.children
        key, subkey = jax.random.split(key)
        return m0.initialize(key) + m1.initialize(subkey)

    def project(self, w):
        m0, m1 = self.children
        w0 = w[:m0.atoms]
        w1 = w[m0.atoms:]
        return m0.project(w0) + m1.project(w1)

    def dualize(self, grad_w, target_norm=1.0):
        if self.mass > 0:
            m0, m1 = self.children
            grad_w0, grad_w1 = grad_w[:m0.atoms], grad_w[m0.atoms:]
            d_w0 = m0.dualize(grad_w0, target_norm = target_norm * m0.mass / self.mass / m1.sensitivity)
            d_w1 = m1.dualize(grad_w1, target_norm = target_norm * m1.mass / self.mass)
            d_w = d_w0 + d_w1
        else:
            d_w = [0 * grad_weight for grad_weight in grad_w]
        return d_w

class TupleModule(Module):
    def __init__(self, python_tuple_of_modules):
        super().__init__()
        self.children = python_tuple_of_modules
        self.atoms       = sum(m.atoms       for m in self.children)
        self.bonds       = sum(m.bonds       for m in self.children)
        self.smooth      = all(m.smooth      for m in self.children)
        self.mass        = sum(m.mass        for m in self.children)
        self.sensitivity = sum(m.sensitivity for m in self.children)

    def forward(self, x, w):
        output_list = []
        for m in self.children:
            output = m.forward(x, w[:m.atoms])
            output_list.append(output)
            w = w[m.atoms:]
        return output_list

    def initialize(self, key):
        w = []
        for m in self.children:
            key, subkey = jax.random.split(key)
            w += m.initialize(subkey)
        return w

    def project(self, w):
        projected_w = []
        for m in self.children:
            projected_w_m = m.project(w[:m.atoms])
            projected_w += projected_w_m
            w = w[m.atoms:]
        return projected_w

    def dualize(self, grad_w, target_norm=1.0):
        if self.mass > 0:
            d_w = []
            for m in self.children:
                grad_w_m = grad_w[:m.atoms]
                d_w_m = m.dualize(grad_w_m, target_norm = target_norm * m.mass / self.mass)
                d_w += d_w_m
                grad_w = grad_w[m.atoms:]
        else:
            d_w = [0 * grad_weight for grad_weight in grad_w]
        return d_w

class Identity(Bond):
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1

    def forward(self, x, w):
        return x

class Add(Bond):
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1

    def forward(self, x, w):
        return sum(x)

class Mul(Bond):
    def __init__(self, scalar):
        super().__init__()
        self.smooth = True
        self.sensitivity = scalar

    def forward(self, x, w):
        return x * self.sensitivity

def get_leaf_modules(module):
    """
    Walk through a module tree and return the leaf modules (Atom or Bond instances)
    in the same order as the corresponding weights would be in the list returned by initialize().
    
    Args:
        module: A Module instance (typically CompositeModule at top level)
        
    Returns:
        List of leaf modules (Atom or Bond instances)
    """
    # Base case: if this is a leaf module (Atom or Bond)
    if isinstance(module, (Atom, Bond)):
        return [module]
    
    # If this is a CompositeModule
    elif isinstance(module, CompositeModule):
        m0, m1 = module.children
        # Order matches initialize(): m0 weights first, then m1 weights
        return get_leaf_modules(m0) + get_leaf_modules(m1)
    
    # If this is a TupleModule
    elif isinstance(module, TupleModule):
        leaf_modules = []
        # Order matches initialize(): iterate through children in order
        for child in module.children:
            leaf_modules.extend(get_leaf_modules(child))
        return leaf_modules
    
    # For any other Module type, assume no children or handle as needed
    else:
        return []

def get_leaf_target_norms(module, target_norm=1.0):
    """
    Walk through a module tree the same way dualize() does and compute the target norm
    that would be passed to each leaf module's dualize method, then store it in the module.
    
    Args:
        module: A Module instance
        target_norm: The target norm passed to this module's dualize method
        
    Returns:
        List of target norms for leaf modules, in the same order as get_leaf_modules()
    """
    # Base case: if this is a leaf module (Atom or Bond)
    if isinstance(module, (Atom, Bond)):
        return [target_norm]
    
    # If this is a CompositeModule
    elif isinstance(module, CompositeModule):
        if module.mass > 0:
            m0, m1 = module.children
            # Same logic as in CompositeModule.dualize()
            m0.target_norm = target_norm * m0.mass / module.mass / m1.sensitivity
            m1.target_norm = target_norm * m1.mass / module.mass
            
            # Recursively get target norms for children (order: m0 first, then m1)
            return (get_leaf_target_norms(m0, m0.target_norm) + 
                    get_leaf_target_norms(m1, m1.target_norm))
        else:
            # When mass is 0, we still need to traverse to get the structure right
            m0, m1 = module.children
            return (get_leaf_target_norms(m0, 0.0) + 
                    get_leaf_target_norms(m1, 0.0))
    
    # If this is a TupleModule
    elif isinstance(module, TupleModule):
        if module.mass > 0:
            target_norms = []
            # Same logic as in TupleModule.dualize()
            for child in module.children:
                child.target_norm = target_norm * child.mass / module.mass
                target_norms.extend(get_leaf_target_norms(child, child.target_norm))
            return target_norms
        else:
            # When mass is 0, we still need to traverse to get the structure right
            target_norms = []
            for child in module.children:
                target_norms.extend(get_leaf_target_norms(child, 0.0))
            return target_norms
    
    # For any other Module type, assume no children
    else:
        return []

def traverse_forward_order(module, func):
    """
    Traverse composite modules in the order that forward() will execute them,
    applying func to each module.
    
    Args:
        module: The module to traverse
        func: Function to apply to each module. Should accept a module as argument.
    """
    def _traverse(mod):
        if isinstance(mod, CompositeModule):
            # For composite modules, traverse m0 first, then m1 (execution order)
            m0, m1 = mod.children
            _traverse(m0)
            _traverse(m1)
        elif isinstance(mod, TupleModule):
            # For tuple modules, traverse all children (they execute in parallel)
            for child in mod.children:
                _traverse(child)
        
        # Apply function to current module after traversing children
        func(mod)
    
    _traverse(module)

def set_atomic_weights(module, weights):
    """
    Set weights as attributes on atomic modules by traversing in forward execution order. Assumes each atomic module has only one weight.
    
    Args:
        module: The root module to traverse
        weights: List of weights corresponding to atomic modules
    """
    weight_index = [0]  # Use list to make it mutable in closure
    
    def assign_weight(mod):
        if isinstance(mod, Atom):
            if weight_index[0] < len(weights):
                mod.weight = weights[weight_index[0]]
                weight_index[0] += 1
    
    traverse_forward_order(module, assign_weight)
    
    # Validate that we used all weights
    if weight_index[0] != len(weights):
        raise ValueError(f"Number of weights ({len(weights)}) doesn't match number of atomic modules ({weight_index[0]})")

