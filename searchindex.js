Search.setIndex({"docnames": ["algorithms/manifold/hypersphere", "algorithms/manifold/index", "algorithms/manifold/orthogonal", "algorithms/newton-schulz", "bad-scaling", "examples/hello-mnist", "examples/hello-world", "examples/weight-erasure", "faq", "golden-rules", "history", "index", "intro/reading-list", "theory/atom/conv2d", "theory/atom/embed", "theory/atom/index", "theory/atom/linear", "theory/bond/index", "theory/bond/nonlinearities", "theory/compound/gpt", "theory/compound/index", "theory/module", "theory/vector"], "filenames": ["algorithms/manifold/hypersphere.rst", "algorithms/manifold/index.rst", "algorithms/manifold/orthogonal.rst", "algorithms/newton-schulz.rst", "bad-scaling.rst", "examples/hello-mnist.nblink", "examples/hello-world.nblink", "examples/weight-erasure.nblink", "faq.rst", "golden-rules.rst", "history.rst", "index.rst", "intro/reading-list.rst", "theory/atom/conv2d.rst", "theory/atom/embed.rst", "theory/atom/index.rst", "theory/atom/linear.rst", "theory/bond/index.rst", "theory/bond/nonlinearities.rst", "theory/compound/gpt.rst", "theory/compound/index.rst", "theory/module.rst", "theory/vector.rst"], "titles": ["Hypersphere", "Manifold duality maps", "Orthogonal manifold", "Newton-Schulz", "Bad scaling", "Hello, MNIST!", "Hello, World!", "Weight erasure", "Frequently asked questions", "Golden rules for scaling", "The science of scale", "Welcome to the Modula docs!", "Reading list", "Conv2d", "Embedding", "Atomic modules", "Linear", "Bond modules", "Nonlinearities", "GPT", "Compound modules", "Modules", "Vectors"], "terms": {"On": [0, 2, 3, 4, 6, 8, 9, 10, 12], "thi": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], "page": [0, 2, 3, 6, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], "we": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "work": [0, 1, 2, 3, 4, 9, 10], "out": [0, 1, 2, 3, 4, 8, 9, 10, 11], "an": [0, 2, 3, 5, 6, 7, 8, 9, 10, 12], "algorithm": [0, 1, 2, 3, 4, 8, 11, 12], "perform": [0, 2, 3, 4, 8, 9], "under": [0, 1, 2, 4, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], "euclidean": [0, 1, 8, 9], "norm": [0, 1, 2, 6, 7, 8, 9, 10, 12], "while": [0, 2, 3, 4, 8, 9], "mai": [0, 4, 8, 9, 10], "seem": [0, 8, 9, 10], "obviou": [0, 8], "i": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], "good": [0, 4, 8, 9], "warmup": 0, "technic": [0, 8], "scaffold": 0, "help": [0, 1, 11], "more": [0, 2, 3, 6, 8, 9, 10, 11], "complic": [0, 8], "exampl": [0, 2, 8, 9], "consid": [0, 1, 2, 3, 7, 8, 9, 11], "weight": [0, 2, 3, 4, 5, 6, 8, 9, 10, 12], "vector": [0, 1, 2, 3, 5, 6, 8, 9, 10], "w": [0, 2, 3, 5, 6, 7, 8, 10, 12], "mathbb": [0, 2, 3, 7, 8], "r": [0, 2, 3, 7, 8], "n": [0, 2, 3, 7, 8], "unit": [0, 2, 6, 8, 9], "mean": [0, 2, 3, 4, 5, 6, 7, 8, 9, 12], "squar": [0, 2, 7, 8], "_2": [0, 8], "2": [0, 2, 3, 5, 6, 7, 8, 9], "sum_": [0, 2], "1": [0, 2, 3, 5, 6, 7, 8, 9, 10], "w_i": 0, "suppos": [0, 2, 4, 8], "gradient": [0, 2, 3, 4, 6, 7, 8, 9, 10, 12], "g": [0, 2, 3, 7, 8], "deriv": [0, 1, 2, 8, 10, 12], "some": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12], "loss": [0, 2, 3, 4, 5, 6, 7, 8, 9, 12], "function": [0, 2, 3, 4, 6, 7, 8, 9], "evalu": [0, 2, 6, 8], "given": [0, 1, 2, 7, 8], "step": [0, 1, 2, 5, 6, 7, 8, 9], "size": [0, 2, 5, 7, 8, 9, 10], "eta": [0, 2, 8, 12], "0": [0, 2, 5, 6, 7, 8, 9], "claim": [0, 2, 7, 8], "follow": [0, 1, 2, 3, 4, 9, 10], "updat": [0, 2, 3, 6, 7, 8, 9, 10, 12], "stai": [0, 2], "mapsto": [0, 2, 3, 8], "frac": [0, 2, 3, 7, 8, 9, 10], "sqrt": [0, 2, 7, 8, 9, 10], "time": [0, 2, 3, 4, 6, 7, 8], "left": [0, 2, 4, 7, 8], "i_n": [0, 2], "top": [0, 2, 3, 8, 9], "right": [0, 2, 4, 7, 8, 9], "so": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10], "simpli": [0, 2, 7, 10], "project": [0, 1, 2, 8], "subspac": [0, 8], "orthogon": [0, 1, 3, 8, 9], "normal": [0, 4, 6, 7, 8, 9, 10, 12], "obtain": [0, 2, 8, 10], "offload": 0, "problem": [0, 3, 4, 5, 8, 9], "set": [0, 2, 3, 6, 7, 8, 9, 12], "choos": [0, 8, 9, 12], "paramet": 0, "divid": [0, 2, 7, 8, 9], "through": [0, 2, 7, 8, 9, 10], "back": [0, 1, 2, 9, 10], "": [0, 1, 2, 4, 5, 7, 8, 9, 10, 11], "probabl": 0, "overkil": 0, "let": [0, 2, 4, 5, 7, 8, 9], "show": [0, 5, 7, 8, 9, 10], "formal": [0, 2, 8], "possibl": [0, 2, 8, 12], "veloc": [0, 2], "curv": [0, 1, 2, 7], "pass": [0, 2, 6], "For": [0, 2, 3, 7, 8, 9], "real": 0, "valu": [0, 2, 3, 7, 8, 9], "t": [0, 2, 4, 5, 6, 7, 8, 9, 10], "If": [0, 2, 8, 10, 11], "differenti": [0, 2], "condit": [0, 8, 10, 12], "partial": [0, 2], "must": [0, 2, 8, 10], "convers": [0, 2, 9], "satisfi": [0, 2], "manifold": [0, 11], "can": [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12], "seen": [0, 9], "studi": [0, 8, 9, 10], "cdot": [0, 3, 8], "co": 0, "sin": 0, "realli": [0, 8], "To": [0, 2, 7, 8, 9], "solv": [0, 1, 2, 3, 4, 6, 10, 12], "operatornam": [0, 2, 3, 7, 12], "arg": [0, 2, 3, 12], "max": [0, 2, 3, 7, 8], "_": [0, 2, 3, 7, 8, 10, 12], "leq": [0, 2, 3, 8, 12], "text": [0, 2, 3, 7, 8, 10], "us": [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12], "method": [0, 6, 8], "lagrang": 0, "multipli": [0, 8, 9], "write": [0, 4, 5, 7, 8, 10], "down": [0, 9], "lagrangian": 0, "mathcal": [0, 8, 12], "l": [0, 7, 8, 9, 10, 12], "lambda": [0, 7, 9], "mu": 0, "take": [0, 1, 2, 7, 8, 9, 10, 12], "respect": [0, 2, 4, 8, 9, 12], "zero": [0, 2, 3, 7, 9], "substitut": 0, "constraint": [0, 1, 2, 8, 12], "final": [0, 2, 5, 6, 7, 9], "make": [0, 2, 4, 6, 7, 8, 9, 10, 11], "along": [0, 4, 8], "calcul": [0, 3, 5, 8], "previou": [0, 2, 7, 8, 9], "section": [0, 1, 2, 4, 7, 8, 9, 11], "leav": [0, 1, 2, 7], "In": [0, 1, 2, 3, 4, 7, 8, 9, 10, 12], "fact": [0, 2, 7, 8, 9], "pythagora": 0, "theorem": [0, 8], "have": [0, 2, 4, 7, 8, 9, 10, 12], "scalar": [0, 2], "steepest": [1, 8], "descent": [1, 4, 7, 8, 9, 10, 12], "optim": [1, 2, 3, 4, 7, 8, 10], "equip": [1, 2, 8], "certain": [1, 3, 4, 8], "These": [1, 4, 8], "want": [1, 4, 8, 9, 11], "construct": [1, 2, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], "modul": [1, 6, 7, 8], "where": [1, 2, 3, 4, 7, 8, 9, 10, 12], "tensor": [1, 8, 9], "obei": 1, "natur": [1, 8, 9], "particular": [1, 2, 3, 8, 10, 11], "hyperspher": 1, "matric": [1, 2, 7, 8, 9, 10], "spectral": [1, 2, 7, 8, 9, 10, 12], "each": [1, 2, 5, 7, 8, 9], "case": [1, 2, 4, 7, 8, 9], "adopt": [1, 8], "strategi": 1, "character": [1, 2, 7], "tangent": 1, "space": [1, 3, 8, 9, 12], "direct": [1, 3, 8, 9, 10], "retract": 1, "which": [1, 2, 3, 4, 7, 8, 9, 10, 12], "taken": [1, 8, 10], "think": [1, 7, 8, 9], "plane": 1, "ly": [1, 2], "point": [1, 2, 5, 7, 8, 9], "ha": [1, 2, 8, 9, 10], "its": [1, 2, 4, 5, 6, 8, 9], "own": 1, "sinc": [1, 2, 7, 8, 9], "discret": 1, "therefor": [1, 2, 8, 9], "need": [1, 2, 4, 7, 8, 9], "It": [1, 2, 4, 8, 9, 10], "alwai": [1, 8, 9], "keep": [1, 3, 9], "mind": [1, 9], "pictur": [1, 4], "ar": [2, 4, 6, 7, 8, 9, 11, 12], "matrix": [2, 3, 7, 8, 9, 10], "maxim": [2, 6, 8, 10], "linear": [2, 3, 5, 6, 7, 8, 10, 12], "improv": [2, 4, 6, 11], "send": [2, 3, 8], "from": [2, 4, 5, 6, 7, 8, 9, 10], "oftentim": 2, "simpl": [2, 4, 6, 8, 9], "multipl": [2, 5, 8], "first": [2, 3, 4, 5, 7, 8, 9, 10, 11, 12], "comput": [2, 3, 6, 8, 9], "x": [2, 3, 5, 7, 8, 9], "sharp": [2, 3], "begin": [2, 7], "full": [2, 8], "rank": [2, 8, 9], "otherwis": [2, 8], "end": [2, 7, 8, 10], "express": [2, 8, 9], "m": [2, 3, 7, 8, 11, 12], "denot": [2, 3, 8], "oper": [2, 3, 4, 6, 7, 8, 9, 10], "appli": [2, 6, 8, 9, 10], "return": [2, 5, 6, 7, 9], "same": [2, 7, 8, 9], "singular": [2, 3, 7, 8, 9], "all": [2, 6, 7, 8, 9, 10], "posit": 2, "one": [2, 3, 5, 7, 8, 9, 10], "run": 2, "newton": [2, 8], "schulz": 2, "iter": [2, 4, 8, 9], "cubic": 2, "m_0": 2, "_f": [2, 7], "qquad": [2, 10], "m_": 2, "3": [2, 3, 5, 6, 7, 9], "m_t": 2, "m_tm_t": 2, "infti": [2, 8], "goe": [2, 8], "kovarik": [2, 3], "1970": 2, "bj\u00f6rck": 2, "bowi": [2, 3], "1971": 2, "One": [2, 8, 9], "reason": [2, 3, 8], "interest": [2, 8, 10], "A": [2, 8, 9, 10, 12], "riemmanian": 2, "call": [2, 6, 8, 9, 10, 12], "metric": 2, "inner": [2, 3, 8], "product": [2, 3, 8], "defin": [2, 4, 7], "provid": [2, 8, 12], "wai": [2, 4, 7, 8, 9], "measur": [2, 7, 8], "distanc": [2, 7, 10, 12], "geometri": [2, 8], "awar": [2, 7], "There": [2, 8, 12], "been": [2, 9, 10], "lot": [2, 8], "research": [2, 10, 11], "machin": [2, 8], "learn": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "context": [2, 3], "fast": [2, 8], "accur": 2, "without": [2, 7, 8, 9, 10, 12], "effici": [2, 6, 8], "via": [2, 10, 12], "caylei": 2, "transform": [2, 3, 4, 7, 8, 9], "howev": [2, 8], "seemingli": [2, 7, 8], "much": [2, 8, 9], "less": 2, "instanc": [2, 7, 8], "everi": [2, 9, 10], "doe": [2, 4, 7, 8, 9], "emerg": [2, 8], "But": [2, 7, 8, 9, 11], "believ": [2, 7, 8, 10], "kind": [2, 8, 9], "veri": [2, 7, 8, 9], "import": [2, 3, 5, 6, 7, 8, 9, 10, 12], "deep": [2, 3, 7, 8, 9, 10, 11, 12], "would": [2, 8, 10], "like": [2, 4, 8], "figur": [2, 5, 7], "roughli": [2, 9], "speak": 2, "particl": 2, "could": [2, 8], "parameter": [2, 8, 10], "evidenc": 2, "exp": 2, "tw": 2, "complet": [2, 7, 8], "chang": [2, 3, 7, 8, 9, 10], "variabl": [2, 9], "see": [2, 7, 8, 10], "belong": 2, "onli": [2, 7, 9], "skew": 2, "symmetr": [2, 3], "trace": [2, 8], "wish": 2, "simplifi": [2, 4, 10], "now": [2, 4, 5, 7, 8, 9, 10], "over": [2, 3, 7, 8, 9], "next": [2, 4, 5, 6, 7, 9], "decompos": 2, "compon": [2, 9], "realiz": [2, 8], "becaus": [2, 3, 4, 7, 8, 9], "contribut": [2, 5, 6, 8, 9, 10, 11], "part": [2, 8], "vanish": [2, 7], "becom": [2, 9], "ignor": [2, 4], "solut": [2, 9], "actual": [2, 3, 4, 5, 7, 8, 10], "preserv": 2, "symmetri": [2, 8], "easi": [2, 8, 9], "odd": 2, "polynomi": 2, "undo": 2, "our": [2, 5, 6, 7, 8, 9, 10], "suggest": [2, 7, 8], "diverg": [2, 7], "slightli": 2, "fix": [2, 3, 8], "issu": [2, 8, 11], "e": [2, 7, 8, 9], "shortcut": 2, "worth": [2, 9], "note": 2, "least": [2, 8, 9], "cannot": [2, 7], "avoid": [2, 8], "even": [2, 4, 7, 8, 9], "initi": [2, 4, 5, 6, 8, 9], "thought": [2, 3, 8], "easili": [2, 8], "semi": 2, "turn": [2, 4, 8, 9, 10], "gener": [2, 6, 8, 10], "rectangular": [2, 8], "longer": 2, "instead": [2, 8, 9, 10], "wx": 2, "overlin": 2, "y": [2, 7, 8], "column": [2, 3], "miss": [2, 10], "other": [2, 4, 9, 11, 12], "word": [2, 3, 8], "combin": [2, 8], "requir": [2, 6, 9], "unconstrain": 2, "do": [2, 7, 8, 9, 10], "know": [2, 8, 9], "how": [2, 4, 7, 8, 9, 10], "analyt": [2, 10], "result": [2, 7, 8], "ani": [2, 5, 6, 7, 8, 9, 10], "entrywis": 2, "understand": [2, 7, 9, 10, 12], "written": [2, 8, 10], "k": [2, 5, 7, 8], "sigma_i": [2, 7], "u_iv_i": 2, "v_i": 2, "u_i": 2, "neg": [2, 8], "cup": 2, "orthonorm": [2, 3], "lfloor": 2, "rfloor": 2, "admit": [2, 8], "svd": [2, 3, 8, 9], "come": 2, "pair": 2, "conjug": 2, "f": [2, 3, 5, 6, 7, 8], "yield": 2, "intact": 2, "read": [2, 8], "haber": 2, "2016": 2, "youla": 2, "1961": 2, "famili": 3, "either": [3, 10], "row": [3, 9], "form": [3, 8, 9], "map": [3, 7, 8, 9, 11], "reduc": 3, "u": [3, 6, 7, 8, 9, 10], "sigma": [3, 8, 9], "v": [3, 7, 8, 9], "snap": 3, "although": [3, 8, 10], "refer": 3, "correspond": [3, 8, 9, 11], "pronounc": 3, "sometim": [3, 4, 8], "treat": 3, "special": [3, 4], "procedur": [3, 8, 12], "contrast": [3, 8], "gram": 3, "schmidt": 3, "involv": [3, 8, 9, 10], "pick": [3, 9], "remain": [3, 4], "against": 3, "The": [3, 4, 7, 8, 11, 12], "care": [3, 4, 8, 10], "about": [3, 4, 7, 8, 9, 10, 11], "neural": [3, 4, 6, 7, 8, 9, 10, 11, 12], "network": [3, 4, 6, 7, 8, 9, 10, 12], "essenti": [3, 8, 10], "primit": 3, "delta": [3, 8, 10, 12], "langl": [3, 12], "rangl": [3, 12], "frobeniu": [3, 7, 8, 10], "tell": [3, 9], "squeez": [3, 8], "most": [3, 8, 9, 10], "control": [3, 8, 9, 10, 12], "allow": [3, 9], "guarante": 3, "featur": [3, 5, 6, 10, 12], "model": [3, 6, 8], "amount": [3, 7, 8, 10], "appear": [3, 8], "number": [3, 4, 5, 6, 7, 8, 9], "differ": [3, 4, 7, 8, 9, 12], "procrust": 3, "polar": 3, "factor": [3, 8, 9], "decomposit": [3, 8], "wa": [3, 7, 8, 10], "per": [3, 8], "olov": 3, "l\u00f6wdin": 3, "1950": 3, "atom": [3, 5, 6, 7, 8], "molecular": 3, "orbit": 3, "frank": 3, "wolf": 3, "ball": 3, "precondit": [3, 12], "bjorck": 3, "higham": 3, "anil": [3, 8], "gross": 3, "3x": 3, "16": [3, 7], "5": [3, 5, 7], "6": [3, 5, 7], "4445x": 3, "4": [3, 5, 6, 7, 8, 9], "7750x": 3, "0315x": 3, "At": [4, 7, 9], "simplest": [4, 8], "level": [4, 10], "train": [4, 5, 6, 7, 8, 9, 11], "learning_r": [4, 5, 6, 9], "float": [4, 8, 9], "Of": 4, "cours": [4, 7, 8, 9], "practic": [4, 8, 9, 10], "addit": [4, 8], "trick": 4, "momentum": [4, 8, 9], "detail": [4, 8], "unfortun": 4, "well": [4, 7, 8, 10], "up": [4, 6, 7, 8, 10], "architectur": [4, 8, 9, 11, 12], "what": [4, 7, 8, 9, 10], "befor": [4, 8], "grow": 4, "increas": [4, 7, 9, 10], "width": [4, 5, 6, 7, 8, 10], "neuron": 4, "layer": [4, 7, 8, 10, 12], "depth": [4, 8, 10], "might": [4, 7, 8, 9], "dimens": [4, 8, 9], "residu": [4, 8, 9], "block": [4, 8, 9], "stick": 4, "break": 4, "two": [4, 7, 8, 10, 12], "main": [4, 7, 8, 9, 10, 12], "rate": [4, 6, 7, 8, 9, 10, 12], "drift": 4, "re": [4, 5, 8, 10], "tune": 4, "thing": [4, 8], "expens": [4, 8], "consum": 4, "second": [4, 8, 9, 10, 12], "get": [4, 5, 7, 8, 9], "wors": [4, 9], "stabl": [4, 8, 9], "grew": 4, "hope": [4, 8, 9, 11], "better": [4, 8], "cartoon": 4, "illustr": [4, 9], "typic": 4, "behaviour": [4, 7], "deterior": 4, "new": [4, 7, 8, 9, 12], "develop": [4, 8, 10], "machineri": [4, 10], "larg": [4, 8, 9, 12], "woe": 4, "act": 4, "lead": [4, 8, 10, 11, 12], "remov": [4, 8], "caus": [4, 9], "modula": [4, 5, 6, 7, 9, 10, 12], "automat": [4, 8, 10, 12], "infer": 4, "necessari": [4, 8], "user": 4, "focu": [4, 8], "handl": [4, 6, 8], "properli": [4, 8], "doc": [4, 10], "intend": [4, 10], "explain": [4, 7, 8, 9, 10], "also": [4, 6, 7, 8, 9, 10, 11, 12], "introduc": [4, 9], "api": 4, "you": [4, 7, 8, 9, 10, 11], "don": [4, 8, 9], "manual": [4, 8], "framework": [4, 8, 9, 11], "pytorch": [4, 8], "jax": [4, 5, 6, 7], "extend": [5, 8, 9], "world": 5, "download": [5, 7], "data": [5, 6, 7], "load_mnist": 5, "load": 5, "dataset": [5, 7], "train_imag": [5, 7], "train_label": [5, 7], "test_imag": [5, 7], "test_label": [5, 7], "print": [5, 6, 7], "shape": [5, 7, 8, 9], "verifi": [5, 10], "imag": [5, 7], "label": [5, 7], "test": [5, 7, 8], "60000": 5, "28": 5, "10000": 5, "plot": [5, 7], "few": [5, 9], "matplotlib": [5, 7], "pyplot": [5, 7], "plt": [5, 7], "creat": [5, 6, 11], "subplot": [5, 7], "fig": [5, 7], "ax": 5, "figsiz": [5, 7], "10": [5, 6, 7, 9], "rang": [5, 6, 7], "imshow": [5, 7], "cmap": [5, 7], "grai": [5, 7], "set_titl": [5, 7], "axi": [5, 7], "off": [5, 7], "tight_layout": [5, 7], "flatten": [5, 8], "input": [5, 6, 7, 8, 9, 12], "encod": 5, "target": [5, 6, 7], "hot": 5, "mini": 5, "batch": [5, 7], "sampler": 5, "numpi": [5, 6, 7], "jnp": [5, 6, 7], "def": [5, 6, 7, 9], "one_hot": [5, 7], "dtype": 5, "float32": 5, "arrai": [5, 6, 7], "none": [5, 7], "arang": [5, 7], "reshap": [5, 7], "convert": [5, 9, 10], "x_train": [5, 7], "y_train": [5, 7], "get_batch": [5, 7], "kei": [5, 6, 7, 11], "batch_siz": [5, 6, 7], "idx": [5, 7], "random": [5, 6, 7, 8, 9, 10], "choic": [5, 7, 8], "readi": [5, 6], "build": [5, 6, 8, 12], "mlp": [5, 6, 7, 9], "bond": [5, 6, 7], "relu": [5, 6, 7], "input_dim": [5, 6, 7], "output_dim": [5, 6, 7], "256": [5, 6, 7], "jit": [5, 6, 7], "compositemodul": [5, 6], "consist": [5, 6, 8], "non": [5, 6, 7, 8, 9], "smooth": [5, 6, 8, 9, 12], "sensit": [5, 6], "proport": [5, 6, 8, 9], "supermodul": [5, 6], "1000": [5, 6, 7], "tqdm": [5, 7], "notebook": [5, 7], "mse": [5, 6], "output": [5, 6, 7, 8, 9, 12], "mse_and_grad": [5, 6], "value_and_grad": [5, 6, 7], "128": [5, 6, 7], "prngkei": [5, 6, 7], "progress_bar": 5, "desc": 5, "4f": 5, "grad_w": [5, 6, 7], "d_w": [5, 6, 7], "dualiz": [5, 6, 7, 12], "d_weight": [5, 6, 7], "zip": [5, 6, 7], "set_descript": 5, "And": [5, 6, 7, 8, 9, 10], "predict": [5, 7, 8], "x_test": 5, "test_output": 5, "predicted_label": 5, "argmax": [5, 7], "n_sampl": 5, "displai": 5, "int": [5, 7, 9], "accuraci": [5, 7], "sampl": [5, 6], "correct": [5, 9], "sum": [5, 9], "shown": [5, 7], "overal": [5, 8], "total_correct": 5, "total_sampl": 5, "len": [5, 9], "100": [5, 6, 7], "2f": 5, "97": 5, "57": 5, "loop": [6, 7, 9], "fit": 6, "randomli": [6, 8], "start": [6, 8, 9, 11], "explicitli": [6, 8], "state": [6, 8, 9], "784": 6, "basic": [6, 7, 8, 9, 10], "compos": 6, "them": [6, 8, 9, 12], "tri": 6, "intern": [6, 9], "just": [6, 8, 9, 10], "compil": 6, "both": [6, 7, 8, 9], "modular": [6, 7, 8, 12], "schedul": [6, 8], "lr": [6, 7], "3d": 6, "6f": 6, "976154": 6, "001773": 6, "200": [6, 7], "001371": 6, "300": 6, "001002": 6, "400": 6, "000696": 6, "500": 6, "000453": 6, "600": 6, "000282": 6, "700": 6, "000152": 6, "800": 6, "000061": 6, "900": 6, "000011": 6, "heard": [7, 10], "dure": [7, 8, 9, 10], "wide": [7, 8], "enough": [7, 9], "move": [7, 9], "lee": 7, "et": [7, 8], "al": [7, 8], "2019": [7, 10], "never": 7, "around": [7, 11], "jesu": 7, "2020": [7, 10], "conduct": [7, 8, 9], "intrigu": 7, "support": [7, 8], "narr": 7, "thei": [7, 8, 9], "letter": 7, "find": [7, 8], "still": [7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], "visibl": 7, "long": [7, 8, 9], "investig": [7, 12], "phenomenon": 7, "replic": 7, "dualiti": [7, 11, 12], "base": [7, 8, 9, 10], "exhibit": 7, "qualit": 7, "report": 7, "fulli": [7, 8, 9], "eras": 7, "line": [7, 10], "theoret": [7, 8], "made": [7, 10], "paper": [7, 8, 9, 10, 12], "understood": [7, 10], "term": [7, 8, 9], "underscor": 7, "metriz": [7, 8, 10, 11], "numer": [7, 8], "properti": [7, 8, 9, 10, 12], "helper": 7, "contain": [7, 10, 11, 12], "pil": 7, "imagedraw": 7, "imagefont": 7, "create_letter_matrix": 7, "font_nam": 7, "arial": 7, "ttf": 7, "img": 7, "255": 7, "draw": 7, "font": 7, "truetyp": 7, "bbox": 7, "textbbox": 7, "text_width": 7, "text_height": 7, "fill": 7, "resolut": 7, "50": 7, "letter_matric": 7, "side": [7, 8, 11], "enumer": 7, "titl": [7, 11], "cifar": 7, "cifar10": 7, "load_cifar10": 7, "1024": 7, "initial_w": 7, "unmasked_initial_matrix": 7, "masked_initial_matrix": 7, "ax1": 7, "ax2": 7, "vmin": 7, "min": [7, 12], "vmax": 7, "rdbu": 7, "record": 7, "easier": [7, 8, 10], "check": [7, 9, 11], "disabl": 7, "spec_norm": 7, "num_step": 7, "20": 7, "linalg": [7, 9], "true": 7, "eval": 7, "acc": 7, "eval_and_grad": 7, "has_aux": 7, "els": [7, 11], "append": 7, "isfinit": 7, "final_w": 7, "enabl": 7, "7": 7, "1001": 7, "lr_list": 7, "fals": 7, "averag": [7, 8], "last": 7, "overlai": 7, "8": 7, "extract": 7, "final_accs_du": 7, "final_accs_nodu": 7, "final_weights_du": 7, "final_weights_nodu": 7, "small": [7, 9], "main_ax": 7, "gca": 7, "With": [7, 8], "set_xscal": 7, "log": 7, "set_ylim": 7, "add": [7, 9], "lr_disp": 7, "acc_disp": 7, "transdata": 7, "lr_fig": 7, "acc_fig": 7, "transfigur": 7, "invert": 7, "inset": 7, "add_ax": 7, "05": 7, "set_xlabel": 7, "set_ylabel": 7, "after": [7, 8, 9, 10], "effect": [7, 8], "rcparam": 7, "tick_param": 7, "labels": 7, "xaxi": 7, "set_siz": 7, "yaxi": 7, "legend": 7, "loc": 7, "lower": 7, "frameon": 7, "fontsiz": 7, "three": [7, 10], "observ": [7, 8], "reach": [7, 8, 9, 10, 11], "significantli": 7, "higher": 7, "than": [7, 8, 9, 10], "maximum": 7, "short": [7, 8, 10], "faster": [7, 11], "regim": 7, "unlik": [7, 10], "standard": [7, 8, 9, 10], "evolv": 7, "human": [7, 10], "ey": 7, "ratio": 7, "srank": 7, "quantiti": [7, 8, 9], "discuss": 7, "entri": [7, 8, 9], "textrm": 7, "mn": 7, "vari": [7, 10], "depend": [7, 8, 12], "raw": [7, 8], "tend": [7, 9], "high": [7, 8], "between": [7, 8, 9, 10, 12], "place": [7, 11], "sourc": [7, 9], "inflat": 7, "equal": [7, 9], "ll": [7, 8], "altern": 7, "formula": [7, 10], "sum_i": 7, "max_i": 7, "9": 7, "stable_rank_grad": 7, "stable_rank_dualized_grad": 7, "ord": 7, "fro": 7, "32": 7, "64": 7, "512": 7, "2048": 7, "4096": 7, "8192": 7, "grad": [7, 9], "o": 7, "axvlin": 7, "color": 7, "linestyl": 7, "axhlin": 7, "xscale": 7, "yscale": 7, "xlabel": 7, "hidden": [7, 12], "ylabel": 7, "As": [7, 9], "meanwhil": 7, "until": 7, "plateau": 7, "expect": [7, 9], "cap": 7, "beyond": [7, 9], "itself": [7, 8], "when": [7, 8, 9, 10, 12], "vanilla": [7, 8], "hardli": 7, "substanti": [7, 10], "becam": 7, "thank": 7, "tweet": 7, "nora": 7, "belros": 7, "tongzhou": 7, "wang": 7, "ran": 7, "question": [7, 9], "feel": [8, 12], "free": [8, 9], "github": [8, 11], "post": 8, "answer": [8, 9], "common": 8, "mental": 8, "jump": [8, 11], "list": [8, 9], "mathbf": [8, 10], "_1": 8, "dot": 8, "_l": 8, "_k": 8, "sens": 8, "th": 8, "being": 8, "nabla_": 8, "w_k": 8, "meaningfulli": 8, "approach": [8, 9, 12], "theori": [8, 10, 11], "whole": 8, "big": 8, "thu": 8, "lose": 8, "structur": [8, 10, 12], "why": [8, 9, 10], "adam": [8, 9], "beat": 8, "sgd": 8, "challeng": 8, "certainli": 8, "languag": 8, "known": 8, "here": [8, 9, 11, 12], "aim": 8, "mechanist": 8, "explan": 8, "idea": [8, 9, 10, 11, 12], "should": [8, 9], "rel": 8, "across": [8, 9, 10], "major": [8, 10, 12], "rebal": 8, "give": 8, "concret": 8, "Then": [8, 11], "global": 8, "toi": 8, "w_": 8, "w_1": 8, "_0": 8, "depress": 8, "includ": [8, 9], "limit": [8, 9, 10], "advoc": 8, "inclus": 8, "nonlinear": 8, "simplic": 8, "_3": 8, "kill": 8, "comparison": 8, "stuck": 8, "sever": 8, "middl": [8, 9], "individu": [8, 9], "rebalanc": 8, "transfer": [8, 10], "scale": 8, "By": [8, 9], "definit": 8, "mathsf": 8, "behav": [8, 9], "gpt": [8, 9], "independ": 8, "littl": [8, 9, 10], "bit": [8, 10], "lipschitz": [8, 12], "inequ": 8, "hold": [8, 9], "tightli": 8, "approxim": 8, "approx": 8, "statement": [8, 9], "recurs": [8, 9, 12], "submodul": 8, "desir": 8, "compound": 8, "gut": 8, "found": [8, 9], "b": [8, 10], "insid": 8, "friendli": 8, "dimension": [8, 9], "symbol": [8, 12], "length": [8, 9, 10], "ell_2": 8, "cauchi": 8, "schwarz": 8, "tight": 8, "opposit": 8, "configur": 8, "higer": 8, "order": [8, 9, 10], "relev": 8, "li": 8, "sub": [8, 9], "sai": [8, 9], "quit": [8, 9, 10], "slack": 8, "orient": 8, "central": [8, 11], "tenet": [8, 9], "suffici": 8, "warm": [8, 9], "fall": [8, 9], "type": [8, 9, 12], "mention": 8, "variou": 8, "outlin": 8, "captur": 8, "whether": 8, "govern": 8, "expon": 8, "coupl": 8, "bear": 8, "empir": [8, 10], "describ": 8, "valuabl": 8, "futur": [8, 10], "uniqu": 8, "freedom": 8, "your": [8, 9], "said": 8, "agre": 8, "eas": 8, "commun": 8, "lora": 8, "\u03bcp": 8, "afficionado": 8, "d": 8, "fan": [8, 10], "propto": [8, 10], "restrict": 8, "homogeneu": 8, "ba": 8, "noth": 8, "exactli": [8, 9], "dynam": 8, "repres": 8, "scheme": 8, "represent": 8, "low": [8, 9], "format": 8, "charli": 8, "blake": 8, "exploit": 8, "parametr": 8, "summari": 8, "demonstr": 8, "sensibl": 8, "default": 8, "situat": [8, 9], "specif": 8, "modifi": 8, "rule": [8, 12], "advantag": 8, "design": [8, 9], "relationship": 8, "embed": 8, "arbitrari": 8, "composit": [8, 10], "concaten": 8, "net": [8, 9], "program": 8, "scienc": 8, "releas": 8, "almost": 8, "year": [8, 10, 11], "incarn": 8, "ground": 8, "elementari": 8, "math": [8, 9, 10], "estim": 8, "ingredi": 8, "bound": [8, 12], "track": [8, 12], "emploi": 8, "probabilist": 8, "analys": 8, "asymptot": 8, "unifi": [8, 10], "thread": 8, "seri": 8, "infinit": [8, 9, 10], "encumb": 8, "signific": [8, 10], "mathemat": [8, 9, 10], "overhead": 8, "often": [8, 9], "confront": 8, "thorni": 8, "commut": 8, "deal": 8, "directli": [8, 9], "finit": 8, "worri": 8, "lost": 8, "talk": 8, "built": [8, 11], "graph": 8, "priori": 8, "prior": 8, "hairi": 8, "henc": 8, "later": 8, "shift": [8, 10], "alreadi": [8, 9, 10], "impos": 8, "furthermor": 8, "done": [8, 9], "hand": [8, 11], "tabl": 8, "versu": 8, "skinni": 8, "ultim": 8, "inspir": [8, 9, 10, 11], "followup": 8, "clean": 8, "sound": 8, "toward": 8, "extens": 8, "ell_": 8, "rm": 8, "therebi": 8, "induc": [8, 10], "view": [8, 10], "stabil": [8, 10, 12], "infin": 8, "sort": 8, "consider": 8, "agd": 8, "analysi": [8, 12], "focus": [8, 10], "minim": [8, 10, 12], "style": [8, 9], "connect": 8, "surpris": 8, "aspect": 8, "decai": [8, 9], "hyperparamet": [8, 9, 10, 12], "slower": [8, 10], "convent": [8, 9], "setup": 8, "autom": [8, 9], "came": 8, "overli": 8, "pessimist": 8, "experi": [8, 9], "opt": 8, "rather": 8, "jeremi": [8, 10, 11, 12], "analogu": [8, 9], "peopl": [8, 10], "konstantin": 8, "mishchenko": 8, "aaron": 8, "defazio": 8, "prodigi": 8, "great": 8, "shampoo": 8, "myself": [8, 12], "whose": 8, "further": [8, 10, 11], "mathrm": [8, 9], "righthand": 8, "shorthand": 8, "minimis": 8, "That": 8, "juic": 8, "somewhat": 8, "classic": 8, "2015": [8, 10], "stochast": 8, "tim": [8, 12], "me": 8, "did": 8, "laker": [8, 12], "newhous": [8, 12], "who": 8, "talent": 8, "undergrad": [8, 10], "mit": [8, 10], "stop": 8, "cool": 8, "had": 8, "recent": [8, 10], "anoth": [8, 10], "tfrac": 8, "precondition": 8, "accumul": 8, "drop": 8, "perspect": [8, 10], "interpret": 8, "oppos": 8, "predomin": 8, "coolest": 8, "someth": [8, 9, 11], "rohan": 8, "slide": 8, "raphson": 8, "topic": [8, 10, 12], "root": 8, "slobodan": 8, "laki\u0107": 8, "implement": [8, 10], "gist": 8, "speedup": 8, "willing": 8, "toler": 8, "error": 8, "extrem": 8, "shampoolinear": 8, "replac": [8, 9], "zeroth": 8, "power": 8, "jack": 8, "gallagh": 8, "modulax": 8, "p": 8, "q": 8, "separ": 8, "satur": 8, "subvector": 8, "intial": 8, "No": 8, "gaussian": [8, 9], "wise": 8, "varianc": [8, 9], "difficult": [8, 9], "benign": 8, "subtl": [8, 9], "share": 8, "Not": 8, "present": [8, 10], "savant": 8, "field": 8, "bring": 8, "advanc": [8, 10], "tool": 8, "physic": 8, "my": [8, 10], "usual": 8, "simpler": [8, 9, 10], "strongli": 8, "stage": 8, "reson": 8, "rahimi": 8, "recht": 8, "georg": 8, "dahl": 8, "healthi": 8, "dose": 8, "skeptic": 8, "literatur": 8, "huh": [9, 12], "too": [9, 11], "boil": 9, "principl": [9, 10], "algebra": 9, "bad": 9, "unlearn": 9, "concept": 9, "taught": 9, "lectur": 9, "activ": 9, "101": 9, "compar": 9, "deviat": 9, "class": [9, 10], "__init__": 9, "self": 9, "fan_out": 9, "fan_in": 9, "torch": 9, "randn": 9, "forward": 9, "matmul": 9, "reduct": 9, "smaller": 9, "happen": [9, 10], "classifi": 9, "larger": 9, "null": 9, "huge": 9, "lie": 9, "nullspac": 9, "align": 9, "chose": 9, "far": 9, "hindsight": 9, "blow": 9, "assumpt": [9, 10], "fine": 9, "quickli": 9, "nice": 9, "bonu": 9, "switch": 9, "init": 9, "trivial": 9, "distil": 9, "largest": 9, "carefulli": 9, "expand": 9, "diagon": 9, "preced": 9, "head": 9, "tail": 9, "backpropag": 9, "rest": 9, "interv": 9, "equival": [9, 10], "achiev": 9, "intuit": 9, "spit": [9, 10], "clever": 9, "reparameter": 9, "reparameterizedlinear": 9, "empti": 9, "nn": 9, "orthogonal_": 9, "spectral_norm": 9, "option": 9, "frobenius_norm": 9, "look": 9, "resnet": 9, "residue_list": 9, "block_multipli": 9, "ad": 9, "ensur": 9, "third": 9, "total": 9, "similar": 9, "though": 9, "prevent": 9, "link": 9, "analogi": 9, "plai": 9, "role": 9, "safe": 9, "accord": 9, "convention": 9, "assum": 9, "uncorrel": 9, "spell": 9, "clearli": 9, "wisdom": 9, "logic": 9, "associ": 9, "scalabl": [9, 10, 11, 12], "anyth": 9, "object": 9, "vein": 9, "obviat": 9, "histori": 9, "behind": [9, 11], "applic": [9, 12], "outer": 9, "propos": [9, 10], "mani": [9, 10, 12], "best": 9, "knowledg": 9, "twist": 10, "particip": 10, "subfield": 10, "purpos": 10, "reader": 10, "llm": 10, "histor": 10, "potenti": 10, "bias": 10, "hi": 10, "piec": 10, "relat": 10, "love": 10, "pull": [10, 11], "request": [10, 11], "email": [10, 11], "internship": 10, "nvidia": 10, "instabl": 10, "biggan": 10, "arash": 10, "vahdat": 10, "ming": 10, "yu": 10, "liu": [10, 12], "perturb": 10, "learnt": 10, "senthil": 10, "todadri": 10, "graduat": 10, "quantum": 10, "mechan": 10, "took": 10, "continu": 10, "caltech": 10, "phd": 10, "advisor": 10, "yisong": 10, "yue": 10, "bernstein": [10, 11, 12], "neurip": [10, 12], "core": 10, "emphasis": 10, "version": 10, "account": 10, "anticip": [10, 11], "unlock": 10, "workflow": 10, "deeper": 10, "complex": 10, "minut": 10, "youtub": 10, "video": 10, "lai": 10, "wrote": 10, "arxiv": [10, 12], "2002": 10, "03432": 10, "greg": 10, "yang": [10, 12], "edward": 10, "hu": 10, "j": 10, "icml": 10, "2021": 10, "argument": 10, "parameteris": 10, "arguabl": 10, "innov": 10, "sweep": 10, "earlier": 10, "were": 10, "inaccur": 10, "fromag": 10, "lar": 10, "team": 10, "jami": 10, "simon": 10, "reconcil": 10, "jame": 10, "2023": 10, "streamlin": 10, "singl": 10, "mathtt": 10, "_out": 10, "_in": 10, "formul": 10, "inspect": 10, "anecdot": 10, "hug": 10, "face": 10, "nanotron": 10, "brain": 10, "reliabl": 10, "shouldn": 10, "artifici": 10, "system": [10, 11], "organ": 10, "pursu": 10, "agenda": 10, "chri": 10, "mingard": 10, "kevin": 10, "huang": 10, "navid": 10, "azizan": 10, "surprisingli": 10, "techniqu": 10, "abandon": 10, "intrins": 11, "benefit": 11, "current": 11, "process": 11, "overhaul": 11, "code": 11, "unclear": 11, "faq": 11, "everyon": 11, "arrow": 11, "panel": 11, "origin": 11, "publish": 11, "anywher": 11, "bibtex": 11, "misc": 11, "author": [11, 12], "url": 11, "http": 11, "2025": 11, "academ": 12, "assign": 12, "avail": 12, "minyoung": 12, "hyojin": 12, "bahng": 12, "phillip": 12, "isola": 12, "2024": 12, "subject": 12, "nabla": 12, "due": 12, "leverag": 12, "univers": 12, "old": 12, "anthologi": 12, "muon": 12, "margin": 12, "pac": 12, "bayesian": 12}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"hyperspher": 0, "steepest": [0, 2, 3], "descent": [0, 2, 3], "The": [0, 2, 9, 10], "structur": [0, 2], "tangent": [0, 2], "space": [0, 2], "direct": [0, 2], "find": [0, 2], "retract": [0, 2], "map": [0, 1, 2], "manifold": [1, 2], "dualiti": 1, "orthogon": 2, "non": 2, "riemannian": 2, "method": 2, "open": 2, "problem": 2, "extend": 2, "stiefel": 2, "newton": 3, "schulz": 3, "under": 3, "spectral": 3, "norm": 3, "histor": 3, "connect": 3, "polynomi": 3, "iter": 3, "A": 3, "cubic": 3, "quintic": 3, "speedi": 3, "bad": 4, "scale": [4, 9, 10], "hello": [5, 6], "mnist": 5, "world": 6, "weight": 7, "erasur": 7, "setup": 7, "creat": 7, "watermark": 7, "initi": 7, "run": 7, "experi": 7, "interpret": 7, "via": 7, "stabl": 7, "rank": 7, "conclus": 7, "acknowledg": 7, "frequent": 8, "ask": 8, "question": 8, "conceptu": 8, "other": 8, "notion": 8, "align": 8, "matter": 8, "precis": 8, "relat": 8, "work": 8, "modula": [8, 11], "packag": 8, "research": 8, "philosophi": 8, "golden": 9, "rule": 9, "linear": [9, 16], "layer": 9, "three": 9, "fix": 9, "width": 9, "depth": 9, "kei": 9, "queri": 9, "dot": 9, "product": 9, "wrap": 9, "up": 9, "scienc": 10, "warn": [10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], "pre": 10, "histori": 10, "\u03bcp": 10, "enter": 10, "chat": 10, "truth": 10, "reconcili": 10, "autom": 10, "train": 10, "welcom": 11, "doc": 11, "purpos": 11, "navig": 11, "cite": 11, "read": 12, "list": 12, "optim": 12, "gener": 12, "conv2d": 13, "embed": 14, "atom": 15, "modul": [15, 17, 20, 21], "bond": 17, "nonlinear": 18, "gpt": 19, "compound": 20, "vector": 22}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "nbsphinx": 4, "sphinx": 57}, "alltitles": {"Hypersphere": [[0, "hypersphere"]], "Steepest descent on the hypersphere": [[0, "steepest-descent-on-the-hypersphere"]], "The structure of the tangent space": [[0, "the-structure-of-the-tangent-space"], [2, "the-structure-of-the-tangent-space"]], "Steepest direction in the tangent space": [[0, "steepest-direction-in-the-tangent-space"], [2, "steepest-direction-in-the-tangent-space"]], "Finding the retraction map": [[0, "finding-the-retraction-map"], [2, "finding-the-retraction-map"]], "Manifold duality maps": [[1, "manifold-duality-maps"]], "Orthogonal manifold": [[2, "orthogonal-manifold"]], "Steepest descent on the orthogonal manifold": [[2, "steepest-descent-on-the-orthogonal-manifold"]], "Non-Riemannian manifold methods": [[2, "non-riemannian-manifold-methods"]], "Open problem: Extending to the Stiefel Manifold": [[2, "open-problem-extending-to-the-stiefel-manifold"]], "Newton-Schulz": [[3, "newton-schulz"]], "Steepest descent under the spectral norm": [[3, "steepest-descent-under-the-spectral-norm"]], "Historical connections": [[3, "historical-connections"]], "Polynomial iterations": [[3, "polynomial-iterations"]], "A cubic iteration": [[3, "a-cubic-iteration"]], "A quintic iteration": [[3, "a-quintic-iteration"]], "A speedy iteration": [[3, "a-speedy-iteration"]], "Bad scaling": [[4, "bad-scaling"]], "Hello, MNIST!": [[5, "Hello,-MNIST!"]], "Hello, World!": [[6, "Hello,-World!"]], "Weight erasure": [[7, "Weight-erasure"]], "Setup: Creating a watermark": [[7, "Setup:-Creating-a-watermark"]], "Watermarking the initial weights": [[7, "Watermarking-the-initial-weights"]], "Running the experiment": [[7, "Running-the-experiment"]], "Interpretation via the stable rank": [[7, "Interpretation-via-the-stable-rank"]], "Conclusion": [[7, "Conclusion"]], "Acknowledgements": [[7, "Acknowledgements"]], "Frequently asked questions": [[8, "frequently-asked-questions"]], "Conceptual questions": [[8, "conceptual-questions"]], "Other notions of alignment": [[8, null]], "Matters of precision": [[8, null]], "Related work": [[8, "related-work"]], "Modula package": [[8, "modula-package"]], "Research philosophy": [[8, "research-philosophy"]], "Golden rules for scaling": [[9, "golden-rules-for-scaling"]], "The linear layer": [[9, "the-linear-layer"]], "Three golden rules": [[9, "three-golden-rules"]], "Fixing width scaling": [[9, "fixing-width-scaling"]], "Fixing depth scaling": [[9, "fixing-depth-scaling"]], "Fixing key-query dot product scaling": [[9, "fixing-key-query-dot-product-scaling"]], "Wrapping up": [[9, "wrapping-up"]], "The science of scale": [[10, "the-science-of-scale"]], "Warning": [[10, null], [13, null], [14, null], [15, null], [16, null], [17, null], [18, null], [19, null], [20, null], [21, null], [22, null]], "Pre-history": [[10, "pre-history"]], "\u03bcP enters the chat": [[10, "p-enters-the-chat"]], "Truth and reconciliation": [[10, "truth-and-reconciliation"]], "Automation of training": [[10, "automation-of-training"]], "Welcome to the Modula docs!": [[11, "welcome-to-the-modula-docs"]], "Purpose of the docs": [[11, "purpose-of-the-docs"]], "Navigating the docs": [[11, "navigating-the-docs"]], "Citing the docs": [[11, "citing-the-docs"]], "Reading list": [[12, "reading-list"]], "Optimization": [[12, "optimization"]], "Generalization": [[12, "generalization"]], "Conv2d": [[13, "conv2d"]], "Embedding": [[14, "embedding"]], "Atomic modules": [[15, "atomic-modules"]], "Linear": [[16, "linear"]], "Bond modules": [[17, "bond-modules"]], "Nonlinearities": [[18, "nonlinearities"]], "GPT": [[19, "gpt"]], "Compound modules": [[20, "compound-modules"]], "Modules": [[21, "modules"]], "Vectors": [[22, "vectors"]]}, "indexentries": {}})