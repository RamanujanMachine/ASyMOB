import sympy as sp

A = sp.symbols('A')
B = sp.symbols('B')
C = sp.symbols('C')
D = sp.symbols('D')
E = sp.symbols('E')
F = sp.symbols('F')
G = sp.symbols('G')
H = sp.symbols('H')
J = sp.symbols('J')
K = sp.symbols('K')

x = sp.symbols('x')
n = sp.symbols('n')
m = sp.symbols('m')
k = sp.symbols('k')

var_mapping = {
    'A': A,
    'B': B,
    'C': C,
    'D': D,
    'E': E,
    'F': F,
    'G': G,
    'x': x,
    'k': k,
    'n': n,
    'm': m
}

used_vars = list(var_mapping.keys())
