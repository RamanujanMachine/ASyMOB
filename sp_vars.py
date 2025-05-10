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

# This is a constant, not a variable. It is used since sympy identifies 
# it as `E`, which causes ambiguity in the code.
pi = sp.var('pi')
i = sp.var('i')


var_mapping = {
    'A': A,
    'B': B,
    'C': C,
    'D': D,
    'E': E,
    'F': F,
    'G': G,
    'H': H,
    'J': J,
    'K': K,
    'x': x,
    'k': k,
    'n': n,
    'e': sp.exp(1),  # e is not a variable, but a constant
    'pi': pi,
    'i': i
}

used_vars = list(var_mapping.keys())
