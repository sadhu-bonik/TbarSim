import time
from crossbar import Tbar

t0 = time.time()
cb = Tbar(n=4, m=4, p=3)
t1 = time.time()
print(f"Graph build: {t1-t0:.3f}s -> {cb.G.number_of_nodes()} nodes, {cb.G.number_of_edges()} edges")

cb.randomize_conductances(low=1, high=100, seed=42)
cb.set_bias(vin=3.2, vout=0.0)
t2 = time.time()
print(f"Conductances + bias assigned: {t2-t1:.3f}s")

t3 = time.time()
cb.solve()
t4 = time.time()
print(f"Solve: {t4-t3:.3f}s")

print()
print("Sample results:")
print("  V(W(0,0,0)) =", cb.get_voltage("W(0,0,0)"))
print("  V(W(3,3,3)) =", cb.get_voltage("W(2,2,2)"))
print("  V(B(0,0,0)) =", cb.get_voltage("B(0,0,0)"))
print("  I(W(0,0,0)-B(0,0,0)) =", cb.get_current("W(0,0,0)", "B(0,0,0)"))
print()
print(f"Total time: {t4-t0:.3f}s")
cb.show()