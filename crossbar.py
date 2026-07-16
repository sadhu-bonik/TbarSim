import numpy as np
import networkx as nx
import plotly.graph_objects as go
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from typing import Self

class Tbar:
    """
    A simulator for a 3D memristor crossbar array using Nodal Analysis (KCL).

    Attributes:
        n (int): Number of rows (wordlines).
        m (int): Number of columns (bitlines).
        p (int): Number of layers in the 3D structure.
        S (float): Input source conductance (Siemens).
        L (float): Output load conductance (Siemens).
        GW (float): Wire segment and via conductance (Siemens).
        G (nx.Graph): The underlying NetworkX graph structure.
        GMem (np.ndarray): A 3D array of shape (n, m, p) storing memristor conductances.
        VInput (np.ndarray): A 2D array of shape (n, p) storing input bias voltages.
        VOutput (np.ndarray): A 2D array of shape (m, p) storing output bias voltages.
    """

    def __init__(self, n: int, m: int, p: int, S: float = 1e12, L: float = 1e12, GW: float = 1e12) -> None:
        """
        Initializes the 3D crossbar array structure and sets parasitic conductances.

        Args:
            m (int): Number of columns (bitlines) in the array.
            n (int): Number of rows (wordlines) in the array.
            p (int): Number of vertical layers in the array.
            S (float, optional): Input resistor conductance from voltage source to the 
                first wordline node. Defaults to 1e12 (near-perfect conductor).
            L (float, optional): Output resistor conductance from the last bitline node 
                to the output sink. Defaults to 1e12 (near-perfect conductor).
            GW (float, optional): Wire segment and via conductance between internal 
                nodes. Defaults to 1e12.
        """
        self.n: int = n  
        self.m: int = m  
        self.p: int = p
        self.S = S
        self.L = L
        self.GW = GW

        self.G = nx.Graph()
        self._build_nodes()
        self._build_edges()

        self.GMem = np.full((p, n, m), np.nan)
        self.VInput = np.full((n, p), np.nan)
        self.VOutput = np.full((m, p), np.nan)

        # Set once solve() has run successfully.
        self._voltages = None  # dict: node_name -> voltage

    def set_parasitic_resistance(self, RW: float):
        """
        Updates the parasitic resistance

        Args:
            RW (float, optional): New wire/via resistance.
        """
        self.GW = 1.0 / RW if RW != 0 else 1e12

    def set_parisitic_conductance(self, GW: float):
        """
        Updates the parasitic conductance

        Args:
            RW (float, optional): New wire/via conductance.
        """
        self.GW = GW

    def set_input_resistance(self, S: float):
        """
        Updates the input resistance

        Args:
            S (float, optional): New input resistance.
        """
        self.S = 1.0 / S if S != 0 else 1e12

    def set_output_resistance(self, L: float):
        """
        Updates the output resistance

        Args:
            L (float, optional): New output resistance.
        """
        self.L = 1.0 / L if L != 0 else 1e12

    def set_input_conductance(self, S: float):
        """
        Updates the input conductance

        Args:
            S (float, optional): New input conductance.
        """
        self.S = S

    def set_output_conductance(self, L: float):
        """
        Updates the output condutance

        Args:
            L (float, optional): New output conductance.
        """
        self.L = L

    def _build_nodes(self) -> None:
        n, m, p = self.n, self.m, self.p

        # Wordline nodes
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    self.G.add_node(f"W({i},{j},{k})", pos=(j, i, k), category="W")

        # Bitline nodes (offset so they sit next to their W node)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    self.G.add_node(f"B({i},{j},{k})", pos=(j + 0.5, i + 0.5, k - 0.5), category="B")

        # Input nodes: one per (row, layer)
        for i in range(n):
            for k in range(p):
                self.G.add_node(f"I({i},{k})", pos=(-1, i, k), category="I")

        # Output nodes: one per (column, layer)
        for j in range(m):
            for k in range(p):
                self.G.add_node(f"O({j},{k})", pos=(j + 0.5, n + 0.5, k - 0.5), category="O")

    def _build_edges(self) -> None:
        n, m, p = self.n, self.m, self.p

        # Input resistor: I -> W
        for i in range(n):
            for k in range(p):
                self.G.add_edge(f"I({i},{k})", f"W({i},0,{k})", category="1")

        # Output resistor: B -> O
        for j in range(m):
            for k in range(p):
                self.G.add_edge(f"B({n-1},{j},{k})", f"O({j},{k})", category="2")

        # Wordline segments (along j)
        for i in range(n):
            for j in range(m - 1):
                for k in range(p):
                    self.G.add_edge(f"W({i},{j},{k})", f"W({i},{j+1},{k})", category="3a")

        # Bitline segments (along i)
        for i in range(n - 1):
            for j in range(m):
                for k in range(p):
                    self.G.add_edge(f"B({i},{j},{k})", f"B({i+1},{j},{k})", category="3b")

        # Vertical vias (along k), for both W and B
        for i in range(n):
            for j in range(m):
                for k in range(p - 1):
                    self.G.add_edge(f"W({i},{j},{k})", f"W({i},{j},{k+1})", category="3c")
                    self.G.add_edge(f"B({i},{j},{k})", f"B({i},{j},{k+1})", category="3c")

        # Memristors: W -> B
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    self.G.add_edge(f"W({i},{j},{k})", f"B({i},{j},{k})", category="4")

    # ---- Step 2: conductances ----

    def set_conductances(self, matrix):
        """
        Assigns memristor conductances in bulk using a 3D array.

        Args:
            matrix (np.ndarray | list): A 3D array of conductances. 

        Raises:
            ValueError: If the provided matrix does not match the shape (n, m, p).
        """
        matrix = np.asarray(matrix, dtype=float)
        expected = (self.p, self.n, self.m)
        if matrix.shape != expected:
            raise ValueError(f"matrix has shape {matrix.shape}, expected {expected} (p, n, m)")
        self.GMem = matrix

    def set_conductance(self, i: int, j: int, k: int, value: float) -> None:
        """
        Sets the conductance for a specific memristor at a given 3D coordinate.

        Args:
            i (int): Row index.
            j (int): Column index.
            k (int): Layer index.
            value (float): Conductance value in Siemens.
        """        
        self.GMem[k, i, j] = value

    def randomize_conductances(
        self,
        low: int = 0,
        high: int = 100,
        seed: int | None = None,
        only_unset: bool = True,
    ) -> None:
        """
        Fills the crossbar array with random memristor conductances.

        Args:
            low (int, optional): Minimum random conductance value. Defaults to 0.
            high (int, optional): Maximum random conductance value. Defaults to 100.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            only_unset (bool, optional): If True, only fills cells that have not been 
                manually set yet (NaN). If False, overwrites the entire array. Defaults to True.
        """
        rng = np.random.default_rng(seed)
        random_vals = rng.integers(low, high, size=self.GMem.shape).astype(float)
        if only_unset:
            mask = np.isnan(self.GMem)
            self.GMem[mask] = random_vals[mask]
        else:
            self.GMem = random_vals

    # ---- Step 3: bias voltages ----

    def set_input_voltages(self, matrix):
        """
        Assigns input biases in bulk using a 2D array.

        Args:
            matrix (np.ndarray | list): A 2D array of voltages.

        Raises:
            ValueError: If the matrix does not match the shape (n, p).
        """
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (self.n, self.p):
            raise ValueError(f"matrix has shape {matrix.shape}, expected {(self.n, self.p)} (n, p)")
        self.VInput = matrix

    def set_output_voltages(self, matrix):
        """
        Assigns output biases in bulk using a 2D array.

        Args:
            matrix (np.ndarray | list): A 2D array of voltages.

        Raises:
            ValueError: If the matrix does not match the shape (m, p).
        """
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (self.m, self.p):
            raise ValueError(f"matrix has shape {matrix.shape}, expected {(self.m, self.p)} (m, p)")
        self.VOutput = matrix

    def set_input_voltage(self, i: int, k: int, value: float) -> None:
        """Set a single input node I(i,k)."""
        self.VInput[i, k] = value

    def set_output_voltage(self, j: int, k: int, value: float) -> None:
        """Set a single output node O(j,k)."""
        self.VOutput[j, k] = value

    def set_bias(
        self,
        vin: float | None = None,
        vout: float | None = None,
        only_unset: bool = False,
    ) -> None:
        """
        Fills all input or output nodes with a uniform constant voltage.

        Args:
            vin (float, optional): The voltage to apply to input nodes.
            vout (float, optional): The voltage to apply to output nodes.
            only_unset (bool, optional): If True, only fills nodes that are currently NaN. 
                If False, overwrites all specified nodes. Defaults to False.
        """
        if vin is not None:
            if only_unset:
                mask = np.isnan(self.VInput)
                self.VInput[mask] = vin
            else:
                self.VInput[:, :] = vin
        if vout is not None:
            if only_unset:
                mask = np.isnan(self.VOutput)
                self.VOutput[mask] = vout
            else:
                self.VOutput[:, :] = vout

    # ---- Step 4: solve (KCL) ----

    def _wordline_node_index(self, i: int, j: int, k: int) -> int:
        return int(2 * self.m * self.n * k + 2 * self.m * i + 2 * j)

    def _bitline_node_index(self, i: int, j: int, k: int) -> int:
        return int(2 * self.m * self.n * k + 2 * self.m * i + 2 * j + 1)

    def _assign_weights(self) -> None:
        """Set the 'weight' attribute on every edge from S, L, GW, and GMem."""
        for u, v, data in self.G.edges(data=True):
            category = data["category"]
            if category == "1":
                data["weight"] = self.S
            elif category == "2":
                data["weight"] = self.L
            elif category.startswith("3"):
                data["weight"] = self.GW
            elif category == "4":
                w_node = u if u.startswith("W") else v
                i, j, k = (int(x) for x in w_node[2:-1].split(","))
                data["weight"] = self.GMem[k, i, j]

    def is_ready_to_solve(self) -> bool:
        """True if conductances and bias voltages are fully specified."""
        cond_ready = not np.isnan(self.GMem).any()
        bias_ready = not np.isnan(self.VInput).any() and not np.isnan(self.VOutput).any()
        return cond_ready and bias_ready

    def solve(self) -> Self:
        """
        Constructs and solves the KCL linear system (G * V = I) for the crossbar array.

        Once solved, it calculates the current passing through every edge and stores the 
        voltages for retrieval.
        """
        if not self.is_ready_to_solve():
            raise RuntimeError(
                "Cannot solve: conductances and/or bias voltages are not fully set. "
                "Call cb.show() to see what's missing."
            )

        self._assign_weights()
        n, m, p = self.n, self.m, self.p
        size_kcl = 2 * n * m * p

        GKCL = sps.lil_matrix((size_kcl, size_kcl))
        IKCL = np.zeros(size_kcl)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for node_type, node_index_fn in (("W", self._wordline_node_index),
                                                   ("B", self._bitline_node_index)):
                        node = f"{node_type}({i},{j},{k})"
                        node_index = node_index_fn(i, j, k)
                        total_conductance = 0.0

                        for neighbor in self.G.neighbors(node):
                            weight = self.G[node][neighbor]["weight"]

                            if neighbor.startswith("I"):
                                ni, nk = (int(x) for x in neighbor[2:-1].split(","))
                                IKCL[node_index] += self.VInput[ni, nk] * weight
                                total_conductance += weight
                            elif neighbor.startswith("O"):
                                nj, nk = (int(x) for x in neighbor[2:-1].split(","))
                                IKCL[node_index] += self.VOutput[nj, nk] * weight
                                total_conductance += weight
                            elif neighbor.startswith("W"):
                                ni, nj, nk = (int(x) for x in neighbor[2:-1].split(","))
                                neighbor_index = self._wordline_node_index(ni, nj, nk)
                                GKCL[node_index, neighbor_index] += -weight
                                total_conductance += weight
                            elif neighbor.startswith("B"):
                                ni, nj, nk = (int(x) for x in neighbor[2:-1].split(","))
                                neighbor_index = self._bitline_node_index(ni, nj, nk)
                                GKCL[node_index, neighbor_index] += -weight
                                total_conductance += weight

                        GKCL[node_index, node_index] = total_conductance

        VKCL = spla.spsolve(GKCL.tocsr(), IKCL)

        # Store voltages for every node, including I/O nodes (bias values directly).
        self._voltages = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    self._voltages[f"W({i},{j},{k})"] = VKCL[self._wordline_node_index(i, j, k)]
                    self._voltages[f"B({i},{j},{k})"] = VKCL[self._bitline_node_index(i, j, k)]
        for i in range(n):
            for k in range(p):
                self._voltages[f"I({i},{k})"] = self.VInput[i, k]
        for j in range(m):
            for k in range(p):
                self._voltages[f"O({j},{k})"] = self.VOutput[j, k]

        # Annotate every edge with its current: I = |Vu - Vv| * weight
        for u, v, data in self.G.edges(data=True):
            vu = self._voltages[u]
            vv = self._voltages[v]
            data["current"] = abs(vu - vv) * data["weight"]

        return self

    @property
    def solved(self) -> bool:
        return self._voltages is not None

    def get_voltage(self, node_name: str) -> float:
        """
        Retrieves the calculated voltage for a specific node.

        Args:
            node_name (str): The node string identifier (e.g., 'W(0,0,0)', 'I(1,0)').

        Returns:
            float: Voltage at the requested node.

        Raises:
            RuntimeError: If called before solve().
        """        
        if not self.solved:
            raise RuntimeError("Call solve() first.")
        return self._voltages[node_name]

    def get_current(self, u: str, v: str) -> float:
        """
        Retrieves the calculated absolute current flowing between two nodes.

        Args:
            u (str): First node identifier.
            v (str): Second node identifier.

        Returns:
            float: Current through the edge connecting u and v.

        Raises:
            RuntimeError: If called before solve().
        """
        if not self.solved:
            raise RuntimeError("Call solve() first.")
        return self.G[u][v]["current"]

    def _conductance_summary(self) -> tuple[int, int]:
        """Returns (num_set, total) for memristor conductances."""
        total = self.GMem.size
        num_set = int(np.sum(~np.isnan(self.GMem)))
        return num_set, total

    def _print_status(self) -> None:
        """Prints what's currently defined and what to do next."""
        num_set, total = self._conductance_summary()
        vin_set = int(np.sum(~np.isnan(self.VInput)))
        vin_total = self.VInput.size
        vout_set = int(np.sum(~np.isnan(self.VOutput)))
        vout_total = self.VOutput.size

        print(f"--- Crossbar status ({self.n}x{self.m}x{self.p}) ---")
        print(f"Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

        if num_set == 0:
            print("Conductances: none set yet.")
            print("Next: set memristor conductances with set_conductances(), "
                  "set_conductance(i,j,k,value), or randomize_conductances().")
        elif num_set < total:
            vals = self.GMem[~np.isnan(self.GMem)]
            print(f"Conductances: {num_set}/{total} memristors set "
                  f"(range {vals.min():.3g} - {vals.max():.3g}).")
            print("Next: finish setting the remaining memristors before solving.")
        else:
            vals = self.GMem
            print(f"Conductances: all {total} memristors set "
                  f"(range {vals.min():.3g} - {vals.max():.3g}).")

            print(f"Input bias: {vin_set}/{vin_total} nodes set.")
            print(f"Output bias: {vout_set}/{vout_total} nodes set.")

            if vin_set < vin_total or vout_set < vout_total:
                print("Next: finish setting bias voltages with set_bias(), "
                      "set_input_voltage(i,k,value), or set_output_voltage(j,k,value).")
            elif not self.solved:
                print("Next: call solve() to compute voltages and currents.")
            else:
                print("Solved. Use get_voltage(node) / get_current(u,v), "
                      "or show() to view results.")
        print("-" * 40)

    def show(self) -> go.Figure:
        """
        Renders an interactive 3D visualization of the graph structure in Plotly.

        Returns:
            go.Figure: The generated Plotly figure object.
        """
        self._print_status()

        edge_styles = {
            "1": {"name": "Input Resistor", "color": "#00BFA5"},
            "2": {"name": "Output Resistor", "color": "#29B6F6"},
            "3a": {"name": "Wordline segment", "color": "#EC407A"},
            "3b": {"name": "Bitline segment", "color": "#5C6BC0"},
            "3c": {"name": "Vertical via", "color": "#BDC3C7"},
            "4": {"name": "Memristor", "color": "#FF9100"},
        }
        node_styles = {
            "W": {"name": "Wordline Node", "color": "#0000FF"},
            "B": {"name": "Bitline Node", "color": "#FF0000"},
            "I": {"name": "Input Node", "color": "#00FFFF"},
            "O": {"name": "Output Node", "color": "#FF00FF"},
        }

        traces = []
        for cat, style in edge_styles.items():
            ex, ey, ez = [], [], []
            mid_x, mid_y, mid_z, mid_text = [], [], [], []

            for u, v, data in self.G.edges(data=True):
                if data["category"] != cat:
                    continue
                x0, y0, z0 = self.G.nodes[u]["pos"]
                x1, y1, z1 = self.G.nodes[v]["pos"]
                ex += [x0, x1, None]
                ey += [y0, y1, None]
                ez += [z0, z1, None]

                mid_x.append((x0 + x1) / 2)
                mid_y.append((y0 + y1) / 2)
                mid_z.append((z0 + z1) / 2)

                if cat == "4":
                    i, j, k = (int(x) for x in u[2:-1].split(","))
                    g = self.GMem[k, i, j]
                    g_label = "not set" if np.isnan(g) else f"{g:.3g}"
                    text = f"{u} - {v}<br>G = {g_label}"
                else:
                    text = f"{u} - {v}"

                if self.solved:
                    current = data.get("current")
                    text += f"<br>I = {current:.4g}"

                mid_text.append(text)

            # the wire itself: no hover, just the visual line(s)
            traces.append(go.Scatter3d(
                x=ex, y=ey, z=ez, mode="lines",
                line=dict(color=style["color"], width=10 if cat == "4" else 4),
                name=style["name"], hoverinfo="none",
                showlegend=(cat != "4"),
            ))

            # one shared marker trace per category: this is what you hover on
            # (Plotly only triggers hover at line endpoints, not the interior,
            # so a dedicated midpoint marker is needed to inspect edge values)
            marker_style = (dict(size=4, color="#1A1A1A", symbol="diamond")
                             if cat == "4" else dict(size=2, color=style["color"]))
            traces.append(go.Scatter3d(
                x=mid_x, y=mid_y, z=mid_z, mode="markers",
                marker=marker_style,
                name=style["name"], text=mid_text, hoverinfo="text",
                showlegend=(cat == "4"),
            ))

        for cat, style in node_styles.items():
            nx_, ny_, nz_, ntext = [], [], [], []
            for node, data in self.G.nodes(data=True):
                if data["category"] != cat:
                    continue
                x, y, z = data["pos"]
                nx_.append(x); ny_.append(y); nz_.append(z)
                if cat == "I":
                    i, k = (int(v) for v in node[2:-1].split(","))
                    val = self.VInput[i, k]
                elif cat == "O":
                    j, k = (int(v) for v in node[2:-1].split(","))
                    val = self.VOutput[j, k]
                elif cat in ("W", "B") and self.solved:
                    val = self._voltages[node]
                else:
                    val = None
                if val is None:
                    ntext.append(node)
                else:
                    label = "not set" if np.isnan(val) else f"{val:.3g}"
                    ntext.append(f"{node}<br>V = {label}")
            traces.append(go.Scatter3d(
                x=nx_, y=ny_, z=nz_, mode="markers",
                marker=dict(size=5, color=style["color"]),
                name=style["name"], text=ntext, hoverinfo="text",
            ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"Crossbar topology ({self.n}x{self.m}x{self.p})",
            showlegend=True,
            scene=dict(
                dragmode="orbit",
                camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.5, y=1.5, z=0.8),
                            projection=dict(type="orthographic")),
                xaxis_title="X (j) [Columns]",
                yaxis_title="Y (i) [Rows]",
                zaxis_title="Z (k) [Layers]",
                bgcolor="white", aspectmode="data",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig.show()
        return fig


def test_2x2_math():
    """Definitively proves the generalized KCL solver matches a hand-derived 2x2 matrix."""
    print(">>> Running strict 2x2 mathematical verification...")
    
    # 1. Setup a specific scenario
    cb = Tbar(n=2, m=2, p=1, S=10, L=10, GW=100)
    
    cb.set_conductance(0, 0, 0, 50)
    cb.set_conductance(0, 1, 0, 60)
    cb.set_conductance(1, 0, 0, 70)
    cb.set_conductance(1, 1, 0, 80)
    
    cb.set_input_voltages([[5.0], [2.0]])
    cb.set_output_voltages([[0.0], [1.0]])
    
    cb.solve()
    
    # 2. Hand-derive the exact 8x8 Conductance Matrix
    S, L, GW = cb.S, cb.L, cb.GW
    G00, G01, G10, G11 = 50, 60, 70, 80
    
    G_hand = np.array([
        #W00                B00               W01               B01               W10               B10               W11               B11
        [S + GW + G00,      -G00,             -GW,              0,                0,                0,                0,                0],             # Node W00 
        [-G00,              GW + G00,         0,                0,                0,                -GW,              0,                0],             # Node B00 (FIXED: removed G10)
        [-GW,               0,                GW + G01,         -G01,             0,                0,                0,                0],             # Node W01 
        [0,                 0,                -G01,             GW + G01,         0,                0,                0,                -GW],           # Node B01 (FIXED: removed G11)
        [0,                 0,                0,                0,                S + GW + G10,     -G10,             -GW,              0],             # Node W10 
        [0,                 -GW,              0,                0,                -G10,             L + GW + G10,     0,                0],             # Node B10 
        [0,                 0,                0,                0,                -GW,              0,                GW + G11,         -G11],          # Node W11 
        [0,                 0,                0,                -GW,              0,                0,                -G11,             L + GW + G11]   # Node B11 
    ], dtype=float)
    
    I_hand = np.array([
        5.0 * S,  # W00 gets VIn[0]
        0,        # B00
        0,        # W01
        0,        # B01
        2.0 * S,  # W10 gets VIn[1]
        0.0 * L,  # B10 gets VOut[0]
        0,        # W11
        1.0 * L   # B11 gets VOut[1]
    ], dtype=float)
    
    V_hand = np.linalg.solve(G_hand, I_hand)
    
    # 3. Compare the results
    nodes = ["W(0,0,0)", "B(0,0,0)", "W(0,1,0)", "B(0,1,0)", 
             "W(1,0,0)", "B(1,0,0)", "W(1,1,0)", "B(1,1,0)"]
    
    for idx, node in enumerate(nodes):
        v_algo = cb.get_voltage(node)
        v_math = V_hand[idx]
        assert abs(v_algo - v_math) < 1e-8, f"Mismatch at {node}: Algo={v_algo}, Math={v_math}"
        
    print("    SUCCESS: Your generalized solver perfectly matches the exact physical hand-derived matrix!")

if __name__ == "__main__":
    cb = Tbar(n=3, m=3, p=3)
    cb.randomize_conductances(low=10, high=100, seed=42)
    cb.show()