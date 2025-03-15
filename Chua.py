import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def chua_system(t, state, alpha, beta, a, b):
    x, y, z = state  
    fx = b*x + 0.5*(a-b)*(abs(x+1) - abs(x-1))  
    
    # Chua's circuit equations
    dx = alpha * (y - x - fx)  
    dy = x - y + z  
    dz = -beta * y  

    return [dx, dy, dz]

# Function 
def plot_chua_attractor(alpha, beta, a, b):
    state0 = [0.2, 0.3, -0.1]  
    t_span = (0, 100)  # Time span 
    t_eval = np.linspace(*t_span, 10000)  

    # Solve ODE
    sol = solve_ivp(chua_system, t_span, state0, args=(alpha, beta, a, b), t_eval=t_eval)
    x, y, z = sol.y 

    # Plot the chaotic attractor
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=t_eval, cmap='plasma', s=0.3)
    ax.set_xlabel("Voltage C1 (x)")
    ax.set_ylabel("Voltage C2 (y)")
    ax.set_zlabel("Inductor Current (z)")
    ax.set_title(f"Chua's Circuit Attractor (α={alpha}, β={beta}, a={a}, b={b})")
    fig.colorbar(scatter, ax=ax, label="Time")

    st.pyplot(fig)  

# Streamlit UI
st.title("Math 273 Project: Chua's Circuit Attractor Visualization")

# Display the Chua Circuit equations
st.latex(r"""
\begin{aligned}
\frac{dx}{dt} &= \alpha (y - x - f(x)) \\
\frac{dy}{dt} &= x - y + z \\
\frac{dz}{dt} &= -\beta y
\end{aligned}
""")
st.latex(r"""
f(x) = bx + \frac{1}{2}(a-b)(|x+1| - |x-1|)
""")

st.write("""
**Parameter meanings in Chua's Circuit:**
- **x** → Voltage across capacitor **C1**
- **y** → Voltage across capacitor **C2**
- **z** → Current through the inductor **L**
- **α (alpha)** → Related to **resistor values** in the circuit
- **β (beta)** → Ratio of **capacitors** \( C_1 / C_2 \)
- **a, b** → Define the nonlinear behavior of **Chua's diode**
""")


st.sidebar.header("Adjust Circuit Parameters")

alpha = st.sidebar.slider("α (Conductance-related)", min_value=5.0, max_value=20.0, value=9.0, step=0.1)
beta = st.sidebar.slider("β (Capacitor ratio C1/C2)", min_value=10.0, max_value=30.0, value=14.0, step=0.1)
a = st.sidebar.slider("a (Diode parameter)", min_value=-2.0, max_value=2.0, value=-1.27, step=0.01)
b = st.sidebar.slider("b (Diode parameter)", min_value=-1.0, max_value=1.0, value=-0.68, step=0.01)

# Generate the plot
plot_chua_attractor(alpha, beta, a, b)
